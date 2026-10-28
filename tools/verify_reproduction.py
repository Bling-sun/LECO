#!/usr/bin/env python3
"""Build and exercise the public LECO pipeline without using another workspace."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ros-domain-id', type=int, default=187)
    parser.add_argument('--skip-build', action='store_true', help='use this clone\'s existing build/install')
    args = parser.parse_args()
    if not 0 <= args.ros_domain_id <= 232:
        parser.error('ROS domain must be in [0, 232]')
    root = Path(__file__).resolve().parents[1]
    nt = root / 'neural_tracker'
    python = nt / '.venv/bin/python'
    if not python.exists():
        parser.error('create neural_tracker/.venv and install requirements-lock.txt first')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + str(os.getpid())
    output = root / '.runtime' / ('reproduction_' + stamp)
    output.mkdir(parents=True)
    env = os.environ.copy()
    for name in ('AMENT_PREFIX_PATH', 'COLCON_PREFIX_PATH', 'CMAKE_PREFIX_PATH',
                 'PYTHONPATH', 'LD_LIBRARY_PATH', 'VIRTUAL_ENV', 'PYTHONHOME',
                 'ROS_LOCALHOST_ONLY', 'ROS_STATIC_PEERS'):
        env.pop(name, None)
    env.update(ROS_DOMAIN_ID=str(args.ros_domain_id), ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MPLCONFIGDIR=str(output / 'matplotlib'),
               CMAKE_BUILD_PARALLEL_LEVEL='2')
    steps = []
    owned_sessions = []

    def run(label, command, *, cwd=None, ros=False, workspace=True, timeout=180):
        command = [str(item) for item in command]
        if ros:
            setup = 'source /opt/ros/jazzy/setup.bash'
            if workspace:
                setup += ' && source ' + shlex.quote(str(root / 'install/setup.bash'))
            command = ['/bin/bash', '--noprofile', '--norc', '-c',
                       setup + ' && exec ' + shlex.join(command)]
        print(f'[{label}]', flush=True)
        log = output / (label + '.log')
        with log.open('w') as stream:
            process = subprocess.Popen(command, cwd=cwd or root, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        steps.append({'step': label, 'returncode': code})
        if code:
            print(log.read_text()[-6000:], file=sys.stderr)
            raise RuntimeError(f'{label} failed with exit {code}; see {log}')
        return log

    def capture_session(session, label):
        panes = subprocess.check_output(
            ['tmux', 'list-panes', '-s', '-t', '=' + session, '-F', '#{pane_id}'], text=True)
        text = '\n'.join(subprocess.check_output(
            ['tmux', 'capture-pane', '-p', '-S', '-300', '-t', pane], text=True)
            for pane in panes.splitlines())
        (output / (label + '.log')).write_text(text)

    launcher = root / 'start_tracking.sh'
    report = {'date_utc': stamp, 'platform': platform.platform(),
              'python': platform.python_version(), 'ros_domain_id': args.ros_domain_id}
    try:
        run('model_checksums', ['sha256sum', '-c', 'neural_tracker/models/SHA256SUMS'])
        run('fixture_checksums', ['sha256sum', '-c', 'examples/repro_data/SHA256SUMS'], cwd=nt)
        if not args.skip_build:
            run('build', ['colcon', 'build', '--base-paths', 'src', '--packages-select',
                         'simulator', 'tracker', 'experiment', '--executor', 'sequential',
                         '--cmake-args', '-DCMAKE_BUILD_TYPE=Release'],
                ros=True, workspace=False, timeout=600)
        run('ros_tests', ['colcon', 'test', '--base-paths', 'src', '--packages-select',
                         'simulator', 'tracker', 'experiment', '--executor', 'sequential'], ros=True)
        run('ros_test_results', ['colcon', 'test-result', '--verbose'], ros=True)
        run('model_tests', [python, '-m', 'unittest', 'test_regressions.py'], cwd=nt)
        run('collect', ['bash', nt / 'scripts/collect_expert_data_ros.sh',
                        '--seeds', '1', '--start-seed', '100', '--map-styles', 'blocks',
                        '--samples-per-seed', '2', '--episode-seconds', '12', '--sample-hz', '10',
                        '--startup-wait', '2', '--save-dir', output / 'raw',
                        '--log-dir', output / 'collection_logs', '--no-rviz'], ros=True)
        run('quality', [python, nt / 'scripts/filter_expert_visibility.py',
                        '--data_dir', output / 'raw', '--recursive'])
        run('merge', [python, nt / 'scripts/merge_expert_data.py', '--input_dir', output / 'raw',
                      '--recursive', '--output_dir', output / 'dataset', '--val_ratio', '0.25',
                      '--seed', '42', '--split-unit', 'episode'])
        run_name = 'reproduce_' + stamp
        model = nt / 'models' / run_name
        run('train', [python, 'train.py', '--data-dir', output / 'dataset', '--run-name', run_name,
                      '--epochs', '1', '--batch-size', '4', '--num-workers', '0', '--seed', '42',
                      '--max-train-batches', '2', '--max-val-batches', '2', '--device', 'cpu',
                      '--disable-tqdm'], cwd=nt)
        run('evaluate', [python, 'scripts/eval_models.py', '--models', model / 'best.pth',
                         '--datasets', output / 'dataset/val.pkl', '--batch-size', '4',
                         '--num-workers', '0', '--max-batches', '2', '--device', 'cpu'], cwd=nt)
        run('export', [python, 'scripts/export_onnx.py', '--model', model / 'best.pth',
                       '--output', model / 'policy.onnx'], cwd=nt)
        run('random_route', [root / 'generate_random_goals.sh', 'maze2', output / 'random_goals.json',
                             '--count', '5', '--seed', '42'])
        for backend in ('neural', 'traditional', 'python'):
            session = f'repro_{backend}_{os.getpid()}'
            owned_sessions.append(session)
            run('start_' + backend, [launcher, '--backend', backend, '--session', session,
                                     '--no-rviz'], ros=True)
            goal = {'header': {'frame_id': 'world'}, 'pose': {
                'position': {'x': -5.52, 'y': 8.08} if backend == 'python' else {'x': -2.0, 'y': 0.0},
                'orientation': {'w': 1.0}}}
            run('goal_' + backend, ['ros2', 'topic', 'pub', '--once', '/goal_pose',
                                   'geometry_msgs/msg/PoseStamped', json.dumps(goal)], ros=True)
            time.sleep(3)
            capture_session(session, backend + '_panes')
            run('stop_' + backend, [launcher, '--session', session, '--stop'])
            owned_sessions.remove(session)
        goals = output / 'formal_goals.json'
        goals.write_text(json.dumps({'format': 'neural-tracker-goal-sequence', 'version': 1,
            'topic': '/goal_pose', 'goals': [{'index': 1, 'time': 0.0, 'frame_id': 'world',
            'position': {'x': -2.0, 'y': 0.0, 'z': 0.0},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}}]}))
        for label, policy in (('published', nt / 'models/v41_directional_recovery/policy.onnx'),
                               ('new_model', model / 'policy.onnx')):
            experiment_root = output / label
            run('experiment_' + label, ['ros2', 'run', 'experiment', 'neural_tracker_experiment',
                goals, '--run-label', 'reproduction_' + label, '--algorithm-label', label,
                '--experiment-root', experiment_root, '--map-file', 'src/simulator/map/maze2.json',
                '--policy-file', policy, '--route-timeout', '40', '--startup-timeout', '20'], ros=True)
            matches = list(experiment_root.rglob('metrics.json'))
            if len(matches) != 1:
                raise RuntimeError(f'missing unique metrics for {label}')
            metrics = json.loads(matches[0].read_text())
            neural_cycles = sum(t['branch_usage']['counts'].get('neural', 0) for t in metrics['trials'])
            if not metrics['validation']['valid'] or neural_cycles == 0:
                raise RuntimeError(f'invalid experiment or no ONNX inference: {label}')
            report[label] = {'valid': True, 'control_samples': metrics['control']['samples_K'],
                             'neural_cycles': neural_cycles,
                             'trial_statuses': [t['status'] for t in metrics['trials']]}
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=str(error))
        print(str(error), file=sys.stderr)
    finally:
        for session in owned_sessions:
            subprocess.run([str(launcher), '--session', session, '--stop'], env=env, cwd=root)
        report['steps'] = steps
        (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f"{report['status']}: {output / 'summary.json'}", flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

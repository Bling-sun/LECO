"""Merge ROS episode pickles with reproducible, episode-disjoint splits.

Use --recursive for the per-map directories produced by the collector.
Use --split-unit frame only to reproduce the legacy frame-level split.
"""
import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path


def merge(input_dir, output_dir, val_ratio=0.1, seed=42, recursive=False,
          split_unit='episode'):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not 0 < val_ratio < 1:
        raise ValueError('val_ratio must be between 0 and 1')
    paths = sorted(input_dir.rglob('sample_*.pkl') if recursive
                   else input_dir.glob('*.pkl'))
    if not paths:
        raise ValueError(f'No episode pickle files found in {input_dir}')
    if any((output_dir / name).exists() for name in ('train.pkl', 'val.pkl', 'split_manifest.json')):
        raise ValueError(f'Output already exists in {output_dir}; choose a new directory')
    episodes = []
    for path in paths:
        with path.open('rb') as stream:
            samples = pickle.load(stream)
        if not isinstance(samples, list) or not samples or not all(isinstance(s, dict) for s in samples):
            raise ValueError(f'Expected a nonempty list of sample dictionaries: {path}')
        episodes.append((path, samples))
    rng = random.Random(seed)
    if split_unit == 'episode':
        if len(episodes) < 2:
            raise ValueError('Episode split requires at least two episode files')
        rng.shuffle(episodes)
        n_val = min(len(episodes) - 1, max(1, int(len(episodes) * val_ratio)))
        groups = {'val': episodes[:n_val], 'train': episodes[n_val:]}
        datasets = {name: [s for _, samples in group for s in samples]
                    for name, group in groups.items()}
    elif split_unit == 'frame':
        samples = [s for _, frames in episodes for s in frames]
        if len(samples) < 2:
            raise ValueError('Frame split requires at least two samples')
        rng.shuffle(samples)
        n_val = min(len(samples) - 1, max(1, int(len(samples) * val_ratio)))
        datasets = {'val': samples[:n_val], 'train': samples[n_val:]}
        groups = {'val': episodes, 'train': episodes}
    else:
        raise ValueError(f'Unknown split unit: {split_unit}')
    manifest = {'seed': seed, 'val_ratio': val_ratio, 'split_unit': split_unit,
                'inputs': [{'path': str(p.relative_to(input_dir)),
                            'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                            'frames': len(samples)} for p, samples in episodes],
                'splits': {name: {'frames': len(datasets[name]),
                                 'episodes': [str(p.relative_to(input_dir)) for p, _ in group]}
                           for name, group in groups.items()}}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, samples in datasets.items():
        with (output_dir / f'{name}.pkl').open('wb') as stream:
            pickle.dump(samples, stream)
        print(f'{name}.pkl: {len(samples)} frames')
    (output_dir / 'split_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', default='./data/expert_ros')
    parser.add_argument('--output_dir', default='./data/scenarios')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--split-unit', choices=['episode', 'frame'], default='episode')
    args = parser.parse_args()
    try:
        merge(args.input_dir, args.output_dir, args.val_ratio, args.seed,
              args.recursive, args.split_unit)
    except (ValueError, OSError) as error:
        parser.exit(1, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()

cmake_minimum_required(VERSION 3.8)
project(experiment)

find_package(ament_cmake REQUIRED)

# This workspace coexists with Eva/Elastic workspaces that also install a ROS
# package named "experiment".  The normal ament helper de-duplicates prefixes
# without moving an existing prefix to the front, so sourcing this workspace a
# second time would otherwise leave the other package selected.
ament_environment_hooks(env-hooks/experiment_prefix_priority.sh)

install(PROGRAMS
  scripts/goal_pose_sequence.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME goal_pose_sequence
)
install(PROGRAMS
  scripts/neural_tracker_metrics.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME neural_tracker_metrics
)
install(PROGRAMS
  scripts/traditional_tracker_metrics.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME traditional_tracker_metrics
)
install(PROGRAMS
  scripts/neural_tracker_results.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME neural_tracker_results
)
install(PROGRAMS
  scripts/traditional_tracker_results.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME traditional_tracker_results
)
install(PROGRAMS
  scripts/neural_tracker_fov_heatmap.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME neural_tracker_fov_heatmap
)
install(PROGRAMS
  scripts/traditional_tracker_fov_heatmap.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME traditional_tracker_fov_heatmap
)
install(PROGRAMS
  scripts/neural_tracker_experiment.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME neural_tracker_experiment
)
install(PROGRAMS
  scripts/random_goal_sequence.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME random_goal_sequence
)
install(PROGRAMS
  scripts/current_target_reference.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME current_target_reference
)
install(PROGRAMS
  scripts/elastic_target_reference_replay.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME elastic_target_reference_replay
)
install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(experiment_scripts_test test/test_experiment_scripts.py)
endif()

ament_package()

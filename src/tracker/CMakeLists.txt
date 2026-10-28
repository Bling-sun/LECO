cmake_minimum_required(VERSION 3.8)
project(tracker)

# Enable O3 optimization
add_compile_options(-O3 -Wall -Wextra -Wpedantic)

# Find packages
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)

if(BUILD_TESTING)
    find_package(ament_cmake_pytest REQUIRED)
    ament_add_pytest_test(
        target_observation_mux_test
        test/test_target_observation_mux.py
    )
endif()

# ONNX Runtime (C++) for the neural tracker variant. Prebuilt package lives in
# <workspace>/third_party/onnxruntime-linux-x64-<ver>. Override with -DONNXRUNTIME_ROOT.
if(NOT DEFINED ONNXRUNTIME_ROOT)
    file(GLOB _ORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/onnxruntime-linux-*")
    if(_ORT_DIRS)
        list(GET _ORT_DIRS 0 ONNXRUNTIME_ROOT)
    endif()
endif()
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}/include")
find_library(ONNXRUNTIME_LIB onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}/lib")

# Ament specific configuration
ament_package()

# Specify additional locations of header files
include_directories(include)

# Declare C++ executables
add_executable(tracker
    src/planning.cpp
    src/planner/map.cpp
    src/planner/mpc.cpp
    src/planner/astar.cpp
    src/predictor/bezier.cpp
)
add_executable(prediction
    src/prediction.cpp
    src/predictor/bezier.cpp
    src/predictor/predictor.cpp
)

# Neural tracker variant (ONNX policy for the target-visible branch)
if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
    add_executable(tracker_neural
        src/planning_neural.cpp
        src/planner/map.cpp
        src/planner/mpc.cpp
        src/planner/astar.cpp
        src/planner/neural_policy.cpp
        src/predictor/bezier.cpp
    )
    # ONNX Runtime headers use constructs -Wpedantic flags as errors on; relax.
    target_compile_options(tracker_neural PRIVATE -Wno-pedantic)
    target_include_directories(tracker_neural PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    ament_target_dependencies(tracker_neural
        rclcpp
        OpenCV
        nav_msgs
        sensor_msgs
        geometry_msgs
        std_msgs
        pcl_conversions
    )
    target_link_libraries(tracker_neural ${ONNXRUNTIME_LIB})
    # Embed the onnxruntime lib dir as rpath so the loader finds libonnxruntime.so.1
    # at runtime (it lives in third_party, not a system path).
    get_filename_component(_ORT_LIBDIR "${ONNXRUNTIME_LIB}" DIRECTORY)
    set_target_properties(tracker_neural PROPERTIES
        BUILD_RPATH "${_ORT_LIBDIR}"
        INSTALL_RPATH "${_ORT_LIBDIR}")
    install(TARGETS tracker_neural DESTINATION lib/${PROJECT_NAME})

    # Offline parity test (not installed; run manually).
    add_executable(parity_test
        test/parity_test.cpp
        src/planner/map.cpp
        src/planner/neural_policy.cpp
    )
    target_compile_options(parity_test PRIVATE -Wno-pedantic)
    target_compile_definitions(parity_test PRIVATE
        DEFAULT_ONNX_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../neural_tracker/models/v41_directional_recovery/policy.onnx")
    target_include_directories(parity_test PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    ament_target_dependencies(parity_test rclcpp OpenCV nav_msgs geometry_msgs pcl_conversions)
    target_link_libraries(parity_test ${ONNXRUNTIME_LIB})
    set_target_properties(parity_test PROPERTIES
        BUILD_RPATH "${_ORT_LIBDIR}" INSTALL_RPATH "${_ORT_LIBDIR}")
else()
    message(WARNING
        "ONNX Runtime not found; skipping tracker_neural. "
        "Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-<ver>")
endif()

# Specify library to link a library or executable target against
ament_target_dependencies(tracker
    rclcpp 
    OpenCV
    nav_msgs
    sensor_msgs
    geometry_msgs
    pcl_conversions
)
ament_target_dependencies(prediction
    rclcpp
    nav_msgs
    geometry_msgs
)

# Copy files for installation
install(TARGETS tracker
    DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS prediction
    DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS scripts/target_observation_mux.py
    DESTINATION lib/${PROJECT_NAME}
    RENAME target_observation_mux
)
install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY config
    DESTINATION share/${PROJECT_NAME}
)

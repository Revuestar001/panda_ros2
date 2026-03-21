cmake_minimum_required(VERSION 3.8)
project(panda_ocs2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(panda_interfaces REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem log_setup log)

find_package(ocs2_core REQUIRED)
find_package(ocs2_oc REQUIRED)
find_package(ocs2_ddp REQUIRED)
find_package(ocs2_mpc REQUIRED)
find_package(ocs2_ros_interfaces REQUIRED)
find_package(ocs2_robotic_tools REQUIRED)
find_package(ocs2_pinocchio_interface REQUIRED)
find_package(ocs2_mobile_manipulator REQUIRED)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(panda_ocs2_node
  src/panda_ocs2_node.cpp
)

add_executable(panda_arm_nmpc_node
  src/PandaArmNmpcInterface.cpp
  src/PandaArmNmpcNode.cpp
)

target_include_directories(panda_ocs2_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
  ${pinocchio_INCLUDE_DIRS}
)

target_include_directories(panda_arm_nmpc_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
  ${pinocchio_INCLUDE_DIRS}
)

ament_target_dependencies(panda_ocs2_node
  rclcpp
  std_msgs
  sensor_msgs
  trajectory_msgs
  geometry_msgs
  panda_interfaces
  ocs2_core
  ocs2_oc
  ocs2_mpc
  ocs2_ros_interfaces
  ocs2_robotic_tools
)

ament_target_dependencies(panda_arm_nmpc_node
  rclcpp
  std_msgs
  sensor_msgs
  trajectory_msgs
  geometry_msgs
  panda_interfaces
  ocs2_core
  ocs2_oc
  ocs2_ddp
  ocs2_mpc
  ocs2_ros_interfaces
  ocs2_robotic_tools
  ocs2_pinocchio_interface
  ocs2_mobile_manipulator
)

target_link_libraries(panda_ocs2_node
  Eigen3::Eigen
  pinocchio::pinocchio
)

target_link_libraries(panda_arm_nmpc_node
  Eigen3::Eigen
  pinocchio::pinocchio
)

if(DEFINED OCS2_CXX_FLAGS)
  target_compile_options(panda_ocs2_node PRIVATE ${OCS2_CXX_FLAGS})
endif()

if(DEFINED OCS2_PINOCCHIO_FLAGS)
  target_compile_options(panda_arm_nmpc_node PRIVATE ${OCS2_PINOCCHIO_FLAGS})
elseif(DEFINED OCS2_CXX_FLAGS)
  target_compile_options(panda_arm_nmpc_node PRIVATE ${OCS2_CXX_FLAGS})
endif()

target_compile_features(panda_ocs2_node PUBLIC cxx_std_17)
target_compile_features(panda_arm_nmpc_node PUBLIC cxx_std_17)

install(TARGETS panda_ocs2_node panda_arm_nmpc_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

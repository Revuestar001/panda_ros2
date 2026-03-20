cmake_minimum_required(VERSION 3.8)
project(panda_moveit)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tinyxml2_vendor REQUIRED)
find_package(TinyXML2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(trajectory_msgs REQUIRED)

add_executable(global_planning src/global_planning.cpp)
target_include_directories(global_planning PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(global_planning PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  global_planning
  "geometry_msgs"
  "moveit_msgs"
  "rclcpp"
  "moveit_ros_planning_interface"
  "std_srvs"
  "tinyxml2_vendor"
  "tf2_geometry_msgs"
  "tf2_ros"
  "trajectory_msgs"
)
target_link_libraries(global_planning tinyxml2::tinyxml2)

add_executable(goal_commander src/goal_commander.cpp)
target_include_directories(goal_commander PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(goal_commander PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  goal_commander
  "geometry_msgs"
  "rclcpp"
  "std_srvs"
)

add_executable(static_obstacles_publisher src/static_obstacles_publisher.cpp)
target_include_directories(static_obstacles_publisher PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(static_obstacles_publisher PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  static_obstacles_publisher
  "geometry_msgs"
  "moveit_msgs"
  "rclcpp"
  "shape_msgs"
  "tinyxml2_vendor"
)
target_link_libraries(static_obstacles_publisher tinyxml2::tinyxml2)

install(TARGETS global_planning goal_commander static_obstacles_publisher
  DESTINATION lib/${PROJECT_NAME})

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME})
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

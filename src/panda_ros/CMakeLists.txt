cmake_minimum_required(VERSION 3.8)
project(panda_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

find_package(pinocchio REQUIRED)
find_package(mujoco REQUIRED)
find_package(glfw3 REQUIRED)

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

add_executable(panda_test_node src/panda_test_node.cpp)
ament_target_dependencies(panda_test_node
  rclcpp
)

target_link_libraries(panda_test_node 
    Eigen3::Eigen
    mujoco::mujoco
    glfw
    pinocchio::pinocchio
)

install(TARGETS panda_test_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  launch
  model
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

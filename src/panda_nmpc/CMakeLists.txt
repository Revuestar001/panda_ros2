cmake_minimum_required(VERSION 3.8)
project(panda_nmpc)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. 查找 ROS 2 依赖
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(tinyxml2_vendor REQUIRED)
find_package(TinyXML2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)

find_package(panda_interfaces REQUIRED)

# ==================== [新增] Acados & 求解器配置 ====================

# 确保环境变量 ACADOS_SOURCE_DIR 已设置
set(ACADOS_DIR $ENV{ACADOS_SOURCE_DIR})
if(NOT ACADOS_DIR)
  message(FATAL_ERROR "未找到 ACADOS_SOURCE_DIR 环境变量！请先执行: export ACADOS_SOURCE_DIR=/你的/acados/根目录")
endif()

# 指定自动生成的 C 代码所在目录 (假设 c_generated_code 放在当前 package 的根目录)
set(GEN_CODE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nmpc_model_gen)

# 设置 Acados 相关的头文件路径
set(ACADOS_INCLUDE_DIRS
  ${ACADOS_DIR}/include
  ${ACADOS_DIR}/include/acados
  ${ACADOS_DIR}/include/blasfeo/include
  ${ACADOS_DIR}/include/hpipm/include
  ${GEN_CODE_DIR}/c_generated_code
)

include_directories(/home/cyh/panda_ros2/src/panda_nmpc/include/panda_nmpc)
include_directories(/home/cyh/panda_ros2/install/panda_interfaces/include/panda_interfaces)

# 查找 Acados 核心依赖库
find_library(ACADOS_LIB acados PATHS ${ACADOS_DIR}/lib REQUIRED)
find_library(HPIPM_LIB hpipm PATHS ${ACADOS_DIR}/lib REQUIRED)
find_library(BLASFEO_LIB blasfeo PATHS ${ACADOS_DIR}/lib REQUIRED)

# 直接用当前 workspace 中最新的 generated C 源码构建 solver 共享库，
# 避免“头文件/生成源码已经更新，但旧的 .so 还没重新 make”导致的参数维度错位。
file(GLOB_RECURSE GENERATED_C_SOURCES CONFIGURE_DEPENDS
  "${GEN_CODE_DIR}/c_generated_code/*.c"
  "${GEN_CODE_DIR}/c_generated_code/**/*.c")
list(FILTER GENERATED_C_SOURCES EXCLUDE REGEX "main_.*\\.c$")

add_library(panda_task_space_nmpc_generated_solver SHARED
  ${GENERATED_C_SOURCES}
)
target_include_directories(panda_task_space_nmpc_generated_solver PUBLIC
  ${ACADOS_INCLUDE_DIRS}
)
# acados 生成的 Makefile 默认会用 -O2 编译 external functions。
# 这里直接把 generated C 挂进 package 的 CMake 后，如果 workspace 没显式
# 指定 Release/RelWithDebInfo，就会退化成“无优化”编译，SQP-RTI 会明显变慢。
target_compile_options(panda_task_space_nmpc_generated_solver PRIVATE
  $<$<COMPILE_LANGUAGE:C>:-O2>
)
set_target_properties(panda_task_space_nmpc_generated_solver PROPERTIES
  C_STANDARD 99
  C_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME acados_solver_panda_task_space_nmpc
)
target_link_libraries(panda_task_space_nmpc_generated_solver
  ${ACADOS_LIB}
  ${HPIPM_LIB}
  ${BLASFEO_LIB}
  m
)

# ===================================================================

# 2. 编译可执行文件
add_executable(nmpc_tau 
  src/nmpc_tau.cpp
)

add_executable(nmpc_tau_original
  src/nmpc_tau_original.cpp
)

add_executable(dynamic_sphere_obstacle_publisher
  src/dynamic_sphere_obstacle_publisher.cpp
)

# 包含所有的头文件路径
target_include_directories(nmpc_tau PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${ACADOS_INCLUDE_DIRS}
)

target_include_directories(nmpc_tau_original PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${ACADOS_INCLUDE_DIRS}
)

target_include_directories(dynamic_sphere_obstacle_publisher PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(nmpc_tau PUBLIC c_std_99 cxx_std_17) 
target_compile_features(nmpc_tau_original PUBLIC c_std_99 cxx_std_17)
target_compile_features(dynamic_sphere_obstacle_publisher PUBLIC cxx_std_17)

# 3. 链接目标与库
ament_target_dependencies(nmpc_tau 
  "ament_index_cpp"
  "geometry_msgs"
  "rclcpp"
  std_msgs
  sensor_msgs
  panda_interfaces
  "trajectory_msgs"
  "tinyxml2_vendor"
)

ament_target_dependencies(nmpc_tau_original
  "ament_index_cpp"
  "geometry_msgs"
  "rclcpp"
  std_msgs
  sensor_msgs
  panda_interfaces
  "trajectory_msgs"
)

ament_target_dependencies(dynamic_sphere_obstacle_publisher
  "rclcpp"
  panda_interfaces
)

# 链接 Acados 库和生成的求解器动态库
target_link_libraries(nmpc_tau 
  ${ACADOS_LIB}
  ${HPIPM_LIB}
  ${BLASFEO_LIB}
  panda_task_space_nmpc_generated_solver
  Eigen3::Eigen
  pinocchio::pinocchio
  tinyxml2::tinyxml2
)

target_link_libraries(nmpc_tau_original
  ${ACADOS_LIB}
  ${HPIPM_LIB}
  ${BLASFEO_LIB}
  panda_task_space_nmpc_generated_solver
  Eigen3::Eigen
  pinocchio::pinocchio
)

# [关键技巧] 设置运行时 RPATH，防止 ros2 run 时报 "error while loading shared libraries"
set_target_properties(nmpc_tau PROPERTIES 
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "${ACADOS_DIR}/lib;${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}/c_generated_code"
)

set_target_properties(nmpc_tau_original PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "${ACADOS_DIR}/lib;${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}/c_generated_code"
)

# 4. 安装规则
# 安装可执行文件
install(TARGETS nmpc_tau nmpc_tau_original
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS dynamic_sphere_obstacle_publisher
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS panda_task_space_nmpc_generated_solver
  DESTINATION lib/${PROJECT_NAME}/c_generated_code)

install(PROGRAMS
  scripts/sync_static_sphere_scene.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

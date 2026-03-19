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

# 查找你刚刚生成的求解器动态库 (需提前在 c_generated_code 目录下执行 make shared_lib)
# 注意：库的名字必须和你在 Python 里设定的 acados_model.name 匹配
find_library(SOLVER_LIB acados_solver_panda_task_space_nmpc PATHS ${GEN_CODE_DIR}/c_generated_code REQUIRED)

# ===================================================================

# 2. 编译可执行文件
add_executable(nmpc_tau 
  src/nmpc_tau.cpp
)

add_executable(nmpc_tau_original
  src/nmpc_tau_original.cpp
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

target_compile_features(nmpc_tau PUBLIC c_std_99 cxx_std_17) 
target_compile_features(nmpc_tau_original PUBLIC c_std_99 cxx_std_17)

# 3. 链接目标与库
ament_target_dependencies(nmpc_tau 
  "ament_index_cpp"
  "geometry_msgs"
  "rclcpp"
  std_msgs
  sensor_msgs
  panda_interfaces
  "trajectory_msgs"
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

# 链接 Acados 库和生成的求解器动态库
target_link_libraries(nmpc_tau 
  ${ACADOS_LIB}
  ${HPIPM_LIB}
  ${BLASFEO_LIB}
  ${SOLVER_LIB}
  Eigen3::Eigen
  pinocchio::pinocchio
)

target_link_libraries(nmpc_tau_original
  ${ACADOS_LIB}
  ${HPIPM_LIB}
  ${BLASFEO_LIB}
  ${SOLVER_LIB}
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

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

# 安装自动生成的动态库，确保 ROS 2 install 空间里有 .so 文件可用
install(DIRECTORY ${GEN_CODE_DIR}/c_generated_code/
  DESTINATION lib/${PROJECT_NAME}/c_generated_code
  FILES_MATCHING PATTERN "*.so"
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

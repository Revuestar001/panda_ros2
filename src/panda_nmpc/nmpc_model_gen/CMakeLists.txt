cmake_minimum_required(VERSION 3.10)
project(test_panda_nmpc)

set(CMAKE_CXX_STANDARD 17)

# ====== 注意：请将这里的路径修改为你实际的 acados 安装路径 ======
# 如果你是按照官方教程装在 home 目录下，通常是下面这样
set(ACADOS_DIR $ENV{HOME}/acados)

# 头文件包含
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code
    ${ACADOS_DIR}/include
    ${ACADOS_DIR}/include/acados
    ${ACADOS_DIR}/include/blasfeo/include
    ${ACADOS_DIR}/include/hpipm/include
)

# 库文件目录
link_directories(
    ${ACADOS_DIR}/lib
)

# 获取生成的 C 源码文件 (排除掉它自己生成的 main.c 测试文件以免冲突)
file(GLOB GENERATED_C_SOURCES "c_generated_code/*.c" "c_generated_code/**/*.c")
list(FILTER GENERATED_C_SOURCES EXCLUDE REGEX "main_.*\\.c$")

# 编译我们的 C++ 可执行文件
add_executable(test_nmpc test_nmpc.cpp ${GENERATED_C_SOURCES})

# 链接依赖：acados底层依靠 hpipm(二次规划求解器) 和 blasfeo(线性代数加速)
target_link_libraries(test_nmpc 
    acados 
    hpipm 
    blasfeo 
    m
)
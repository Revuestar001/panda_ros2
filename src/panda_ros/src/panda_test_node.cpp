#include <rclcpp/rclcpp.hpp>
#include <mujoco/mujoco.h>
#include <pinocchio/multibody/model.hpp>
#include <iostream>

int main(int argc, char ** argv)
{
    // 1. 初始化 ROS 2
    rclcpp::init(argc, argv);
    std::cout << "ROS 2 initialized successfully." << std::endl;
    
    // 2. 测试 MuJoCo 链接
    std::cout << "MuJoCo version: " << mj_versionString() << std::endl;
    
    // 3. 测试 Pinocchio 链接
    pinocchio::Model model;
    std::cout << "Pinocchio model initialized successfully. Joint count: " << model.njoints << std::endl;

    rclcpp::shutdown();
    return 0;
}
#include <rclcpp/rclcpp.hpp>

#include <ocs2_core/Types.h>

class PandaOcs2Node final : public rclcpp::Node {
public:
  PandaOcs2Node() : Node("panda_ocs2_node") {
    const ocs2::scalar_t warm_start_time = 0.0;
    RCLCPP_INFO(
      get_logger(),
      "panda_ocs2 package is ready. OCS2 scalar_t warm start time = %.1f",
      static_cast<double>(warm_start_time));
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PandaOcs2Node>());
  rclcpp::shutdown();
  return 0;
}

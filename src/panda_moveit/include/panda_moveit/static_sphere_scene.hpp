#ifndef PANDA_MOVEIT_STATIC_SPHERE_SCENE_HPP_
#define PANDA_MOVEIT_STATIC_SPHERE_SCENE_HPP_

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <tinyxml2.h>

namespace panda_moveit {

struct StaticSphereObstacle {
  std::string name;
  double x{0.0};
  double y{0.0};
  double z{0.0};
  double radius{0.0};
};

inline StaticSphereObstacle parseSphereObstacle(const tinyxml2::XMLElement& body) {
  const tinyxml2::XMLElement* geom = body.FirstChildElement("geom");
  if (geom == nullptr) {
    throw std::runtime_error("Sphere obstacle body has no <geom> child.");
  }

  const char* name_attr = body.Attribute("name");
  const char* pos_attr = body.Attribute("pos");
  const char* size_attr = geom->Attribute("size");
  if (name_attr == nullptr || pos_attr == nullptr || size_attr == nullptr) {
    throw std::runtime_error("Sphere obstacle body must contain name/pos and geom size attributes.");
  }

  StaticSphereObstacle obstacle;
  obstacle.name = name_attr;

  std::istringstream pos_stream(pos_attr);
  if (!(pos_stream >> obstacle.x >> obstacle.y >> obstacle.z)) {
    throw std::runtime_error(
        "Failed to parse position for obstacle '" + obstacle.name + "' from pos='" +
        std::string(pos_attr) + "'.");
  }

  std::istringstream radius_stream(size_attr);
  if (!(radius_stream >> obstacle.radius)) {
    throw std::runtime_error(
        "Failed to parse radius for obstacle '" + obstacle.name + "' from size='" +
        std::string(size_attr) + "'.");
  }

  return obstacle;
}

inline StaticSphereObstacle parseConfigObstacle(const tinyxml2::XMLElement& obstacle_elem) {
  const char* name_attr = obstacle_elem.Attribute("name");
  const char* pos_attr = obstacle_elem.Attribute("pos");
  const char* radius_attr = obstacle_elem.Attribute("radius");
  if (name_attr == nullptr || pos_attr == nullptr || radius_attr == nullptr) {
    throw std::runtime_error(
        "Static obstacle config entry must contain name/pos/radius attributes.");
  }

  StaticSphereObstacle obstacle;
  obstacle.name = name_attr;

  std::istringstream pos_stream(pos_attr);
  if (!(pos_stream >> obstacle.x >> obstacle.y >> obstacle.z)) {
    throw std::runtime_error(
        "Failed to parse position for obstacle '" + obstacle.name + "' from pos='" +
        std::string(pos_attr) + "'.");
  }

  std::istringstream radius_stream(radius_attr);
  if (!(radius_stream >> obstacle.radius)) {
    throw std::runtime_error(
        "Failed to parse radius for obstacle '" + obstacle.name + "' from radius='" +
        std::string(radius_attr) + "'.");
  }

  return obstacle;
}

inline std::vector<StaticSphereObstacle> loadStaticSphereObstaclesFromSceneXml(
    const std::string& scene_xml_path) {
  tinyxml2::XMLDocument doc;
  const tinyxml2::XMLError load_status = doc.LoadFile(scene_xml_path.c_str());
  if (load_status != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error(
        "Failed to load MuJoCo scene XML '" + scene_xml_path + "': " +
        (doc.ErrorStr() == nullptr ? std::string("unknown error") : std::string(doc.ErrorStr())));
  }

  const tinyxml2::XMLElement* root = doc.RootElement();
  if (root == nullptr) {
    throw std::runtime_error("Scene XML '" + scene_xml_path + "' has no root element.");
  }

  if (std::string(root->Name()) == "static_sphere_obstacles") {
    std::vector<StaticSphereObstacle> obstacles;
    for (const tinyxml2::XMLElement* obstacle_elem = root->FirstChildElement("obstacle");
         obstacle_elem != nullptr;
         obstacle_elem = obstacle_elem->NextSiblingElement("obstacle")) {
      obstacles.push_back(parseConfigObstacle(*obstacle_elem));
    }
    return obstacles;
  }

  const tinyxml2::XMLElement* worldbody = root->FirstChildElement("worldbody");
  if (worldbody == nullptr) {
    throw std::runtime_error("Scene XML '" + scene_xml_path + "' has no <worldbody> element.");
  }

  std::vector<StaticSphereObstacle> obstacles;
  for (const tinyxml2::XMLElement* body = worldbody->FirstChildElement("body");
       body != nullptr;
       body = body->NextSiblingElement("body")) {
    const tinyxml2::XMLElement* geom = body->FirstChildElement("geom");
    const char* type_attr = geom == nullptr ? nullptr : geom->Attribute("type");
    if (type_attr == nullptr || std::string(type_attr) != "sphere") {
      continue;
    }
    obstacles.push_back(parseSphereObstacle(*body));
  }

  return obstacles;
}

inline moveit_msgs::msg::CollisionObject toCollisionObject(
    const StaticSphereObstacle& obstacle,
    const std::string& world_frame) {
  moveit_msgs::msg::CollisionObject object;
  object.header.frame_id = world_frame;
  object.id = obstacle.name;
  object.operation = moveit_msgs::msg::CollisionObject::ADD;

  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = shape_msgs::msg::SolidPrimitive::SPHERE;
  primitive.dimensions = {obstacle.radius};

  geometry_msgs::msg::Pose pose;
  pose.position.x = obstacle.x;
  pose.position.y = obstacle.y;
  pose.position.z = obstacle.z;
  pose.orientation.w = 1.0;

  object.primitives.push_back(primitive);
  object.primitive_poses.push_back(pose);
  return object;
}

}  // namespace panda_moveit

#endif  // PANDA_MOVEIT_STATIC_SPHERE_SCENE_HPP_

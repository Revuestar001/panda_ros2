#ifndef PANDA_NMPC_STATIC_SPHERE_SCENE_HPP_
#define PANDA_NMPC_STATIC_SPHERE_SCENE_HPP_

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <tinyxml2.h>

namespace panda_nmpc {

struct StaticSphereObstacle {
    std::string name;
    Eigen::Vector3d center{Eigen::Vector3d::Zero()};
    double radius{0.0};
};

inline Eigen::Vector3d parse_xyz_attribute(const std::string& xyz_text) {
    std::istringstream stream(xyz_text);
    Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
    if (!(stream >> xyz.x() >> xyz.y() >> xyz.z())) {
        throw std::runtime_error("Failed to parse xyz attribute: '" + xyz_text + "'.");
    }
    return xyz;
}

inline StaticSphereObstacle parse_config_obstacle(const tinyxml2::XMLElement& obstacle_elem) {
    const char* name_attr = obstacle_elem.Attribute("name");
    const char* pos_attr = obstacle_elem.Attribute("pos");
    const char* radius_attr = obstacle_elem.Attribute("radius");
    if (name_attr == nullptr || pos_attr == nullptr || radius_attr == nullptr) {
        throw std::runtime_error(
            "Static obstacle config entry must contain name/pos/radius attributes.");
    }

    StaticSphereObstacle obstacle;
    obstacle.name = name_attr;
    obstacle.center = parse_xyz_attribute(pos_attr);

    std::istringstream radius_stream(radius_attr);
    if (!(radius_stream >> obstacle.radius)) {
        throw std::runtime_error(
            "Failed to parse sphere radius for obstacle '" + obstacle.name + "' from radius='" +
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
            obstacles.push_back(parse_config_obstacle(*obstacle_elem));
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
        if (geom == nullptr) {
            continue;
        }

        const char* type_attr = geom->Attribute("type");
        if (type_attr == nullptr || std::string(type_attr) != "sphere") {
            continue;
        }

        const char* name_attr = body->Attribute("name");
        const char* pos_attr = body->Attribute("pos");
        const char* size_attr = geom->Attribute("size");
        if (name_attr == nullptr || pos_attr == nullptr || size_attr == nullptr) {
            throw std::runtime_error(
                "Sphere obstacle entry in '" + scene_xml_path +
                "' must contain body name/pos and geom size attributes.");
        }

        StaticSphereObstacle obstacle;
        obstacle.name = name_attr;
        obstacle.center = parse_xyz_attribute(pos_attr);

        std::istringstream radius_stream(size_attr);
        if (!(radius_stream >> obstacle.radius)) {
            throw std::runtime_error(
                "Failed to parse sphere radius for obstacle '" + obstacle.name + "' from size='" +
                std::string(size_attr) + "'.");
        }
        obstacles.push_back(obstacle);
    }

    return obstacles;
}

}  // namespace panda_nmpc

#endif  // PANDA_NMPC_STATIC_SPHERE_SCENE_HPP_

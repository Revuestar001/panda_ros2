#!/usr/bin/env python3

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


BEGIN_MARKER = "    <!-- BEGIN AUTO-GENERATED STATIC SPHERE OBSTACLES -->"
END_MARKER = "    <!-- END AUTO-GENERATED STATIC SPHERE OBSTACLES -->"


def parse_obstacles(obstacle_config_path: Path):
    tree = ET.parse(obstacle_config_path)
    root = tree.getroot()
    if root.tag != "static_sphere_obstacles":
        raise ValueError(
            f"Obstacle config root must be <static_sphere_obstacles>, got <{root.tag}>."
        )

    obstacles = []
    for obstacle_elem in root.findall("obstacle"):
        name = obstacle_elem.get("name")
        pos = obstacle_elem.get("pos")
        radius = obstacle_elem.get("radius")
        rgba = obstacle_elem.get("rgba", "0.8 0.2 0.2 1")
        if not name or not pos or not radius:
            raise ValueError(
                "Each <obstacle> must define name/pos/radius in "
                f"'{obstacle_config_path}'."
            )
        obstacles.append(
            {
                "name": name,
                "pos": pos,
                "radius": radius,
                "rgba": rgba,
            }
        )
    return obstacles


def render_obstacles_block(obstacles, obstacle_config_path: Path) -> str:
    lines = [
        BEGIN_MARKER,
        f"    <!-- Source of truth: {obstacle_config_path} -->",
    ]
    for obstacle in obstacles:
        lines.extend(
            [
                f'    <body name="{obstacle["name"]}" pos="{obstacle["pos"]}">',
                f'      <geom type="sphere" size="{obstacle["radius"]}" rgba="{obstacle["rgba"]}"',
                '            mass="0"',
                '            contype="1" conaffinity="1"/>',
                "    </body>",
                "",
            ]
        )
    if lines[-1] == "":
        lines.pop()
    lines.append(END_MARKER)
    return "\n".join(lines)


def sync_scene(obstacle_config_path: Path, scene_xml_path: Path) -> int:
    obstacles = parse_obstacles(obstacle_config_path)
    scene_text = scene_xml_path.read_text(encoding="utf-8")

    begin = scene_text.find(BEGIN_MARKER)
    end = scene_text.find(END_MARKER)
    if begin < 0 or end < 0 or end < begin:
        raise ValueError(
            "Scene XML does not contain the expected auto-generated obstacle markers: "
            f"'{scene_xml_path}'."
        )

    end += len(END_MARKER)
    updated = (
        scene_text[:begin]
        + render_obstacles_block(obstacles, obstacle_config_path)
        + scene_text[end:]
    )

    if updated != scene_text:
        scene_xml_path.write_text(updated, encoding="utf-8")
    return len(obstacles)


def main():
    parser = argparse.ArgumentParser(
        description="Synchronize static sphere obstacle config into MuJoCo scene_tau_ros.xml"
    )
    parser.add_argument("--obstacle-config", required=True, help="Canonical static obstacle config XML")
    parser.add_argument("--scene-xml", required=True, help="MuJoCo scene XML to update in place")
    args = parser.parse_args()

    obstacle_config_path = Path(args.obstacle_config).resolve()
    scene_xml_path = Path(args.scene_xml).resolve()

    if not obstacle_config_path.exists():
        raise FileNotFoundError(f"Obstacle config does not exist: {obstacle_config_path}")
    if not scene_xml_path.exists():
        raise FileNotFoundError(f"Scene XML does not exist: {scene_xml_path}")

    obstacle_count = sync_scene(obstacle_config_path, scene_xml_path)
    print(
        f"Synchronized {obstacle_count} static sphere obstacles from "
        f"{obstacle_config_path} into {scene_xml_path}."
    )


if __name__ == "__main__":
    main()

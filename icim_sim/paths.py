# -*- coding: utf-8 -*-
"""所有路径都以这个包所在目录为基准，整包拷到哪都能跑（不要写绝对路径）。

    icim_sim/            <- PACKAGE_ROOT
      icim_sim/paths.py
      assets/scene.xml   <- SCENE
      assets/meshes/...
      assets/arm/...
      third_party/       <- 队友的仓库放这里
      runs/              <- 运行产物
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ASSETS = PACKAGE_ROOT / "assets"
SCENE = ASSETS / "scene.xml"
RUNS = PACKAGE_ROOT / "runs"
THIRD_PARTY = PACKAGE_ROOT / "third_party"
#: 起始关节角：队友可以直接改这个文件，不用重建场景
START_POSE = PACKAGE_ROOT / "start_pose.json"
#: 拍照位（法兰位姿）：设备安装和托盘没变时不用重算，见 docs/API.md
CAPTURE_POSE = PACKAGE_ROOT / "capture_pose.json"

#: 环境变量可以覆盖场景文件（换一套夹具/网格时用）
SCENE = Path(os.environ.get("ICIM_SIM_SCENE", SCENE))


def run_dir(name: str) -> Path:
    """runs/ 下面的一个子目录，顺手建好。"""
    p = RUNS / name
    p.mkdir(parents=True, exist_ok=True)
    return p

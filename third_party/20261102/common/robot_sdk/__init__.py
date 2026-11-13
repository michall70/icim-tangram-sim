"""双版本机器人 SDK 管理。

两个版本的厂商 SDK（Robot.py）按子目录原样存放：
- v3_8_2: 适配控制器版本 3.8.2
- v3_9_3: 适配控制器版本 3.9.3

两台机器人 IP 相同，无法按 IP 区分，由 common/robot_client.py
在连接后读取控制器实际版本来决定加载哪个 SDK。
"""

import importlib

SUPPORTED_VERSIONS = ("3.8.2", "3.9.3")

_PACKAGE_BY_VERSION = {
    "3.8.2": "common.robot_sdk.v3_8_2",
    "3.9.3": "common.robot_sdk.v3_9_3",
}

_loaded = {}


def load_robot_sdk(version: str):
    """加载指定控制器版本对应的 SDK 模块（含 RPC 类），同版本只加载一次。"""
    if version not in _PACKAGE_BY_VERSION:
        raise ValueError(f"不支持的 SDK 版本: {version}，可选: {SUPPORTED_VERSIONS}")
    if version not in _loaded:
        _loaded[version] = importlib.import_module(_PACKAGE_BY_VERSION[version] + ".Robot")
    return _loaded[version]

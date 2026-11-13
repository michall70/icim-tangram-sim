"""七巧板场景的纯几何层。

本包的约束：**只有 numpy / math，不 import cv2、不 import 相机或机器人**。
原因是在没有硬件的情况下，这是唯一能用合成数据穷举验证「解对不对」的地方。
"""

from scenarios.tangram_puzzle.tangram import piece_set  # noqa: F401

__all__ = ["piece_set"]

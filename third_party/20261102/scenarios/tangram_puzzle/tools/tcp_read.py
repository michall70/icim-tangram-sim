"""实时打印机器人 TCP 坐标（**只读，不发送任何运动指令**）。

    python -B -m scenarios.tangram_puzzle.tools.tcp_read

按 Ctrl+C 退出。

## 用来干什么

判断"托盘是不是和机器人 X/Y 轴平行"，不需要相机：

1. 点动吸盘尖端**轻触托盘左前角**，记下打印的 x、y；
2. 沿托盘前边点动到**右前角**，再记一次；
3. 算两条记录的夹角：

   ```
   angle = degrees(atan2(y2 - y1, x2 - x1))
   ```

   - ≈0° → 这条边与机器人 **X 轴平行**；
   - 同法量左边：`atan2(y2-y1, x2-x1)` ≈ 90° → 与 **Y 轴平行**。

   经验阈值：|angle| ≤ 2° 视为平行；2—5° 可接受；> 5° 建议把托盘摆正。

也可以只按一次、看数值变化来判断：**点动时哪个坐标不变，那条边就平行于哪个轴**。
"""

import argparse
import time

from common.robot_client import CobotClient
from scenarios.tangram_puzzle.config import ROBOT_IP, SDK_VERSION, TOOL, USER


def main(argv=None):
    parser = argparse.ArgumentParser(description="实时打印 TCP（只读）")
    parser.add_argument("--interval", type=float, default=0.5, help="刷新间隔秒，默认 0.5")
    args = parser.parse_args(argv)

    robot = CobotClient(ROBOT_IP, tool=TOOL, user=USER, sdk_version=SDK_VERSION)
    print("[tcp_read] 只读：不会发送任何运动指令；点动用示教器。Ctrl+C 退出。")
    robot.connect()
    try:
        robot.check_coordinate_context()
        while True:
            x, y, z, rx, ry, rz = robot.get_tcp_pose()[:6]
            print("\r  x=%+9.2f  y=%+9.2f  z=%+9.2f   rz=%+8.2f   %s"
                  % (x, y, z, rz, time.strftime("%H:%M:%S")), end="", flush=True)
            time.sleep(max(0.05, args.interval))
    except KeyboardInterrupt:
        print("\n[tcp_read] 结束。")
    finally:
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

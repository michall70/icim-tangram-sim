"""夹爪控制：通过机器人控制箱数字输出（DO）驱动气动吸盘。

赛题工位使用**单吸盘**，接控制箱 **DO0**（`config.GRIPPER_DO = 0`），高电平吸。
吸盘没有到位反馈，只能按固定延时等待动作完成；若现场接线相反，改
`config.GRIPPER_CLOSE_LEVEL` 即可，不要改本文件。

历史说明：早期方案曾用 OPC UA 读写 PLC 夹爪信号（`OpcUaGripper`）。该方案与本项目
的 DO 吸盘不兼容，也没有任何调用点，已删除，避免误用与额外的 opcua 依赖。
"""

import time


class RobotDOGripper:
    """通过控制箱 DO 控制的吸盘/夹爪，接口为 `grip()` / `release()`。

    robot: 已连接的 CobotClient；do_id: 控制箱数字输出编号（默认 0 = DO0）；
    close_level: 夹紧电平（1/0，若实际动作相反对调即可）。
    """

    def __init__(self, robot, do_id=0, close_level=1, action_time=2.0):
        self.robot = robot
        self.do_id = do_id
        self.close_level = close_level
        self.release_level = 1 - close_level
        self.action_time = action_time

    def grip(self):
        self.robot.rpc.SetDO(self.do_id, self.close_level)
        time.sleep(self.action_time)

    def release(self):
        self.robot.rpc.SetDO(self.do_id, self.release_level)
        time.sleep(self.action_time)

    def is_gripped(self):
        _, (_, do_l) = self.robot.rpc.GetDO()
        return ((do_l >> self.do_id) & 0x01) == self.close_level

"""吸盘偏心工具：预测、实测反解、实测落点复核。默认纯离线，只有 --measure 会连机器人（只读位姿）。

原理（见 config.py SUCTION_OFFSET_XY 注释）：吸盘中心偏离法兰 Z 轴 offset 时，
放置阶段绕 Z 旋转 delta 使块中心偏离规划点 (R(delta)-I)·offset，大小约
2|offset|·sin(|delta|/2)；抓取阶段 delta=0，不产生偏差。

四种用法（项目根目录执行）：

1) 预测：给定旋转角，看当前偏心会带来多大偏差、以及 planner 会补多少
   python -m scenarios.tangram_puzzle.tool_offset_check --predict -174.8 -84.3 13.5

2) 反解：用实测落点算偏心（补偿前的历史数据）
   python -m scenarios.tangram_puzzle.tool_offset_check --estimate 实测.csv
   CSV 每行：delta_deg,dx_mm,dy_mm（dx,dy = 实测块中心 - 规划落点）

3) 实测（需要机械臂，只读取位姿，不发送任何运动指令）：
   python -m scenarios.tangram_puzzle.tool_offset_check --measure
   由你在示教器上点动：尖端对准同一标记，分别在原姿态和转 180° 后读数，
   程序按 u = (R(rzA)-R(rzB))⁻¹(F_B-F_A) 求偏心。

4) 落点复核（需要机械臂拍照，离线分析）：
   python -m scenarios.tangram_puzzle.tool_offset_check --landing 成品区实拍.png --plans plans.json
   在拍照位拍一张成品区，检测每块实际位置并与计划比对，输出逐块偏差、均值向量，
   再用最小二乘反解偏心。要求照片与 plans.json 对应同一次运行、同一拍照位。
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scenarios.tangram_puzzle.config import SUCTION_OFFSET_XY
from scenarios.tangram_puzzle.puzzle_planner import rotation_offset


def predicted_deviation(delta_deg, offset):
    """块中心偏差 = (R(delta)-I)·offset（mm）。"""
    return rotation_offset(delta_deg, offset)


# 两个样本的旋转角差小于该值时，(R(delta)-I) 方向几乎相同，反解结果由噪声主导。
MIN_SAMPLE_SPREAD_DEG = 10.0

# 反解残差超过该值，说明偏差里还有偏心以外的成分（识别、标定、吸放滑移）。
MAX_SOLVE_RESIDUAL_MM = 2.0


def solve_suction_offset(samples):
    """由 (delta_deg, dx, dy) 样本最小二乘反解偏心，返回 (offset, 残差RMS mm)。"""
    rows, rhs = [], []
    for delta, dx, dy in samples:
        values = [float(delta), float(dx), float(dy)]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"样本含非有限数值: {values}")
        a = math.radians(float(delta))
        c, s = math.cos(a), math.sin(a)
        rows.append([c - 1.0, -s])
        rows.append([s, c - 1.0])
        rhs.extend([float(dx), float(dy)])
    if len(samples) < 2:
        raise ValueError("至少需要两个样本（且旋转角不同）才能反解偏心")
    a_matrix = np.asarray(rows, float)
    if np.linalg.matrix_rank(a_matrix) < 2:
        raise ValueError("样本旋转角全部为 0°（或互相重复），没有可用于反解的数据")
    deltas = [float(s[0]) % 360.0 for s in samples]
    spread = 0.0
    for i, first in enumerate(deltas):
        for second in deltas[i + 1:]:
            diff = abs(first - second) % 360.0
            spread = max(spread, min(diff, 360.0 - diff))
    if spread < MIN_SAMPLE_SPREAD_DEG:
        raise ValueError(f"样本旋转角过于接近（最大差 {spread:.1f}° < {MIN_SAMPLE_SPREAD_DEG:.0f}°），"
                         f"反解会被噪声主导；请补测旋转角差别更大的块")
    offset, *_ = np.linalg.lstsq(a_matrix, np.asarray(rhs, float), rcond=None)
    residual = float(np.sqrt(np.mean((a_matrix @ offset - np.asarray(rhs, float)) ** 2)))
    return offset, residual


def read_samples(path):
    """读取 CSV/空白分隔的 delta,dx,dy 行，忽略空行与 # 注释；兼容带 BOM 的文件。"""
    samples = []
    text = Path(path).read_text(encoding="utf-8-sig").replace("\ufeff", "")
    for number, line in enumerate(text.splitlines(), 1):
        text_line = line.split("#", 1)[0].strip()
        if not text_line:
            continue
        parts = [p for p in text_line.replace(",", " ").split() if p]
        if len(parts) != 3:
            raise ValueError(f"第 {number} 行应为 delta,dx,dy 三个数: {line!r}")
        samples.append(tuple(float(p) for p in parts))
    if not samples:
        raise ValueError(f"{path} 中没有有效数据行")
    return samples


def describe(offset):
    magnitude = float(np.linalg.norm(offset))
    direction = math.degrees(math.atan2(offset[1], offset[0])) if magnitude else 0.0
    return f"({offset[0]:+.2f}, {offset[1]:+.2f}) mm，长度 {magnitude:.2f} mm，方向 {direction:+.1f}°"


# ---------------- 实测：两次对准同一标记 ----------------

def _rotation_xy(degrees):
    angle = math.radians(degrees)
    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]])


def solve_offset_from_pair(first, second, reference_rz=None):
    """由两次"尖端对准同一标记"的法兰位姿求偏心（mm，参考姿态下的基坐标向量）。

    尖端固定不动时法兰位置 F = M - R(rz)·e，e 为法兰坐标系中的偏心，于是
        e = (R(rzA) - R(rzB))⁻¹ · (F_B - F_A)
    而 planner 需要的是"参考姿态下的基坐标偏心" u = R(rz_ref)·e（rz_ref 为抓取姿态，
    标定就是在该姿态下完成的）。缺省用第一次测量的 rz 作为参考。

    返回 (u, 两次法兰 Z 差 mm, 两次 Z 角差 °)。
    """
    if len(first) != 6 or len(second) != 6:
        raise ValueError("位姿必须是六个数 [x, y, z, rx, ry, rz]")
    matrix = _rotation_xy(first[5]) - _rotation_xy(second[5])
    if abs(float(np.linalg.det(matrix))) < 0.2:
        raise ValueError("两次的 Z 角变化太小（接近 0° 或 360°），无法分离偏心方向；请转到约 180° 再测")
    flange_offset = np.linalg.solve(matrix,
                                    np.array([second[0] - first[0], second[1] - first[1]], float))
    rz_ref = float(first[5] if reference_rz is None else reference_rz)
    offset = _rotation_xy(rz_ref) @ flange_offset
    turn = (second[5] - first[5]) % 360.0
    turn = min(turn, 360.0 - turn)
    return offset, abs(second[2] - first[2]), turn


def _read_tcp(robot):
    ret = robot.rpc.GetActualTCPPose()
    if not isinstance(ret, (tuple, list)) or len(ret) != 2:
        raise RuntimeError(f"GetActualTCPPose 返回格式异常: {ret}")
    robot._check(ret[0], "读取TCP位姿")
    pose = [float(v) for v in ret[1]]
    if len(pose) != 6:
        raise RuntimeError(f"TCP 位姿长度异常: {pose}")
    return pose


def measure_suction_offset(ask=input, echo=print):
    """连接机器人只读取位姿，由用户点动完成两次对准，返回偏心向量（mm）。"""
    from common.robot_client import CobotClient
    from scenarios.tangram_puzzle.config import ROBOT_IP, SDK_VERSION, TCP_RPY, TOOL, USER

    robot = CobotClient(ROBOT_IP, tool=TOOL, user=USER, sdk_version=SDK_VERSION)
    echo("连接机器人：本命令只读位姿，不发送任何运动指令；点动由你在示教器上完成。")
    robot.connect()
    try:
        robot.check_coordinate_context()
        echo(f"要求姿态 RX/RY/RZ = {TCP_RPY}（吸盘偏心在该姿态下定义）")
        ask("① 把吸盘尖端对准纸上十字标记（轻触即可），按 Enter 记录 A …")
        first = _read_tcp(robot)
        echo(f"   A = {[round(v, 3) for v in first]}")
        ask("② 只把末端绕 Z 轴转约 180°（不要改变 XYZ），再点动使尖端回到同一标记，按 Enter 记录 B …")
        second = _read_tcp(robot)
        echo(f"   B = {[round(v, 3) for v in second]}")
    finally:
        robot.disconnect()

    offset, dz, turn = solve_offset_from_pair(first, second, reference_rz=TCP_RPY[2])
    echo(f"两次 Z 角差 {turn:.1f}°，两次法兰 Z 差 {dz:.2f} mm")
    if turn < 150.0:
        echo("提示：转角偏离 180°，结果仍可用但灵敏度下降，建议重测。")
    if dz > 3.0:
        echo("警告：两次 Z 相差较大，说明尖端没有落在同一点/同高度，结果可能有误。")
    tilt = max(abs(first[3] - TCP_RPY[0]), abs(first[4] - TCP_RPY[1]))
    if tilt > 2.0:
        echo(f"警告：当前 RX/RY 与抓取姿态相差 {tilt:.1f}°，吸盘可能不是垂直朝下，"
             f"偏心会投影到不同方向；建议把吸盘摆正后重测。")
    rz_gap = abs((first[5] - TCP_RPY[2] + 180.0) % 360.0 - 180.0)
    echo(f"（当前 RZ 与抓取姿态相差 {rz_gap:.1f}°，不影响结果：已按参考 RZ={TCP_RPY[2]} 换算）")
    echo(f"\n反解偏心: {describe(offset)}")
    echo(f"写回 config.py: SUCTION_OFFSET_XY = ({offset[0]:.2f}, {offset[1]:.2f})")
    return offset


# ---------------- 落点复核：成品区实拍 vs 计划 ----------------

def load_plans(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(item["piece_id"]): item for item in data["plans"]}


def measure_landing(image_path, plans_path, already_undistorted=False, echo=print,
                    detect_params=None):
    """检测成品区实拍图中各块位置，与 plans.json 的放置位姿比对，返回 [(id, Δ, 偏差)]。"""
    import cv2 as cv

    from common.calibration import CameraMapper
    from scenarios.tangram_puzzle.config import (CAMERA_MATRIX, DETECT_PARAMS, DIST_COEFFS,
                                                 HOMOGRAPHY)
    from scenarios.tangram_puzzle.tangram_detector import TangramDetector

    image = cv.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    mapper = CameraMapper(CAMERA_MATRIX, DIST_COEFFS, HOMOGRAPHY)
    if not already_undistorted:
        image = mapper.undistort(image)
    pieces = TangramDetector(detect_params or DETECT_PARAMS).detect_template(image)
    plans = load_plans(plans_path)
    rows = []
    for piece in pieces:
        plan = plans.get(piece.piece_id)
        if plan is None:
            echo(f"警告：图中检出 ID={piece.piece_id}，但 plans.json 没有对应计划，已跳过")
            continue
        actual = np.asarray(mapper.pixel_to_robot(piece.grasp_point), float)
        planned = np.asarray(plan["place_pose"][:2], float)
        rows.append((piece.piece_id, float(plan["delta_angle"]), actual - planned))
    return rows


def report_landing(rows, echo=print):
    """打印逐块偏差、均值向量与反解偏心。返回 (offset 或 None, 残差 或 None)。"""
    if not rows:
        raise ValueError("没有可比对的块：确认照片是成品区实拍、且与 plans.json 属于同一次运行")
    echo(f"\n{'ID':>3} {'Δ(°)':>9} {'偏差X':>9} {'偏差Y':>9} {'长度':>8}")
    for piece_id, delta, error in rows:
        echo(f"{piece_id:>3} {delta:>9.2f} {error[0]:>9.2f} {error[1]:>9.2f} "
             f"{float(np.linalg.norm(error)):>8.2f}")
    lengths = [float(np.linalg.norm(error)) for _, _, error in rows]
    mean_vector = np.mean([error for _, _, error in rows], axis=0)
    echo(f"平均偏差长度 {np.mean(lengths):.2f} mm；均值向量 ({mean_vector[0]:+.2f}, "
         f"{mean_vector[1]:+.2f})，长度 {float(np.linalg.norm(mean_vector)):.2f} mm")
    echo("解释：均值向量≈0 而逐块偏差随 Δ 变化 → 与吸盘偏心一致；均值向量明显不为 0 → "
         "还存在与旋转无关的固定偏移（标定/高度/放置点）。")

    samples = [(delta, error[0], error[1]) for _, delta, error in rows]
    if len(samples) < 2:
        echo("样本不足 2 块，无法反解偏心。")
        return None, None
    try:
        offset, residual = solve_suction_offset(samples)
    except ValueError as exc:
        echo(f"无法反解偏心：{exc}")
        return None, None
    echo(f"\n反解偏心: {describe(offset)}，残差 RMS={residual:.3f} mm")
    if residual > MAX_SOLVE_RESIDUAL_MM:
        echo(f"残差 {residual:.2f} mm 超过 {MAX_SOLVE_RESIDUAL_MM:.0f} mm：偏差中还有偏心以外的成分，"
             f"不要直接写回配置。")
    else:
        echo("若这批计划是用 SUCTION_OFFSET_XY=(0,0) 生成的，该值就是应写回 config 的偏心；"
             "若计划已启用补偿而反解值明显偏离配置，说明偏心或其它误差源需复查。")
    return offset, residual


# ---------------- 命令行 ----------------

def main():
    parser = argparse.ArgumentParser(description="吸盘偏心预测/实测/落点复核（默认离线）")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predict", nargs="+", type=float, metavar="DELTA",
                       help="按给定旋转角（度）预测偏差与补偿量")
    group.add_argument("--estimate", metavar="CSV", help="由实测 (delta,dx,dy) 反解偏心")
    group.add_argument("--measure", action="store_true",
                       help="连接机器人（只读位姿），两次对准同一标记实测偏心")
    group.add_argument("--landing", metavar="IMAGE", help="成品区实拍图，与 --plans 比对该次落点")
    parser.add_argument("--plans", metavar="JSON", help="--landing 使用的 plans.json")
    parser.add_argument("--already-undistorted", action="store_true",
                        help="--landing 输入图已去畸变")
    parser.add_argument("--offset", nargs=2, type=float, metavar=("X", "Y"),
                        help=f"预测用偏心，缺省取 config.SUCTION_OFFSET_XY（当前 {tuple(SUCTION_OFFSET_XY)}）")
    args = parser.parse_args()

    if args.measure:
        measure_suction_offset()
        return

    if args.landing:
        if not args.plans:
            parser.error("--landing 需要同时给出 --plans plans.json")
        rows = measure_landing(args.landing, args.plans, args.already_undistorted)
        report_landing(rows)
        return

    offset = np.asarray(args.offset if args.offset else SUCTION_OFFSET_XY, float)
    print(f"当前使用偏心: {describe(offset)}")

    if args.predict:
        print(f"\n{'Δ(°)':>9} {'偏差X':>9} {'偏差Y':>9} {'偏差长度':>10} {'补偿X':>9} {'补偿Y':>9}")
        for delta in args.predict:
            deviation = predicted_deviation(delta, offset)
            print(f"{delta:>9.2f} {deviation[0]:>9.2f} {deviation[1]:>9.2f} "
                  f"{np.linalg.norm(deviation):>10.2f} {-deviation[0]:>9.2f} {-deviation[1]:>9.2f}")
        print("\n偏差=补偿前块中心会偏多少；补偿=planner 实际加到放置位姿上的位移（取负）。")
        return

    samples = read_samples(args.estimate)
    solved, residual = solve_suction_offset(samples)
    print(f"样本 {len(samples)} 组，残差 RMS={residual:.3f} mm")
    print(f"反解偏心: {describe(solved)}")
    if residual > MAX_SOLVE_RESIDUAL_MM:
        print(f"残差 {residual:.2f} mm 超过 {MAX_SOLVE_RESIDUAL_MM:.0f} mm：说明除偏心外还有识别/标定/滑移误差，"
              f"不要直接采用该值，先查其他误差源。")
    else:
        print(f"写回 config.py: SUCTION_OFFSET_XY = ({solved[0]:.2f}, {solved[1]:.2f})")
    print("注意：反解用的是补偿前的数据；若已启用补偿，应先把 SUCTION_OFFSET_XY 置 (0,0) 再采数据。")


if __name__ == "__main__":
    main()

"""目标布局的尺度归一：让"要拼多大多大"由实物决定，而不是由模板图的检测误差决定。

## 为什么需要

`plan_puzzle` 的放置位姿直接取自**模板图检测出的七块形心**。模板图里的块有多大，取决于
检测掩模有多大。实测 `stage3_20261025/template_Color_Color.png`：正方形、小三角、平行四边形
三块反推的 S 都是 115.2—115.4mm（卡尺值 115.84，误差 <0.6%，很准），但**两个大三角分别偏大
3.5% 和 7.8%**。大三角是面积最大的两块，它们一胀，整个布局的包围盒就从理论 116mm 撑到 146mm。

机器人按形心放置，于是两个最大块被推出去、其余块相对被拉开 —— 现场表现就是
"**整体放大、块之间缝很大**"。

## 做法

七块面积恒为 S² 的固定份额，所以**由单块面积就能反推出这套布局隐含的 S**：

    大三角 4·A = S²    中三角/正方形/平行四边形 8·A = S²    小三角 16·A = S²

取各块反推值的中位数作为整幅布局的尺度（中位数天然抗单块离群），再按

    k = S_phys / S_隐含

把所有放置点绕布局形心缩放 k 倍。这样即使个别块掩模偏大，**布局的总尺度仍然正确**。
"""

import dataclasses
import math

import numpy as np

from scenarios.tangram_puzzle.planning import assign
from scenarios.tangram_puzzle.puzzle_planner import _transform

# 类别 → 面积占 S² 的份额的倒数（A = S² / 分母）
AREA_DIVISOR = {"L": 4.0, "M": 8.0, "Q": 8.0, "P": 8.0, "T": 16.0}

# 尺度修正的触发阈值：小于 2% 视为检测噪声，不做调整
SCALE_THRESHOLD = 0.02


def implied_scale(template_pieces, homography):
    """由模板七块反推这套布局隐含的 S（mm），返回 (中位数, 明细列表)。"""
    labeled = assign.classify_pieces(template_pieces)
    detail = []
    values = []
    for piece, kind in labeled:
        area_mm = abs(_polygon_area(_transform(piece.vertices, homography)))
        if area_mm <= 0:
            continue
        scale = math.sqrt(area_mm * AREA_DIVISOR[kind])
        values.append(scale)
        detail.append({"piece_id": piece.piece_id, "kind": kind,
                       "area_mm2": round(area_mm, 1), "implied_s": round(scale, 2)})
    if not values:
        return None, detail
    return float(np.median(values)), detail


def _polygon_area(points):
    points = np.asarray(points, dtype=float)
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def normalize_plans(plans, template_pieces, homography, s_phys):
    """按实物尺度 S_phys 把目标布局整体缩放，返回 (新计划列表, 报告)。

    只改放置点的 x/y（绕布局形心缩放），不动 z、姿态角、抓取点。
    检测尺度已经足够准时（|k-1| < SCALE_THRESHOLD）原样返回，避免无谓扰动。
    """
    report = {"s_phys": float(s_phys), "factor": 1.0, "applied": False, "detail": []}
    if not plans:
        return plans, report
    implied, detail = implied_scale(template_pieces, homography)
    report["detail"] = detail
    report["implied_s"] = implied
    if implied is None or implied <= 0:
        report["note"] = "无法反推尺度，保持原计划"
        return plans, report

    factor = float(s_phys) / implied
    report["factor"] = factor
    if abs(factor - 1.0) <= SCALE_THRESHOLD:
        report["note"] = "尺度偏差在容差内，未调整"
        return plans, report

    xs = np.array([plan.place_pose[0] for plan in plans])
    ys = np.array([plan.place_pose[1] for plan in plans])
    center = np.array([float(xs.mean()), float(ys.mean())])
    adjusted = []
    for plan in plans:
        pose = list(plan.place_pose)
        point = np.array([pose[0], pose[1]])
        scaled = center + (point - center) * factor
        pose[0], pose[1] = float(scaled[0]), float(scaled[1])
        adjusted.append(dataclasses.replace(plan, place_pose=tuple(pose)))
    report["applied"] = True
    report["center"] = [float(center[0]), float(center[1])]
    return adjusted, report


def describe(report):
    if not report.get("applied"):
        return "尺度归一未启用（%s）；隐含 S=%s" % (
            report.get("note", "-"),
            "%.2f" % report["implied_s"] if report.get("implied_s") else "未知")
    worst = max(report["detail"], key=lambda item: abs(item["implied_s"] - report["s_phys"]))
    return ("尺度归一已生效：隐含 S=%.2fmm → 实物 S=%.2fmm（系数 %.4f）；"
            "最大离群 %s id=%s（%.2fmm）"
            % (report["implied_s"], report["s_phys"], report["factor"],
               worst["kind"], worst["piece_id"], worst["implied_s"]))

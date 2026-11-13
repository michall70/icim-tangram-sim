"""上机前的几何自检：一条命令判断"这张模板/这次标定能不能用"。

## 解决的问题

以前换模板或动了相机，只能真上机器人跑 7 次抓放才知道对不对——代价高、且失败时只看到
"偏移很大"这种笼统现象。本工具把判断提前到离线：

- **逐块反推 S**：七块面积恒为 S² 的固定份额，由单块面积就能反推这套布局隐含的 S。
  三块小件都指向 115.2—115.4mm、两个大三角却指向 119.9/124.9mm —— 一眼就能看出是哪块不对。
- **整体包围盒与填充率**：七巧板正方形应≈S×S、填充率≈100%。实测 189×118.6、61% 就说明
  这张模板本身不自洽，不能用。
- **与卡尺值比较**：给出全局尺度偏差，决定要不要做尺度归一。

## 用法

    python -B -m scenarios.tangram_puzzle.tools.check_geometry --template <模板图>
    python -B -m scenarios.tangram_puzzle.tools.check_geometry --template <模板图> --raw <实拍帧>

只读，不连机器人、不动硬件。
"""

import argparse
import sys

import cv2 as cv
import numpy as np

from common.calibration import CameraMapper
from scenarios.tangram_puzzle.config import (CAMERA_MATRIX, DETECT_PARAMS, DIST_COEFFS,
                                             HOMOGRAPHY)
from scenarios.tangram_puzzle.planning import scale
from scenarios.tangram_puzzle.puzzle_planner import _transform
from scenarios.tangram_puzzle.tangram_detector import TangramDetector

# 判定阈值
SCALE_TOLERANCE = 0.03      # 单块反推 S 与卡尺值的允许偏差
BBOX_TOLERANCE = 0.08       # 包围盒边长与 S 的允许偏差
FILL_MIN = 0.85             # 正方形图形的填充率下限


def measure(pieces, label):
    """逐块换算到毫米并反推 S，返回 (报告 dict)。"""
    mapper_report = {"label": label, "pieces": [], "rows": []}
    implied, detail = scale.implied_scale(pieces, HOMOGRAPHY)
    mapper_report["implied_s"] = implied
    mapper_report["pieces"] = detail

    all_points = []
    total_area = 0.0
    for piece, row in zip(pieces, detail):
        vertices = _transform(piece.vertices, HOMOGRAPHY)
        all_points.append(vertices)
        total_area += row["area_mm2"]
    if not all_points:
        return mapper_report
    points = np.vstack(all_points)
    width = float(points[:, 0].max() - points[:, 0].min())
    height = float(points[:, 1].max() - points[:, 1].min())
    mapper_report.update({"width": width, "height": height,
                          "total_area": total_area,
                          "fill": total_area / (width * height) if width * height > 0 else 0.0})
    return mapper_report


def check(report, s_phys):
    """给出结论与问题清单。"""
    issues = []
    implied = report.get("implied_s")
    if implied is None:
        return ["无法反推尺度：检测到的块不足或面积为零"], False
    rows = report["pieces"]
    for row in rows:
        deviation = (row["implied_s"] - s_phys) / s_phys
        if abs(deviation) > SCALE_TOLERANCE:
            issues.append("ID=%s(%s) 反推 S=%.2fmm，与卡尺 %.2fmm 差 %+.1f%%"
                          % (row["piece_id"], row["kind"], row["implied_s"], s_phys,
                             deviation * 100))
    width, height = report["width"], report["height"]
    for name, value in (("宽", width), ("高", height)):
        if abs(value - s_phys) / s_phys > BBOX_TOLERANCE:
            issues.append("整体%s %.1fmm，与 S=%.2fmm 差 %+.1f%%（正方形图形应≈S）"
                          % (name, value, s_phys, (value - s_phys) / s_phys * 100))
    if abs(width - height) / max(width, height) > BBOX_TOLERANCE:
        issues.append("包围盒长宽比 %.2f:1，正方形图形应接近 1:1" % (width / height))
    if report["fill"] < FILL_MIN:
        issues.append("填充率 %.1f%%，低于 %.0f%%：块之间有异常缝隙"
                      % (report["fill"] * 100, FILL_MIN * 100))
    return issues, not issues


def main(argv=None):
    parser = argparse.ArgumentParser(description="七巧板几何自检（只读，不连机器人）")
    parser.add_argument("--template", required=True, help="模板图（成品区拼好的图）")
    parser.add_argument("--raw", help="可选：原料区实拍帧，一并检查")
    parser.add_argument("--s-phys", type=float, default=None, help="实物 S（mm），默认取配置")
    args = parser.parse_args(argv)

    from scenarios.tangram_puzzle.config import S_PHYS_MM
    s_phys = args.s_phys if args.s_phys is not None else S_PHYS_MM
    mapper = CameraMapper(CAMERA_MATRIX, DIST_COEFFS, HOMOGRAPHY)

    ok_all = True
    for label, path, detector_call in (
            ("模板", args.template, "template"),
            ("原料", args.raw, "raw")):
        if not path:
            continue
        image = cv.imread(path)
        if image is None:
            print("无法读取 %s: %s" % (label, path))
            ok_all = False
            continue
        image = mapper.undistort(image)
        detector = TangramDetector(DETECT_PARAMS)
        pieces = (detector.detect_template(image) if detector_call == "template"
                  else detector.detect_raw_material(image))
        print("=" * 72)
        print("%s：%s  检出 %d 块" % (label, path, len(pieces)))
        if not pieces:
            print("  [警告] 未检出任何块")
            ok_all = False
            continue
        report = measure(pieces, label)
        implied = report.get("implied_s")
        print("  隐含 S = %s mm（卡尺 %.2f mm）"
              % ("%.2f" % implied if implied else "未知", s_phys))
        print("  %-4s %-4s %10s %10s %9s" % ("ID", "类型", "面积mm2", "反推S", "偏差"))
        for row in report["pieces"]:
            dev = (row["implied_s"] - s_phys) / s_phys * 100 if s_phys else 0.0
            print("  %-4s %-4s %10.1f %10.2f %8.1f%%"
                  % (row["piece_id"], row["kind"], row["area_mm2"], row["implied_s"], dev))
        if "width" in report:
            print("  包围盒 %.1f x %.1f mm   填充率 %.1f%%"
                  % (report["width"], report["height"], report["fill"] * 100))
        issues, ok = check(report, s_phys)
        ok_all = ok_all and ok
        if ok:
            print("  结论：[通过] 几何自洽，可用于规划")
        else:
            print("  结论：[失败] 不可用")
            for issue in issues:
                print("        - " + issue)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

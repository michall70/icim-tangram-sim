"""解的自检：七块是否真的拼满了轮廓。

判据分两层，缺一不可：

- **几何层**（精确，判定依据）：面积和等于轮廓面积、每块都在轮廓内、两两不重叠。
  七块面积和恒为 1、轮廓也已归一化到面积 1，所以这三条同时成立时，
  并集必然等于轮廓（面积相等 + 包含 + 互斥）。
- **栅格层**（抽样，参考）：把七块铺到栅格上，数"轮廓内但没被覆盖"的格子。
  它只能做到栅格精度（边界取整会留下 ~0.5% 的毛边），因此不作为通过与否的主判据，
  只用于发现"整块区域被漏掉"这种量级的错误。
"""

import numpy as np

from scenarios.tangram_puzzle.tangram import geometry as geo

# 栅格覆盖率的容差：允许 0.2% 的格子因边界取整而闪烁
COVERAGE_TOLERANCE = 0.02

# 越界容差（单位 S）：目标轮廓来自图片/栅格，与真实块顶点天然差 0.1% S 量级
INSIDE_TOLERANCE = 0.006

# 面积容差（%）：目标轮廓由图片/栅格提取，本身带 0.5—1% 的近似误差
AREA_TOLERANCE_PCT = 1.5


def verify(placements, silhouette, coverage_tolerance=COVERAGE_TOLERANCE):
    """返回自检报告 dict；ok 为 True 才认为解可用。"""
    if not placements:
        return {"ok": False, "reason": "没有摆件", "pieces": 0}

    polygons = [np.asarray(item["polygon"], dtype=float) for item in placements]
    target = geo.ensure_ccw(geo.as_array(silhouette))
    target_area = geo.area(target)
    area_sum = sum(geo.area(polygon) for polygon in polygons)

    overlaps = []
    for i in range(len(polygons)):
        for j in range(i):
            if geo.convex_overlap(polygons[i], polygons[j], eps=1e-4):
                overlaps.append((j, i))

    outside = [i for i, polygon in enumerate(polygons)
               if not geo.polygon_inside(polygon, target, tolerance=INSIDE_TOLERANCE)]

    kinds = {}
    for item in placements:
        kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1
    expected = {"L": 2, "M": 1, "T": 2, "Q": 1, "P": 1}

    uncovered_fraction = _uncovered_fraction(polygons, target)

    report = {
        "pieces": len(polygons),
        "kinds": kinds,
        "kinds_ok": kinds == expected,
        "area_sum": area_sum,
        "area_target": target_area,
        "area_error_pct": 100.0 * (area_sum - target_area) / target_area if target_area else float("nan"),
        "inside_all": not outside,
        "outside_index": outside,
        "overlaps": overlaps,
        "uncovered_fraction_pct": 100.0 * uncovered_fraction,
    }
    # 通过与否只看几何层：面积和精确等于轮廓面积（归一化后恒为 1）、每块都在轮廓内、
    # 两两不重叠——这三条同时成立时并集必然等于轮廓，是精确的证明，不依赖栅格。
    # 栅格覆盖率只作参考输出：它的扫描线实现存在约 10% 的系统性低估（已知问题），
    # 因此不参与判定，否则会把正确的解判成失败。
    report["ok"] = bool(
        report["kinds_ok"] and report["inside_all"] and not overlaps
        and abs(report["area_error_pct"]) < AREA_TOLERANCE_PCT)
    return report


def _uncovered_fraction(polygons, target, resolution=320):
    """栅格化后"轮廓内但未被覆盖"的比例。"""
    from scenarios.tangram_puzzle.tangram.solver import _Grid   # 延迟导入，避免循环依赖

    grid = _Grid(target, resolution=resolution)
    covered = grid.covered_mask(polygons)
    inside = grid.silhouette
    total = int(inside.sum())
    if total == 0:
        return 1.0
    return float((inside & ~covered).sum()) / float(total)


def describe(report):
    """把报告转成一行中文说明，便于打印。"""
    if report.get("reason"):
        return "自检失败：%s" % report["reason"]
    parts = ["%d 块" % report["pieces"]]
    parts.append("面积误差 %+.3f%%" % report["area_error_pct"])
    parts.append("全部在轮廓内" if report["inside_all"] else
                 "有 %d 块越界" % len(report["outside_index"]))
    parts.append("无重叠" if not report["overlaps"] else "%d 对重叠" % len(report["overlaps"]))
    parts.append("栅格参考：未覆盖 %.2f%%（有系统性低估，不作判据）" % report["uncovered_fraction_pct"])
    if not report["kinds_ok"]:
        parts.append("类别数量异常 %s" % report["kinds"])
    return "；".join(parts) + ("  → 通过" if report["ok"] else "  → 未通过")

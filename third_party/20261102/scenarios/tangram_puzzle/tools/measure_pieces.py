"""从一张照片测量七巧板的实际尺寸，反推尺度 S，并检验这副牌是否标准。

为什么需要它：比赛现场会另发一副七巧板，尺寸不能写死；进阶（白色剪影）任务必须用
实物尺度 S 把归一化的解缩放成毫米。本工具就是产生 S 和逐件实测边长的地方。

## 方法（两处取舍都是实测踩出来的）

**取舍一：S 用「边长法」，不用「面积法」。**
面积 ∝ 边长²，边界误差在面积上被放大一倍。实测中高光让轮廓外扩约 1 px，
面积法得到的 S 比卡尺偏大 **9.5%**，而边长法只偏大 **1.9%**。

**取舍二：四边形只用 `approxPolyDP`，不用 `minAreaRect` 兜底。**
`minAreaRect` 求的是**外接**矩形，会把正方形系统性撑大（实测 S 因此从 118 涨到 127）。
只有 `approxPolyDP` 的拟合是双向逼近的。

流程：轮廓 → 拟合 3/4 边形（边上带硬约束：最短边/最长边 ≥ 0.62，见 piece_set）→
按面积排序分类 → 由每条边反推 S → 取中位数并剔除 >5% 的离群 → 输出。

## 用法

    # 生产分辨率 1280x720 的原图（内部去畸变 + 单应映射到毫米）
    python -B -m scenarios.tangram_puzzle.tools.measure_pieces --image 某张.png

    # 任意分辨率/任意相机（只比比例，S 的单位是像素）
    python -B -m scenarios.tangram_puzzle.tools.measure_pieces --image 某张.png --px

    # 直接用相机拍一张（不移动机器人）
    python -B -m scenarios.tangram_puzzle.tools.measure_pieces --camera

    # 用已知的一件实物尺寸做绝对标定（推荐在现场做）
    # 大三角直角边卡尺 82.5mm：
    python -B -m scenarios.tangram_puzzle.tools.measure_pieces --camera \
        --expect-kind L --expect-edge leg --expect-value 82.5

## 判读

- 「反推 S」逐条列出每块每条边给出的估计值；它们应当聚集在 ±2% 内。
- 被剔除的块会单列出来。**小块被高光吃掉一角时必然离群，这是判据问题，不是本工具问题。**
- 若七块不齐或剔除过多，S 不可采用，报告里会明确写。
"""

import argparse
import datetime
import json
import os
import sys

import cv2 as cv
import numpy as np

from scenarios.tangram_puzzle.config import (CAMERA_MATRIX, CAMERA_PARAMS, DETECT_PARAMS,
                                             DIST_COEFFS, HOMOGRAPHY)
from scenarios.tangram_puzzle.tangram import piece_set

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# 面积门槛按实际分辨率等比缩放（配置里的 500 px² 是按 1280x720 定的）。
REF_PIXELS = 1280 * 720
MIN_AREA_RATIO = DETECT_PARAMS["min_area"] / float(REF_PIXELS)
MAX_AREA_RATIO = 1.0 / 12.0

# 与颜色无关的分割：深色台面上的高饱和物体。
# 不能用「饱和度 OR 亮度」：会把台面反光、白墙连成一片，把相邻的块黏在一起（实测）。
SAT_MIN = 70
VAL_MIN = 50

# 块状过滤：实心度与长宽比，排掉台面边缘、反光条这类细长区域。
MIN_SOLIDITY = 0.85
MAX_ASPECT = 3.5

# 七巧板任何一块的最短边/最长边都不小于 0.707（见 piece_set 标称几何）；
# 低于这个值就是高光把某个角顶出去形成的伪边，必须拒绝。
MIN_EDGE_RATIO = 0.62

# 正方形 vs 平行四边形：四边最短/最长比。正方形≈1.00，平行四边形≈0.707。
SQUARE_EDGE_RATIO = 0.86

# 逐块 S 估计的离群门槛：偏离中位数超过该比例的不参与取平均。
OUTLIER_RATIO = 0.05


# ---------------------------------------------------------------- 分割与拟合

def _contours(img):
    """与颜色无关的分割 + 块状过滤，返回按面积降序的轮廓列表。"""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    sat = cv.inRange(hsv, np.array((0, SAT_MIN, 0), np.uint8), np.array((180, 255, 255), np.uint8))
    val = cv.inRange(hsv, np.array((0, 0, VAL_MIN), np.uint8), np.array((180, 255, 255), np.uint8))
    mask = cv.bitwise_and(sat, val)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    height, width = img.shape[:2]
    min_area = MIN_AREA_RATIO * height * width
    max_area = MAX_AREA_RATIO * height * width
    found, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    kept = []
    for contour in found:
        area = cv.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        hull_area = cv.contourArea(cv.convexHull(contour))
        if hull_area <= 0 or area / hull_area < MIN_SOLIDITY:
            continue
        _, _, bw, bh = cv.boundingRect(contour)
        if min(bw, bh) <= 0 or max(bw, bh) / float(min(bw, bh)) > MAX_ASPECT:
            continue
        kept.append(contour)
    kept.sort(key=cv.contourArea, reverse=True)
    return kept


def _edges(points):
    """按长度升序的边长列表。"""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    n = len(pts)
    return sorted(float(np.linalg.norm(pts[(i + 1) % n] - pts[i])) for i in range(n))


def _fit_shape(contour):
    """拟合轮廓，返回 (顶点, 顶点数, 最短边/最长边, 交并比)；失败返回 (None, 0, None, 0.0)。

    同时试三角形与四边形，交并比更高者胜、三角形在不显著落后时优先。
    只使用 approxPolyDP；不做外接矩形兜底（会把正方形系统性撑大）。
    """
    hull = cv.convexHull(contour)
    peri = cv.arcLength(hull, True)
    if peri <= 0:
        return None, 0, None, 0.0
    x, y, w, h = cv.boundingRect(contour)
    pad = 3
    origin = np.array([x - pad, y - pad], dtype=np.float32)
    canvas = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    cv.drawContours(canvas, [(contour - origin).astype(np.int32)], -1, 1, cv.FILLED)

    best = None
    for want in (3, 4):
        for eps in (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.14, 0.18):
            approx = cv.approxPolyDP(hull, eps * peri, True)
            if len(approx) != want:
                continue
            points = approx.reshape(-1, 2).astype(np.float32)
            edges = _edges(points)
            if edges[-1] <= 0 or edges[0] < MIN_EDGE_RATIO * edges[-1]:
                continue
            fitted = np.zeros_like(canvas)
            cv.fillPoly(fitted, [np.rint(points - origin).astype(np.int32)], 1)
            union = np.count_nonzero(canvas | fitted)
            iou = np.count_nonzero(canvas & fitted) / union if union else 0.0
            value = iou - (0.04 if want == 4 else 0.0)      # 顶点更少者优先
            if best is None or value > best[0]:
                best = (value, points, want, edges[0] / edges[-1], iou)
        if want == 3 and best is not None and best[4] >= 0.94:
            break
    if best is None:
        return None, 0, None, 0.0
    return best[1], best[2], float(best[3]), float(best[4])


def _to_mm(points):
    """把像素顶点经单应映射到毫米。"""
    h = np.asarray(HOMOGRAPHY, dtype=float)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv.perspectiveTransform(pts, h).reshape(-1, 2)


# ---------------------------------------------------------------- 分类与估计

def _assign_kinds(records):
    """按顶点数 + 面积排序 + 边比，把每块归到 L/M/T/Q/P。

    只用相对关系（谁比谁大），因此可在任意分辨率、任意相机下工作；
    绝对尺度由 _estimate_scale 给出。
    """
    tris = sorted([r for r in records if r["want"] == 3], key=lambda r: -r["area"])
    for rank, record in enumerate(tris):
        record["kind"] = "L" if rank < 2 else ("M" if rank == 2 else "T")
    for record in records:
        if record["want"] == 4:
            record["kind"] = "Q" if (record["edge_ratio"] or 0) >= SQUARE_EDGE_RATIO else "P"
    return records


def _scale_estimates(records):
    """由每块的每条边反推 S，返回 [(index, 边名, 值, S)]。"""
    out = []
    for index, record in enumerate(records):
        edges = record["edges"]
        if not edges:
            continue
        kind = record["kind"]
        if kind in ("L", "M", "T"):
            # 两条直角边分别出估计，不要只取较长的那条——否则标定时会系统性偏大
            picks = [("hyp", edges[2]), ("leg", edges[0]), ("leg", edges[1])]
        elif kind == "Q":
            picks = [("side", float(np.mean(edges)))]
        else:
            picks = [("long", edges[-1]), ("short", edges[0])]
        for name, value in picks:
            coefficient = piece_set.EDGE_COEF[kind][name]
            out.append((index, name, value, value / coefficient))
    return out


# ---------------------------------------------------------------- 主流程

def measure(img, mm=False, label="", expect=None):
    """测量一张图。expect = (kind, edge_name, value) 时用该件的实测值做绝对标定。"""
    unit = "mm" if mm else "px"
    contours = _contours(img)
    report = {"label": label, "unit": unit, "contour_count": len(contours), "pieces": []}
    if len(contours) != 7:
        report["warning"] = "找到 %d 块（应为 7 块）；缺块或黏连会让 S 不可信" % len(contours)
    if not contours:
        report["error"] = "没有找到任何块"
        return report

    records = []
    for contour in contours:
        points, want, edge_ratio, iou = _fit_shape(contour)
        if points is None:
            continue
        shape = _to_mm(points) if mm else points
        edges = _edges(shape)
        # 面积用同一套顶点算，保证与本工具的尺度口径一致
        area = abs(float(cv.contourArea(shape.astype(np.float32))))
        records.append({"points": shape, "want": want, "edge_ratio": edge_ratio,
                        "iou": iou, "edges": edges, "area": area, "kind": None})
    if not records:
        report["error"] = "没有任何一块拟合成功"
        return report
    _assign_kinds(records)

    estimates = _scale_estimates(records)
    if not estimates:
        report["error"] = "没有可用的边长估计"
        return report
    values = np.array([v[3] for v in estimates])
    median = float(np.median(values))
    keep = np.abs(values / median - 1.0) < OUTLIER_RATIO
    scale = float(values[keep].mean()) if keep.any() else median

    # 可选：用已知的一件实测尺寸做绝对标定。
    # 必须「同类同边」对比：拿大三角的直角边卡尺值去比斜边反推的 S 会差一个 √2。
    calibration = None
    if expect:
        kind, edge_name, value = expect
        measured = [val for i, name, val, _ in estimates
                    if records[i]["kind"] == kind and name == edge_name]
        if measured:
            observed = float(np.mean(measured))
            factor = value / observed
            calibration = {"kind": kind, "edge": edge_name, "expected": value,
                           "observed": observed, "factor": factor}
            scale *= factor

    # 逐块整理
    for index, record in enumerate(records):
        rows = [v for v in estimates if v[0] == index]
        # 该块全部边长估计都偏离中位数超过门槛时，标记为离群（未参与 S）
        deviations = [abs(est / median - 1.0) for _, _, _, est in rows]
        report["pieces"].append({
            "kind": record["kind"],
            "kind_name": piece_set.KIND_NAME.get(record["kind"], "未知"),
            "vertices": record["want"],
            "edges": [round(e, 2) for e in record["edges"]],
            "fit_iou": round(record["iou"], 3),
            "scale_estimates": [{"edge": name, "value": round(val, 2),
                                 "scale": round(est * (calibration["factor"] if calibration else 1.0), 2)}
                                for _, name, val, est in rows],
            "outlier": bool(rows) and all(d >= OUTLIER_RATIO for d in deviations),
        })

    counts = {}
    for piece in report["pieces"]:
        counts[piece["kind"]] = counts.get(piece["kind"], 0) + 1
    report.update({
        "scale": scale,
        "scale_unit": unit,
        "scale_median": median,
        "estimates_used": int(keep.sum()),
        "estimates_total": int(len(values)),
        "scale_std": float(values[keep].std()) if keep.any() else float(values.std()),
        "calibration": calibration,
        "kind_counts": counts,
        "kind_counts_expected": dict(piece_set.COUNT),
        "complete": counts == dict(piece_set.COUNT),
        "outliers": [i for i, p in enumerate(report["pieces"]) if p["outlier"]],
    })

    # ---------------- 输出
    print("=" * 84)
    if label:
        print("图: %s" % label)
    print("找到 %d 块；拟合成功 %d 块；S 估计 %d 条（采用 %d 条）"
          % (len(contours), len(records), len(values), int(keep.sum())))
    if report.get("warning"):
        print("[警告] %s" % report["warning"])
    print("%-3s %-9s %-5s %-30s %-8s %s"
          % ("#", "类", "顶点", "实测边长 " + unit, "拟合IoU", "由各边反推 S"))
    for index, piece in enumerate(report["pieces"], 1):
        scales = "  ".join("%s=%.2f" % (s["edge"], s["scale"]) for s in piece["scale_estimates"])
        mark = "  ← 离群" if piece["outlier"] else ""
        print("%-3d %-9s %-5d %-30s %-8.3f %s%s"
              % (index, piece["kind_name"], piece["vertices"],
                 ", ".join("%.2f" % e for e in piece["edges"]), piece["fit_iou"], scales, mark))

    print("-" * 84)
    print("S = %.2f %s（中位 %.2f，标准差 %.2f，采用 %d/%d 条）"
          % (scale, unit, median, report["scale_std"], int(keep.sum()), len(values)))
    if calibration:
        print("已按 %s 的 %s 卡尺值 %.2f %s 做绝对标定：图中实测 %.2f，修正系数 %.4f"
              % (piece_set.KIND_NAME.get(calibration["kind"], calibration["kind"]),
                 calibration["edge"], calibration["expected"], unit,
                 calibration["observed"], calibration["factor"]))
    if report["complete"]:
        print("类别数量正确：2 大三角 / 1 中三角 / 2 小三角 / 1 正方形 / 1 平行四边形")
    else:
        print("[警告] 类别数量不符：实测 %s，应为 %s" % (counts, dict(piece_set.COUNT)))
    if report["outliers"]:
        print("[警告] 第 %s 块离群，未参与 S：通常是被高光吃掉一角或贴边，属判据问题"
              % [i + 1 for i in report["outliers"]])
    if keep.sum() < 8:
        print("[警告] 只有 %d 条估计可用（正常应 ≥8）；建议调整光照/摆放后重拍" % int(keep.sum()))
    print("→ 件库可直接用这段：S_phys = %.2f %s" % (scale, unit))
    if not calibration:
        print("  提示：若手上有卡尺值，加 --expect-kind/--expect-value 可做一次绝对标定，"
              "消除系统性偏差")
    return report


def _load_image(path, already_undistorted):
    img = cv.imread(path)
    if img is None:
        raise FileNotFoundError("无法读取图像: %s" % path)
    if already_undistorted:
        return img
    return cv.undistort(img, CAMERA_MATRIX, DIST_COEFFS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", help="图像路径")
    parser.add_argument("--camera", action="store_true", help="直接用相机拍一张")
    parser.add_argument("--px", action="store_true",
                        help="像素尺度，只比比例（任意分辨率/相机可用）")
    parser.add_argument("--already-undistorted", action="store_true")
    parser.add_argument("--expect-kind", choices=sorted(piece_set.EDGE_COEF),
                        help="用哪一类的实测值做绝对标定")
    parser.add_argument("--expect-edge", choices=["hyp", "leg", "side", "long", "short"],
                        default="leg", help="该类的哪条边（默认 leg）")
    parser.add_argument("--expect-value", type=float, help="该条边的卡尺实测值（mm）")
    parser.add_argument("--out", help="输出目录")
    args = parser.parse_args(argv)

    if not args.image and not args.camera:
        parser.error("请提供 --image 或 --camera")
    if bool(args.expect_kind) != bool(args.expect_value is not None):
        parser.error("--expect-kind 与 --expect-value 必须同时提供")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(RESULTS_DIR, "measure_" + stamp)
    os.makedirs(out_dir, exist_ok=True)

    if args.camera:
        from common.realsense_camera import RealSenseCamera
        camera = RealSenseCamera(**CAMERA_PARAMS)
        try:
            camera.open()
            img = camera.grab_color(warmup=30)
        finally:
            try:
                camera.close()
            except Exception:
                pass
        cv.imwrite(os.path.join(out_dir, "capture_raw.png"), img)
        img = cv.undistort(img, CAMERA_MATRIX, DIST_COEFFS)
        label = "camera"
    else:
        img = _load_image(args.image, args.already_undistorted)
        label = os.path.basename(args.image)

    expect = (args.expect_kind, args.expect_edge, args.expect_value) if args.expect_kind else None
    report = measure(img, mm=not args.px, label=label, expect=expect)

    path = os.path.join(out_dir, "measure.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    print("-" * 84)
    print("报告: %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

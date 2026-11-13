"""离线求解检查：一张黑白剪影图 → 七块摆法 → 自检 → 可视化。

    # 用真实剪影图
    python -B -m scenarios.tangram_puzzle.tools.solve_check --image 某张剪影.png

    # 不接任何输入，跑内置回归图形（正方形）验证管线是否正常
    python -B -m scenarios.tangram_puzzle.tools.solve_check --synthetic square

    # 指定实物尺度（mm），输出会同时给出毫米位姿
    python -B -m scenarios.tangram_puzzle.tools.solve_check --image 剪影.png --scale 116.0

输出：results/solve_<时间戳>/ 下的
    silhouette.png      提取到的轮廓画在原图上
    solution.png        七块解叠加
    solution.json       归一化位姿 + 毫米位姿 + 自检报告

注意：本工具**不连接**相机或机器人。
"""

import argparse
import datetime
import json
import math
import os
import sys

import cv2 as cv
import numpy as np

from scenarios.tangram_puzzle.tangram import geometry as geo
from scenarios.tangram_puzzle.tangram import lattice_solver, verify

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _synthetic_polygon(name):
    """内置回归图形（归一化到面积 1）。"""
    if name == "square":
        polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elif name == "triangle":        # 公告 Fig 2 的示例：底 2、高 1 的等腰直角三角形
        polygon = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    elif name == "pentagon":        # 正五边形：面积同样归一化到 1，但不是七巧板能拼的图形
        angles = [2.0 * math.pi * k / 5.0 + 0.3 for k in range(5)]
        polygon = np.array([[math.cos(a), math.sin(a)] for a in angles])
    else:
        raise ValueError("未知的内置图形: %s" % name)
    normalized, _ = geo.normalize_polygon(polygon, target_area=1.0)
    return geo.ensure_ccw(normalized)


def _mm_placements(result, scale):
    """把归一化位姿换算成毫米位姿（位置×scale，角度不变）。"""
    out = []
    for item in result.placements:
        polygon = np.asarray(item["polygon"], dtype=float) * scale
        out.append({
            "kind": item["kind"], "index": item["index"], "angle": item["angle"],
            "center_mm": [round(float(v), 3) for v in geo.centroid(polygon)],
            "vertices_mm": [[round(float(x), 3), round(float(y), 3)] for x, y in polygon],
        })
    return out


def run(image_path=None, synthetic=None, scale=None, out_dir=None, time_limit=60.0,
        step=None):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir or os.path.join(RESULTS_DIR, "solve_" + stamp)
    os.makedirs(out_dir, exist_ok=True)

    summary = {"input": image_path or synthetic, "scale_mm": scale}

    if image_path:
        from scenarios.tangram_puzzle.vision import silhouette as silhouette_mod
        image = cv.imread(image_path)
        if image is None:
            raise FileNotFoundError("无法读取图像: %s" % image_path)
        silhouette = silhouette_mod.load(image_path)
        polygon = silhouette.polygon
        base_image = image
        summary["silhouette"] = {
            "vertices": len(polygon),
            "notes": silhouette.notes,
            "background_bgr": [round(float(v), 1) for v in silhouette.background],
            "snapped_edges": getattr(silhouette, "snapped_edges", 0),
        }
    else:
        polygon = _synthetic_polygon(synthetic or "square")
        silhouette_mod = None
        base_image = np.full((720, 720, 3), 32, np.uint8)
        summary["silhouette"] = {"vertices": len(polygon), "source": "synthetic"}

    print("=" * 78)
    print("轮廓：%d 个顶点，面积 %.6f（应为 1.0）" % (len(polygon), geo.area(polygon)))
    if summary["silhouette"].get("notes"):
        for note in summary["silhouette"]["notes"]:
            print("  提示：%s" % note)

    result = lattice_solver.solve(polygon, time_limit=time_limit,
                                  step=step or lattice_solver.LATTICE_STEP)
    if result is None:
        print("求解失败：在当前搜索预算内没有找到七块拼法")
        summary["solved"] = False
        _write(out_dir, summary, None, polygon, base_image, silhouette_mod, scale)
        return 1

    print("求解成功：%d 块，搜索节点 %d，耗时 %.2fs；格点=%s，吸附偏差 %.4f" % (
        len(result.placements), result.nodes, result.seconds,
        getattr(result, "basis", None), getattr(result, "snap_error", 0.0) or 0.0))
    reference = getattr(result, "target", None)
    if reference is None:
        reference = polygon
    report = verify.verify(result.placements, reference)
    print("自检：%s" % verify.describe(report))

    for item in sorted(result.placements, key=lambda x: (x["kind"], x["index"])):
        center = geo.centroid(item["polygon"])
        print("   %s%d  角度 %5.1f°  中心 (%.4f, %.4f)"
              % (item["kind"], item["index"], item["angle"], center[0], center[1]))

    summary.update({"solved": True, "verify": report,
                    "nodes": result.nodes, "seconds": round(result.seconds, 2),
                    "placements": [{"kind": p["kind"], "index": p["index"],
                                    "angle": p["angle"],
                                    "center": [round(float(v), 4) for v in geo.centroid(p["polygon"])],
                                    "vertices": [[round(float(x), 4), round(float(y), 4)]
                                                 for x, y in p["polygon"]]}
                                   for p in result.placements]})
    if scale:
        summary["placements_mm"] = _mm_placements(result, scale)
        print("\n毫米位姿（S = %.2f mm）：" % scale)
        for item in summary["placements_mm"]:
            print("   %s%d  中心 (%.2f, %.2f) mm  角度 %.1f°"
                  % (item["kind"], item["index"], item["center_mm"][0],
                     item["center_mm"][1], item["angle"]))

    _write(out_dir, summary, result, polygon, base_image, silhouette_mod, scale)
    return 0


def _write(out_dir, summary, result, polygon, base_image, silhouette_mod, scale):
    # 轮廓可视化
    if silhouette_mod is not None:
        rendered = silhouette_mod.render(base_image, _FakeSilhouette(polygon, scale), None)
        cv.imwrite(os.path.join(out_dir, "silhouette.png"), rendered)
    else:
        rendered = _draw_polygon(base_image, polygon, scale)
        cv.imwrite(os.path.join(out_dir, "silhouette.png"), rendered)

    if result is not None and silhouette_mod is not None:
        solution_image = silhouette_mod.render(base_image, _FakeSilhouette(polygon, scale), result)
    elif result is not None:
        solution_image = _draw_solution(base_image, result, scale)
    else:
        solution_image = rendered
    cv.imwrite(os.path.join(out_dir, "solution.png"), solution_image)

    path = os.path.join(out_dir, "solution.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print("-" * 78)
    print("输出目录: %s" % out_dir)
    print("  silhouette.png / solution.png / solution.json")


class _FakeSilhouette:
    """把裸多边形包装成 silhouette 接口，供 render 复用。"""

    def __init__(self, polygon, scale):
        self.polygon = polygon
        self.scale = scale or 1.0

    def scaled(self, scale):
        return np.asarray(self.polygon, dtype=float) * float(scale)


def _draw_polygon(image, polygon, scale):
    canvas = image.copy()
    height, width = canvas.shape[:2]
    span = max(geo.bounding_box(polygon)[2] - geo.bounding_box(polygon)[0],
               geo.bounding_box(polygon)[3] - geo.bounding_box(polygon)[1])
    pixels = 0.8 * min(height, width) / span
    center = np.array([width / 2.0, height / 2.0])
    points = np.asarray(polygon, dtype=float) * pixels + center
    cv.polylines(canvas, [np.rint(points).astype(np.int32)], True, (0, 255, 255), 2)
    return canvas


def _draw_solution(image, result, scale):
    canvas = _draw_polygon(image, np.vstack([p["polygon"] for p in result.placements]), scale)
    height, width = canvas.shape[:2]
    allpoints = np.vstack([p["polygon"] for p in result.placements])
    x0, y0, x1, y1 = geo.bounding_box(allpoints)
    span = max(x1 - x0, y1 - y0)
    pixels = 0.8 * min(height, width) / span
    center = np.array([width / 2.0, height / 2.0])
    palette = [(60, 60, 255), (60, 140, 255), (60, 255, 255), (60, 255, 60),
               (255, 180, 60), (255, 100, 200), (140, 90, 60)]
    for index, item in enumerate(result.placements):
        points = np.asarray(item["polygon"], dtype=float) * pixels + center
        color = palette[index % len(palette)]
        overlay = canvas.copy()
        cv.fillPoly(overlay, [np.rint(points).astype(np.int32)], color)
        canvas = cv.addWeighted(overlay, 0.35, canvas, 0.65, 0)
        cv.polylines(canvas, [np.rint(points).astype(np.int32)], True, color, 1)
        centroid = points.mean(axis=0)
        cv.putText(canvas, "%s%d" % (item["kind"], item["index"]),
                   (int(centroid[0]) - 14, int(centroid[1]) + 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", help="黑白剪影图路径")
    parser.add_argument("--synthetic", choices=["square", "triangle", "pentagon"],
                        help="内置图形：square 当前可解，triangle 是公告示例（当前不可解），pentagon 应被拒绝")
    parser.add_argument("--scale", type=float, help="实物尺度 S（mm），用于输出毫米位姿")
    parser.add_argument("--out", help="输出目录")
    parser.add_argument("--time-limit", type=float, default=60.0, help="求解时间上限（秒）")
    parser.add_argument("--step", type=float, default=lattice_solver.LATTICE_STEP,
                        help="格点步长（默认 S/4，一般不需要改）")
    args = parser.parse_args(argv)
    if not args.image and not args.synthetic:
        parser.error("请提供 --image 或 --synthetic")
    return run(image_path=args.image, synthetic=args.synthetic, scale=args.scale,
               out_dir=args.out, time_limit=args.time_limit, step=args.step)


if __name__ == "__main__":
    sys.exit(main())

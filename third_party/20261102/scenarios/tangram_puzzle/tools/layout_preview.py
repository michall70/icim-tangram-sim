"""看"目标图形最终会摆在哪" —— 只依赖目标图 + 当前 ROI/锚点，不需要原料区有块。

    python -B -m scenarios.tangram_puzzle.tools.layout_preview --image <任意实拍帧>

输出（默认写到 ``results/layout_preview/``）：

    layout_full.png   整帧，成品区框 + 七块的目标位置（半透明填色 + 槽位名）
    layout_zoom.png   成品区 2× 放大

为什么单独做一个工具：``main`` 的 ``plan_preview.png`` 需要原料区先检出 7 块才生成；
而"图形摆成什么样"只取决于目标图、``template_roi`` 锚点和朝向，
即使原料区空着（块已经放进托盘）也能看。
"""

import argparse
import os

import cv2 as cv
import numpy as np

from common.calibration import CameraMapper
from scenarios.tangram_puzzle.config import (CAMERA_MATRIX, DETECT_PARAMS, DIST_COEFFS,
                                             HOMOGRAPHY, S_PHYS_MM, TARGET_AREA_MM,
                                             TARGET_IMAGE_PATH, TARGET_YAW_DEG)
from scenarios.tangram_puzzle.planning import target_image as ti

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "layout_preview")
PALETTE = [(60, 60, 255), (60, 140, 255), (60, 220, 255), (90, 220, 90),
           (255, 160, 60), (200, 90, 200), (60, 90, 160)]


def build_pieces(target_path, anchor, yaw, scale):
    placements, summary = ti.load_target(target_path)
    pieces = ti.build_template_pieces(placements, scale, anchor, yaw, HOMOGRAPHY)
    return pieces, placements, summary


def draw(image, pieces, roi, yaw, out_full, out_zoom):
    canvas = image.copy()
    x, y, w, h = roi if roi is not None else (0, 0, canvas.shape[1], canvas.shape[0])
    if roi is not None:
        cv.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), (0, 165, 255), 2)
        cv.putText(canvas, "finished area  yaw=%.1f" % yaw, (x + 4, max(18, y - 8)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    for index, piece in enumerate(pieces):
        points = np.rint(np.asarray(piece.vertices, float)).astype(np.int32)
        layer = canvas.copy()
        cv.fillPoly(layer, [points], PALETTE[index % len(PALETTE)])
        canvas = cv.addWeighted(layer, 0.45, canvas, 0.55, 0)
        cv.polylines(canvas, [points], True, (255, 255, 255), 2)
        center = points.mean(axis=0).astype(int)
        label = str(piece.color_name)
        cv.putText(canvas, label, (center[0] - 16, center[1] + 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv.putText(canvas, label, (center[0] - 16, center[1] + 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.imwrite(out_full, canvas)
    pad = 40
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(canvas.shape[1], x + w + pad), min(canvas.shape[0], y + h + pad)
    # 成品区可能根本不在画面里（用 TARGET_AREA_QUAD_MM 的场合）：此时裁剪区域为空或只剩几条像素，
    # cv.resize 会报错或生成退化的图。这种情况只输出整帧，不生成放大图。
    if x1 - x0 < 60 or y1 - y0 < 60:
        print("[layout_preview] 成品区不在画面内，跳过放大图（这是正常的：四角配置不依赖画面）")
        return out_full, None
    crop = cv.resize(canvas[y0:y1, x0:x1], None, fx=2.0, fy=2.0, interpolation=cv.INTER_CUBIC)
    cv.imwrite(out_zoom, crop)
    return out_full, out_zoom


def main(argv=None):
    parser = argparse.ArgumentParser(description="预览目标图形在成品区的落位")
    parser.add_argument("--image", required=True, help="任意实拍帧（块在不在原料区都行）")
    parser.add_argument("--target", default=TARGET_IMAGE_PATH)
    parser.add_argument("--anchor", nargs=2, type=float, metavar=("X", "Y"),
                        help="台面毫米锚点；默认取 config.TARGET_AREA_QUAD_MM 四角中心或 template_roi 中心")
    parser.add_argument("--scale", type=float, default=S_PHYS_MM)
    parser.add_argument("--yaw", type=float, default=None, help="图案朝向；默认自动对齐成品区")
    parser.add_argument("--already-undistorted", action="store_true")
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args(argv)

    image = cv.imread(args.image, cv.IMREAD_COLOR)
    if image is None:
        raise SystemExit("无法读取图片: %s" % args.image)
    if not args.already_undistorted:
        image = CameraMapper(CAMERA_MATRIX, DIST_COEFFS, HOMOGRAPHY).undistort(image)

    roi = DETECT_PARAMS["template_roi"]
    from scenarios.tangram_puzzle.config import TARGET_AREA_QUAD_MM
    area_quad = ti.quad_from_config(TARGET_AREA_QUAD_MM)
    crop_box = roi           # 放大图裁剪区域：四角模式下改成"四角投影的包围框"
    if area_quad is not None:
        quad = area_quad
        anchor = ([float(args.anchor[0]), float(args.anchor[1])] if args.anchor
                  else ti.quad_center(area_quad))
        # 把四角反投影回像素，按它们的包围框裁剪（此时 template_roi 完全不参与）
        inverse = np.linalg.inv(np.asarray(HOMOGRAPHY, float))
        homogeneous = np.column_stack([area_quad, np.ones(4)]) @ inverse.T
        if np.all(np.abs(homogeneous[:, 2]) > 1e-9):
            pixels = homogeneous[:, :2] / homogeneous[:, 2, None]
            bx0, by0 = pixels.min(axis=0)
            bx1, by1 = pixels.max(axis=0)
            crop_box = (int(bx0), int(by0), int(max(1, bx1 - bx0)), int(max(1, by1 - by0)))
    else:
        quad = ti.tray_quad_mm(roi, HOMOGRAPHY)
        anchor = ([float(args.anchor[0]), float(args.anchor[1])] if args.anchor
                  else ti.anchor_from_roi(roi, HOMOGRAPHY))
    yaw = args.yaw
    if yaw is None:
        yaw = 0.0 if TARGET_YAW_DEG is None else float(TARGET_YAW_DEG)
    if True:
        # 统一按真四边形判定（四角配置，或画面里的框），放不下先试"与托盘平行"
        pieces, _, summary = build_pieces(args.target, anchor, yaw, args.scale)
        inside, worst = ti.vertices_inside_quad(pieces, quad)
        if not inside and args.yaw is None:
            auto_yaw = ti.quad_yaw_deg(area_quad) if area_quad is not None else ti.tray_yaw_deg(roi, HOMOGRAPHY)
            candidate, _, summary2 = build_pieces(args.target, anchor, auto_yaw, args.scale)
            inside2, worst2 = ti.vertices_inside_quad(candidate, quad)
            if inside2 and worst2 > worst:
                pieces, summary, yaw, inside, worst = candidate, summary2, auto_yaw, inside2, worst2

    os.makedirs(args.out, exist_ok=True)
    full, zoom = draw(image, pieces, crop_box, yaw,
                      os.path.join(args.out, "layout_full.png"),
                      os.path.join(args.out, "layout_zoom.png"))
    print("类别 %s；块数 %d" % (summary.get("kinds"), len(pieces)))
    if area_quad is not None:
        print("锚点 (%.2f, %.2f) mm；yaw=%.1f°；顶点都在成品区四边形内：%s（最小内距 %.1f mm）"
              % (anchor[0], anchor[1], yaw, "是" if inside else "否", worst))
        print("成品区来自 TARGET_AREA_QUAD_MM（四角，朝向 %.1f°）；目标图 %s"
              % (ti.quad_yaw_deg(area_quad), os.path.basename(args.target)))
    else:
        print("锚点 (%.2f, %.2f) mm；yaw=%.1f°；顶点都在成品区内：%s（最小内距 %.1f mm）"
              % (anchor[0], anchor[1], yaw, "是" if inside else "否", worst))
        print("成品区来自 template_roi 像素框；目标图 %s" % os.path.basename(args.target))
    print("整帧: %s" % full)
    print("放大: %s" % zoom)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

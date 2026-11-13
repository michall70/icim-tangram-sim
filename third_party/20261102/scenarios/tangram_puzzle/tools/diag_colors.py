"""逐色掩模诊断：把一个颜色"从像素到结果"的全过程摊开，定位它到底死在哪一步。

## 为什么需要

`_detect_color` 内部要过好几道关（HSV 取掩模 → 清理 → 取轮廓 → 形状/面积校验 → 多色仲裁），
最终只输出"成功"的块，中间过程全部丢弃。于是"少了两块"时只能猜，猜不准。

本工具对七种颜色分别输出：

    原始 HSV 掩模像素数 → 清理后像素数 → 最大连通域面积 → 最终是否采纳 + 当步产生的说明

并把四联图（原始掩模 / 清理掩模 / 拟合多边形 / 原始画面）存到 results/ 下，便于肉眼核对。

## 用法

    python -B -m scenarios.tangram_puzzle.tools.diag_colors --image <实拍帧> [--area raw|template]

只读，不连机器人、不动硬件。
"""

import argparse
import os
import sys

import cv2 as cv
import numpy as np

from common.calibration import CameraMapper
from scenarios.tangram_puzzle.config import (CAMERA_MATRIX, DETECT_PARAMS, DIST_COEFFS,
                                             HOMOGRAPHY)
from scenarios.tangram_puzzle.main import RESULTS_DIR
from scenarios.tangram_puzzle.tangram_detector import TangramDetector


def _largest_blob(mask):
    count, _, stats, _ = cv.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8)
    if count <= 1:
        return 0
    return int(max(stats[i, cv.CC_STAT_AREA] for i in range(1, count)))


def diagnose(image_path, area="raw"):
    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError("无法读取图片: %s" % image_path)
    mapper = CameraMapper(CAMERA_MATRIX, DIST_COEFFS, HOMOGRAPHY)
    image = mapper.undistort(image)

    detector = TangramDetector(DETECT_PARAMS)
    detector._detect(image, None, DETECT_PARAMS.get("%s_roi" % area))

    roi = detector._roi
    x, y, w, h = roi
    masked = np.full_like(image, 255)
    masked[y:y + h, x:x + w] = image[y:y + h, x:x + w]
    hsv = cv.cvtColor(masked, cv.COLOR_BGR2HSV)
    prepared = detector._prepare_masks(hsv)

    rows = []
    tiles = []
    for spec in detector.specs:
        color_id = spec["id"]
        if "low" in spec and "high" in spec:
            low = np.array(spec["low"], np.uint8)
            high = np.array(spec["high"], np.uint8)
            raw_mask = cv.inRange(hsv, low, high)
        else:
            raw_mask = np.zeros(hsv.shape[:2], np.uint8)

        detector.last_notes = []
        detector.last_issues = []
        piece = detector._detect_color(hsv, spec, prepared.get(color_id))
        notes = list(detector.last_notes) + ["拒绝: " + x for x in detector.last_issues]

        clean_mask = prepared.get(color_id)
        rows.append({
            "id": color_id,
            "name": spec.get("name", "?"),
            "raw_px": int((raw_mask > 0).sum()),
            "clean_px": int((clean_mask > 0).sum()) if clean_mask is not None else 0,
            "largest": _largest_blob(clean_mask) if clean_mask is not None else 0,
            "ok": piece is not None,
            "area": abs(cv.contourArea(np.asarray(piece.vertices, np.float32)))
            if piece is not None else 0.0,
            "notes": notes,
        })

        panels = [cv.cvtColor(raw_mask, cv.COLOR_GRAY2BGR),
                  cv.cvtColor(clean_mask if clean_mask is not None
                              else np.zeros_like(raw_mask), cv.COLOR_GRAY2BGR),
                  image.copy(),
                  np.full((240, 360, 3), 255, np.uint8)]
        if piece is not None:
            cv.polylines(panels[2], [np.asarray(piece.vertices, np.int32)], True, (0, 0, 255), 2)
            cv.putText(panels[3], "OK %s" % piece.shape, (10, 120),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 128, 0), 2)
        else:
            cv.putText(panels[3], "REJECTED", (10, 120),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 2)
        strip = np.hstack([cv.resize(p, (360, 240)) for p in panels])
        cv.putText(strip, "ID=%s %s" % (color_id, spec.get("name", "")), (8, 26),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        tiles.append(strip)

    return rows, tiles, roi


def main(argv=None):
    parser = argparse.ArgumentParser(description="逐色掩模诊断（只读）")
    parser.add_argument("--image", required=True)
    parser.add_argument("--area", choices=("raw", "template"), default="raw")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    rows, tiles, roi = diagnose(args.image, args.area)
    print("ROI = %s" % (roi,))
    print("%-4s %-9s %9s %10s %10s %8s %s"
          % ("ID", "颜色", "原始掩模px", "清理后px", "最大连通域", "结果", "说明"))
    for row in rows:
        print("%-4s %-9s %9d %10d %10d %8s %s"
              % (row["id"], row["name"], row["raw_px"], row["clean_px"], row["largest"],
                 ("%.0fmm2" % row["area"]) if row["ok"] else "拒",
                 "; ".join(row["notes"])[:52]))

    accepted = [row["id"] for row in rows if row["ok"]]
    print("\n采纳 %d/7：%s" % (len(accepted), accepted))
    missing = [row for row in rows if not row["ok"]]
    for row in missing:
        print("  缺 ID=%s(%s)：原始掩模 %d px、清理后 %d px、最大连通域 %d px"
              % (row["id"], row["name"], row["raw_px"], row["clean_px"], row["largest"]))

    out_dir = args.out or os.path.join(RESULTS_DIR, "diag_colors")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    for row, tile in zip(rows, tiles):
        cv.imwrite(os.path.join(out_dir, "%s_%s_%s.png" % (stem, args.area, row["id"])), tile)
    print("\n四联图已保存到 %s（每个颜色一张）" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

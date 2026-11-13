"""离线阶段三检查：原始彩色图 -> 去畸变 -> 分区检测 -> 标注与 JSON。

python -m scenarios.tangram_puzzle.detect_check --mode raw image.png
输入已去畸变图时必须加 --already-undistorted，避免二次校正。不连接设备。
"""
import argparse
import datetime
import json
from pathlib import Path

import cv2 as cv
from common.calibration import CameraMapper
from scenarios.tangram_puzzle.config import CAMERA_MATRIX, DIST_COEFFS, HOMOGRAPHY, DETECT_PARAMS
from scenarios.tangram_puzzle.tangram_detector import TangramDetector


def process_image(detector, path, mode, out_dir, already_undistorted=False):
    image = cv.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    if not already_undistorted:
        image = CameraMapper(CAMERA_MATRIX, DIST_COEFFS, HOMOGRAPHY).undistort(image)
    annotated = image.copy()
    detect = detector.detect_raw_material if mode == "raw" else detector.detect_template
    pieces = detect(image, annotated=annotated)
    report = {
        "input": str(Path(path).resolve()), "mode": mode,
        "input_already_undistorted": already_undistorted,
        "ids": [p.piece_id for p in pieces], "issues": list(detector.last_issues),
        "pieces": [{"id": p.piece_id, "name": p.color_name,
                    "center": p.grasp_point, "vertices": p.vertices,
                    "angle": p.angle} for p in pieces],
        "detection_config": DETECT_PARAMS,
        "camera_matrix": CAMERA_MATRIX.tolist(), "dist_coeffs": DIST_COEFFS.tolist(),
        "note": "Offline detection only; not robot positioning validation."
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem + "_" + mode
    cv.putText(annotated, f"{mode}: {len(pieces)}/7 | issues: {len(detector.last_issues)}",
               (15,35), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    if not cv.imwrite(str(out_dir / (stem + "_annotated.png")), annotated):
        raise OSError("无法保存标注图")
    (out_dir / (stem + ".json")).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{Path(path).name} {mode}: IDs={report['ids']}, issues={report['issues']}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--mode", choices=("raw", "template"), required=True)
    parser.add_argument("--already-undistorted", action="store_true")
    parser.add_argument("--out", default=str(Path(__file__).parent / "results" / ("detect_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))))
    args = parser.parse_args()
    detector = TangramDetector(DETECT_PARAMS)
    for path in args.images:
        process_image(detector, path, args.mode, args.out, args.already_undistorted)

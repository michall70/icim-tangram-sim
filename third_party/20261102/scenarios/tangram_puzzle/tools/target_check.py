"""目标图（赛方给的七块分色图）→ 台面位姿：离线核对工具。

    # 只看目标图本身解析得对不对
    python -B -m scenarios.tangram_puzzle.tools.target_check --target <目标图>

    # 目标图 + 一张原料区实拍帧 → 生成 7 条抓放计划并自检
    python -B -m scenarios.tangram_puzzle.tools.target_check --target <目标图> --raw <原料帧>

    # 旋转图案、换实物尺度
    python -B -m scenarios.tangram_puzzle.tools.target_check --target <图> --raw <帧> --yaw 90 --scale 115.84

与 `main` 的关系：本工具只读，不连相机、不连机器人；它验证的是"目标图这条新的模板来源"
能不能产出合理计划。`main` 尚未接线，接线前先在这里把数字看住。
"""

import argparse
import json
import os
import sys

import cv2 as cv
import numpy as np

from common.calibration import CameraMapper
from scenarios.tangram_puzzle.config import (CAMERA_MATRIX, DETECT_PARAMS, DIST_COEFFS,
                                             GRIP_Z, HOMOGRAPHY, PLACE_ANGLE_OFFSET,
                                             PLACE_Z, S_PHYS_MM, SUCTION_OFFSET_XY,
                                             TARGET_AREA_MM, TCP_RPY)
from scenarios.tangram_puzzle.planning import assign, target_image
from scenarios.tangram_puzzle.puzzle_planner import (plan_puzzle, predicted_piece_xy,
                                                     validate_plan_geometry)
from scenarios.tangram_puzzle.tangram_detector import TangramDetector


def _roi_polygon_mm(roi, matrix):
    x, y, w, h = roi
    corners = np.array([[x, y, 1.0], [x + w - 1, y, 1.0],
                        [x + w - 1, y + h - 1, 1.0], [x, y + h - 1, 1.0]])
    mapped = (matrix @ corners.T).T
    return mapped[:, :2] / mapped[:, 2, None]


def _inside_polygon(points, polygon):
    polygon = np.asarray(polygon, float)
    for point in np.asarray(points, float):
        if cv.pointPolygonTest(polygon.astype(np.float32), (float(point[0]), float(point[1])),
                               False) < 0:
            return False
    return True


def run(target_path, raw_path=None, yaw=None, scale=None, out_dir=None,
        already_undistorted=False, anchor_mm=None):
    scale = float(scale if scale is not None else S_PHYS_MM)
    auto_yaw = yaw is None
    yaw = 0.0 if yaw is None else float(yaw)
    placements, summary = target_image.load_target(target_path)
    print("[target] %s" % target_image.describe(placements, summary))

    roi = DETECT_PARAMS["template_roi"]
    if anchor_mm is not None:
        anchor = [float(anchor_mm[0]), float(anchor_mm[1])]
        source = "命令行 --anchor"
    else:
        anchor = target_image.anchor_from_roi(roi, HOMOGRAPHY)
        source = "template_roi 中心"
    pieces = target_image.build_template_pieces(placements, scale, anchor, yaw, HOMOGRAPHY)
    quad = target_image.tray_quad_mm(roi, HOMOGRAPHY)
    inside, worst = target_image.vertices_inside_quad(pieces, quad)
    yaw_note = "yaw=%.1f°" % yaw
    # 与 main._load_target_pieces 一致：只有未显式指定朝向、且按 0° 放不下时才自动对齐。
    if auto_yaw and not inside:
        auto = target_image.tray_yaw_deg(roi, HOMOGRAPHY)
        candidate = target_image.build_template_pieces(placements, scale, anchor, auto, HOMOGRAPHY)
        inside2, worst2 = target_image.vertices_inside_quad(candidate, quad)
        if inside2:
            pieces, inside, worst = candidate, inside2, worst2
            yaw = auto
            yaw_note = "yaw=%.1f°（自动对齐成品区）" % yaw
    fits, span = target_image.fits_area(pieces, TARGET_AREA_MM)
    ok = inside
    _, bbox = target_image.pixels_inside(pieces, roi, margin=2)
    print("[target] 锚点(%s) = (%.2f, %.2f) mm；%s；像素包围盒 = %s"
          % (source, anchor[0], anchor[1], yaw_note, [round(v, 1) for v in bbox]))
    print("[target] 图案 %.1f×%.1f mm（S_phys=%.2f），成品区 %.0f×%.0f mm；"
          "顶点是否都在成品区四边形内：%s（最小内距 %.1f mm）"
          % (span[0], span[1], scale, TARGET_AREA_MM[0], TARGET_AREA_MM[1],
             "是" if inside else "否 [警告]", worst))

    result = {"target": summary, "scale_mm": scale, "yaw_deg": yaw,
              "anchor_mm": anchor, "inside_roi": bool(ok), "pixel_bbox": bbox,
              "size_mm": [round(span[0], 2), round(span[1], 2)]}

    if raw_path:
        image = cv.imread(raw_path, cv.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError("无法读取原料帧: %s" % raw_path)
        mapper = CameraMapper(CAMERA_MATRIX, DIST_COEFFS, HOMOGRAPHY)
        if not already_undistorted:
            image = mapper.undistort(image)
        detectors_roi = DETECT_PARAMS["raw_roi"]
        x, y, w, h = detectors_roi
        value = cv.cvtColor(image[y:y + h, x:x + w], cv.COLOR_BGR2HSV)[:, :, 2]
        print("[raw] 原料 ROI 内 V 中位数 = %.0f（调阈值那天约 68；低于 40 基本必失败）"
              % float(np.median(value)))
        detector = TangramDetector(DETECT_PARAMS)
        raw_pieces = detector.detect_raw_material(image)
        print("[raw] 原料区检出 %d 块：%s" % (len(raw_pieces), sorted(p.piece_id for p in raw_pieces)))
        if detector.last_issues:
            for issue in detector.last_issues:
                print("       %s" % issue)
        if len(raw_pieces) != 7:
            raise SystemExit("原料区未检出 7 块，先解决检测，再谈目标图")

        mapping, report = assign.assign_ids(pieces, raw_pieces)
        print("[assign] %s" % assign.describe(report))
        pieces = target_image.build_template_pieces(placements, scale, anchor, yaw,
                                                    HOMOGRAPHY, piece_ids=mapping)

        plans = plan_puzzle(raw_pieces, pieces, mapper, TCP_RPY, GRIP_Z, PLACE_Z,
                            PLACE_ANGLE_OFFSET, SUCTION_OFFSET_XY)
        validate_plan_geometry(plans)
        roi_mm = _roi_polygon_mm(roi, np.asarray(HOMOGRAPHY, float))
        landings = [predicted_piece_xy(plan) for plan in plans]
        inside = _inside_polygon(landings, roi_mm)
        print("[plan] %d 条计划，落点是否都在成品区内：%s" % (len(plans), "是" if inside else "否 [警告]"))
        for plan in plans:
            print("   ID=%d(%s) 抓(%.1f,%.1f) 放(%.1f,%.1f) Δ=%.1f° 拟合=%.2fmm 疑似翻面=%s"
                  % (plan.piece_id, plan.color_name, plan.grasp_pose[0], plan.grasp_pose[1],
                     plan.place_pose[0], plan.place_pose[1], plan.delta_angle,
                     plan.fit_error_mm, plan.suspected_flip))
        result["plans"] = [{"id": p.piece_id, "color": p.color_name,
                            "grasp": [round(v, 3) for v in p.grasp_pose],
                            "place": [round(v, 3) for v in p.place_pose],
                            "delta_deg": round(p.delta_angle, 3),
                            "fit_mm": round(p.fit_error_mm, 3),
                            "flip": bool(p.suspected_flip)} for p in plans]
        result["landings_inside_area"] = bool(inside)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "target_check.json"), "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print("[out] %s" % os.path.join(out_dir, "target_check.json"))
    return 0 if result.get("inside_roi") and result.get("landings_inside_area", True) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True, help="赛方给的目标图（七块分色）")
    parser.add_argument("--raw", help="原料区实拍帧（去畸变前的原帧即可，工具内部会去畸变）")
    parser.add_argument("--yaw", type=float, default=None,
                        help="图案整体朝向（度）；缺省=自动（先 0°，放不下再与成品区平行）")
    parser.add_argument("--scale", type=float, help="实物尺度 S_phys（mm），缺省用 config.S_PHYS_MM")
    parser.add_argument("--out", help="输出目录（写 target_check.json）")
    parser.add_argument("--already-undistorted", action="store_true",
                        help="原料帧已经去畸变过（例如 results/<时间戳>/raw_frame.png），不要再畸变一次")
    parser.add_argument("--anchor", nargs=2, type=float, metavar=("X", "Y"),
                        help="图案形心的台面坐标（mm）；给了就不再用 template_roi 中心")
    args = parser.parse_args(argv)
    return run(args.target, args.raw, args.yaw, args.scale, args.out,
               already_undistorted=args.already_undistorted, anchor_mm=args.anchor)


if __name__ == "__main__":
    sys.exit(main())

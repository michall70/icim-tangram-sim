"""七巧板拼图场景入口。

目标来源有两种（`config.TARGET_SOURCE`）：
- `image`（默认，赛场方案）：目标图文件 → 七块布局 → 按 S_PHYS_MM 缩放 →
  锚定成品区 → 复用同一套规划。目标图不是相机帧，因此不套单应。
- `photo`（回退方案）：`config.TEMPLATE_PATH` 指向拍照位实拍模板，与原料图共用单应。

在线流程：回 HOME → 移动到拍照位采图 → 检测原料区散乱七巧板 → 生成目标位姿 →
逐块抓取并放置 → 回 HOME。实机必须显式 `--execute`。

离线调试：
- 目标图模式：`python -m scenarios.tangram_puzzle.main --raw <原料原图>`
- 实拍模板：`python -m scenarios.tangram_puzzle.main <模板图> <原料图>`
输入为相机原始图像，内部统一去畸变；只做视觉与规划，不连接机器人与相机。
"""

import datetime
import json
from dataclasses import asdict
import os

import cv2 as cv
import numpy as np

from common.calibration import CameraMapper
from scenarios.tangram_puzzle.config import (APPROACH_DZ, CAMERA_MATRIX, CAMERA_PARAMS,
                                             CAPTURE_POSE, DETECT_PARAMS, DIST_COEFFS,
                                             GRIPPER_ACTION_TIME, GRIPPER_CLOSE_LEVEL, GRIPPER_DO,
                                             GRIP_Z, HOME_JOINT, HOMOGRAPHY, PLACE_ANGLE_OFFSET,
                                             PLACE_Z, ROBOT_IP, S_PHYS_MM, ENABLE_SCALE_NORMALIZATION, SDK_VERSION,
                                             SUCTION_OFFSET_XY, TCP_RPY, TEMPLATE_PATH, TOOL,
                                             TARGET_AREA_MM, TARGET_AREA_QUAD_MM,
                                             TARGET_IMAGE_PATH,
                                             TARGET_SOURCE, TARGET_YAW_DEG,
                                             USER, VEL)
from scenarios.tangram_puzzle.planning.scale import describe as describe_scale, normalize_plans
from scenarios.tangram_puzzle.puzzle_planner import plan_puzzle, render_plan_preview, validate_plan_geometry
from scenarios.tangram_puzzle.tangram_detector import TangramDetector, validate_detection

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def run(template_path=None, raw_image_path=None, *, execute=False,
        target_source=None, target_image_path=None, use_frame=None) -> None:
    """执行七巧板拼图场景；两个图像路径都提供时仅离线运行视觉与规划。

    target_source="image" 时不需要模板图：目标位姿来自目标图文件
    （`TARGET_IMAGE_PATH` 或 `target_image_path`），实物尺度仍用 `S_PHYS_MM`。

    use_frame: 实机执行时改用**已存在的帧**（例如 capture_check 在拍照位存下的那张），
    不再现场抓帧。前提是这个帧确实是**在 CAPTURE_POSE 拍的**——单应只对拍照位有效；
    另外块从拍帧到现在不能被动过。省下的是 1—2 秒的抓帧时间。
    """
    source = (target_source or TARGET_SOURCE).lower()
    if source not in ("photo", "image"):
        raise ValueError("TARGET_SOURCE 只能是 'photo' 或 'image'，当前是 %r" % source)

    target_image = None
    if source == "image":
        target_image = target_image_path or TARGET_IMAGE_PATH
        if not os.path.isfile(target_image):
            raise FileNotFoundError(
                f"目标图不存在: {target_image}；请放到该路径，或改 config.TARGET_IMAGE_PATH")

    detector = TangramDetector(DETECT_PARAMS)
    # 标定只有 config.HOMOGRAPHY 一个来源；不再读取 calib.json。
    mapper = CameraMapper(CAMERA_MATRIX, DIST_COEFFS, HOMOGRAPHY)

    if raw_image_path and (source == "image" or template_path):
        template_img = None
        if source == "photo":
            template_img = cv.imread(template_path)
            if template_img is None:
                raise FileNotFoundError(f"无法读取模板图: {template_path}")
            template_img = mapper.undistort(template_img)
        raw_img = cv.imread(raw_image_path)
        if raw_img is None:
            raise FileNotFoundError(f"无法读取原料图: {raw_image_path}")
        plans = _run_vision(detector, mapper, template_img, mapper.undistort(raw_img),
                            target_image_path=target_image)
        print(f"[tangram_puzzle] 离线规划完成，共 {len(plans)} 块，结果已保存到 {RESULTS_DIR}")
        return

    if template_path or raw_image_path:
        if source == "photo":
            raise ValueError("离线模式必须同时提供模板原图和原料原图")
        raise ValueError("目标图模式离线只需 --raw <原料帧>；模板位置参数请留空")
    if not execute:
        raise ValueError("默认不运行设备：请提供图片离线检查；实机需显式 --execute")

    from common.gripper import RobotDOGripper
    from common.realsense_camera import RealSenseCamera
    from common.robot_client import CobotClient

    if not target_image and not os.path.isfile(TEMPLATE_PATH):
        raise FileNotFoundError(f"模板图不存在: {TEMPLATE_PATH}，请在拍照位实拍拼好的成品图放入 data/")

    camera = RealSenseCamera(**CAMERA_PARAMS)
    robot = CobotClient(ROBOT_IP, tool=TOOL, user=USER, sdk_version=SDK_VERSION)
    context = "initialization"
    try:
        if use_frame:
            # 复用已有帧：不开相机（省 1—2 秒），但必须先确认机械臂就在拍照位，
            # 否则单应不成立，算出来的坐标会整体偏。
            print("[tangram_puzzle] 使用已存在的帧：%s（不现场抓帧）" % use_frame)
        else:
            camera.open()
        if CAMERA_PARAMS.get("exposure") is None:
            print("[tangram_puzzle] [警告] 相机为自动曝光：同一块在不同光照下 HSV 会漂移，"
                  "颜色阈值不稳定。建议先运行 capture_check 固定曝光/白平衡并写回 "
                  "config.CAMERA_PARAMS，然后重拍模板图。")
        robot.connect()
        robot.check_coordinate_context()
        gripper = RobotDOGripper(robot, GRIPPER_DO, GRIPPER_CLOSE_LEVEL, GRIPPER_ACTION_TIME)
        gripper.release()
        context = "capture_position"
        robot.move_home(HOME_JOINT, vel=VEL)
        robot.move_to(CAPTURE_POSE, vel=VEL)

        print("[tangram_puzzle] 1) 拍照并检测原料区，生成目标位姿与抓放计划")
        if use_frame:
            reused = cv.imread(use_frame, cv.IMREAD_COLOR)
            if reused is None:
                raise FileNotFoundError("use_frame 指定的帧读不到: %s" % use_frame)
            raw_img = mapper.undistort(reused)
        else:
            raw_img = mapper.undistort(camera.grab_color(warmup=30))
        template_img = None
        if not target_image:
            template_img = mapper.undistort(cv.imread(TEMPLATE_PATH))
        plans = _run_vision(detector, mapper, template_img, raw_img,
                            target_image_path=target_image)
        if not plans:
            raise RuntimeError("未生成任何抓放计划，请检查原料区检测与模板图")

        context = "geometry_check"
        validate_plan_geometry(plans)
        context = "all_pieces_preflight"
        checks = robot.preflight_pick_place(plans, APPROACH_DZ)
        _save_json(detector.last_output_dir, "ik_preflight.json", checks)
        print(f"[tangram_puzzle] 逆解预检通过（{len(checks)}个端点；不代表路径无碰撞）")
        print(f"[tangram_puzzle] 2) 按计划顺序逐块抓放（共 {len(plans)} 块）")
        for i, plan in enumerate(plans, 1):
            print(f"  第{i}块 ID={plan.piece_id}({plan.color_name}) "
                  f"抓取(x={plan.grasp_pose[0]:.1f}, y={plan.grasp_pose[1]:.1f}) "
                  f"放置(x={plan.place_pose[0]:.1f}, y={plan.place_pose[1]:.1f}, rz={plan.place_pose[5]:.1f})")
            context = f"ID={plan.piece_id}/pre_pick_check"
            robot.preflight_pick_place([plan], APPROACH_DZ)
            context = f"ID={plan.piece_id}/pick"
            robot.pick(plan.grasp_pose, gripper, APPROACH_DZ, vel=VEL)
            context = f"ID={plan.piece_id}/place"
            robot.place(plan.place_pose, gripper, APPROACH_DZ, vel=VEL)

        context = "return_home"
        robot.move_home(HOME_JOINT, vel=VEL)
        print("[tangram_puzzle] 抓放指令执行结束，已回 HOME（实际拼装结果待检查）")
    except Exception as exc:
        folder = getattr(detector, "last_output_dir", os.path.join(RESULTS_DIR, datetime.datetime.now().strftime("error_%Y%m%d_%H%M%S_%f")))
        try:
            _save_json(folder, "execution_error.json", {
                "context":context, "error":str(exc), "details":getattr(exc,"details",None),
                "note":"No automatic recovery, release, retry or target adjustment was performed."})
        except Exception as log_error:
            print(f"无法保存故障日志: {log_error}")
        raise
    finally:
        # 清理失败不能覆盖原始逆解/运动异常；不在此自动松开吸盘或回HOME。
        for close in (camera.close,robot.disconnect):
            try:
                close()
            except Exception as cleanup_error:
                print(f"设备断开异常: {cleanup_error}")


def _save_json(folder, name, data):
    os.makedirs(folder,exist_ok=True)
    with open(os.path.join(folder,name),"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)



def quantize_template_angles(template_pieces):
    """把模板各块的落位角度吸附到 45° 网格。

    为什么：七巧板拼成任何图形时，每块相对图形的朝向都只可能是 45° 的整数倍。
    但 `_polygon_angle` 对**平行四边形**不成立——它两条边互相斜交，主方向估计会偏，
    实测偏了 18.95°，落位后那块明显歪掉（2026-10-29 实机照片可见）。
    其余各块实测偏差均在 ±0.7° 内，吸附对它们几乎无影响。

    图形的整体朝向可以是任意角，故先取各块"角度 mod 45"的中位数作为基准偏移，
    再吸附到 基准 + k·45。只改角度，不动位置。
    """
    pieces = [pc for pc in template_pieces if pc is not None]
    if not pieces:
        return template_pieces
    offsets = [float(pc.angle) % 45.0 for pc in pieces]
    reference = float(np.median(offsets))
    for pc in pieces:
        angle = float(pc.angle)
        snapped = round((angle - reference) / 45.0) * 45.0 + reference
        try:
            pc.angle = float(snapped % 360.0)
        except Exception:
            pass
    return template_pieces

def _load_target_pieces(detector, mapper, raw_pieces, target_image_path):
    """目标图模式：目标图 → 七块 → 指派到实物 ID → 模板块列表。

    与照片模板的区别：目标图不是相机帧，**不套单应**；它只提供形状，
    实物尺度由 `S_PHYS_MM` 决定。

    成品区（决定图案摆在哪、放不放得下）有两种来源：
      · `config.TARGET_AREA_QUAD_MM` 有四角 → 用它（推荐：能给出朝向，成品区不必在画面里）；
      · 为 None → 退回 `template_roi` 像素框（成品区必须在画面里）。
    """
    from scenarios.tangram_puzzle.planning import assign, target_image

    placements, summary = target_image.load_target(target_image_path)

    area_quad = target_image.quad_from_config(TARGET_AREA_QUAD_MM)
    if area_quad is not None:
        anchor = target_image.quad_center(area_quad)
        anchor_source = "config.TARGET_AREA_QUAD_MM 四角中心"
        quad = area_quad
        auto_yaw = target_image.quad_yaw_deg(area_quad)
    else:
        anchor = target_image.anchor_from_roi(DETECT_PARAMS["template_roi"], mapper.homography)
        anchor_source = "template_roi 中心"
        quad = target_image.tray_quad_mm(DETECT_PARAMS["template_roi"], mapper.homography)
        auto_yaw = target_image.tray_yaw_deg(DETECT_PARAMS["template_roi"], mapper.homography)

    # 朝向：显式给了就照办；没给就先按 0°，放不下再调。
    yaw = 0.0 if TARGET_YAW_DEG is None else float(TARGET_YAW_DEG)
    pieces = target_image.build_template_pieces(placements, S_PHYS_MM, anchor, yaw,
                                                mapper.homography)
    inside, worst = target_image.vertices_inside_quad(pieces, quad)
    yaw_note = "yaw=%.1f°" % yaw

    if not inside and TARGET_YAW_DEG is None:
        # 放不下就转到与成品区平行（四角配置能给出真实朝向；像素框也能）
        candidate = target_image.build_template_pieces(placements, S_PHYS_MM, anchor, auto_yaw,
                                                       mapper.homography)
        inside2, worst2 = target_image.vertices_inside_quad(candidate, quad)
        if inside2 and worst2 > worst:
            pieces, inside, worst = candidate, inside2, worst2
            yaw = auto_yaw
            yaw_note = "yaw=%.1f°（自动对齐成品区）" % yaw

    if not inside:
        raise ValueError("图案放不进成品区：最差顶点超出 %.1f mm（%s）；"
                         "可换更小的图形，或把 TARGET_YAW_DEG 调成与成品区平行（约 %.1f°）"
                         % (-worst, yaw_note, auto_yaw))
    fits, span = target_image.fits_area(pieces, TARGET_AREA_MM)
    if not fits:
        # 旋转后轴对齐包围盒会变大，但顶点可能仍在成品区四边形内。
        # 因此这里只提示，不以它否决；真正的判据是上面的顶点内距。
        print("[target] [警告] 轴对齐包围盒 %.1f×%.1fmm 超过 %.0f×%.0fmm，"
              "但顶点都在成品区内（旋转摆放），继续"
              % (span[0], span[1], TARGET_AREA_MM[0], TARGET_AREA_MM[1]))
    mapping, report = assign.assign_ids(pieces, raw_pieces)
    print("[target] %s" % target_image.describe(placements, summary))
    print("[target] 锚点=%s (%.2f, %.2f) mm；%s；图案 %.1f×%.1fmm；最小内距 %.1f mm；指派 %s"
          % (anchor_source, anchor[0], anchor[1], yaw_note, span[0], span[1], worst,
             assign.describe(report)))
    return target_image.build_template_pieces(placements, S_PHYS_MM, anchor, yaw,
                                              mapper.homography, piece_ids=mapping), summary


def _run_vision(detector, mapper, template_img, raw_img, target_image_path=None):
    """模板检测 + 原料区检测（ROI 外涂白）+ 匹配规划，结果存入 results/，返回抓放计划。

    `target_image_path` 非空时走"目标图"通道：模板位置来自赛方给的图片，
    而不是相机拍摄的模板图；其余（几何门槛、IK 预检、执行）完全一致。
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = os.path.join(RESULTS_DIR, stamp)
    os.makedirs(out_dir, exist_ok=True)
    detector.last_output_dir = out_dir

    raw_vis = raw_img.copy()
    raw_pieces = detector.detect_raw_material(raw_img, annotated=raw_vis)
    raw_issues = list(detector.last_issues)
    # 即使失败也保存标注，便于诊断；禁止部分检测结果进入抓放规划。
    cv.imwrite(os.path.join(out_dir, "raw_annotated.png"), raw_vis)
    # 保存未标注原始帧：出问题时能精确回放检测细节（标注图会覆盖原像素）
    cv.imwrite(os.path.join(out_dir, "raw_frame.png"), raw_img)
    validate_detection(raw_pieces, DETECT_PARAMS["required_ids"], raw_issues)

    if target_image_path:
        template_pieces, target_summary = _load_target_pieces(detector, mapper, raw_pieces,
                                                             target_image_path)
        _save_json(out_dir, "target_summary.json", target_summary)
        template_img = np.full_like(raw_img, 255)   # 预览画布：目标图模式没有相机模板帧
    else:
        template_vis = template_img.copy()
        template_pieces = detector.detect_template(template_img, annotated=template_vis)
        template_issues = list(detector.last_issues)
        cv.imwrite(os.path.join(out_dir, "template_annotated.png"), template_vis)
        validate_detection(template_pieces, DETECT_PARAMS["required_ids"], template_issues)

    quantize_template_angles(template_pieces)
    plans = plan_puzzle(raw_pieces, template_pieces, mapper,
                        TCP_RPY, GRIP_Z, PLACE_Z, PLACE_ANGLE_OFFSET, SUCTION_OFFSET_XY)

    # 尺度归一：模板检测出的块大小有误差（实测两个大三角分别偏大 3.5% 与 7.8%），
    # 会把目标布局整体撑大，现场表现为"整体放大、缝很大"。这里按实物 S 把布局缩回正确尺度。
    if ENABLE_SCALE_NORMALIZATION:
        plans, scale_report = normalize_plans(plans, template_pieces, mapper.homography, S_PHYS_MM)
        print("[tangram_puzzle] %s" % describe_scale(scale_report))
    else:
        print("[tangram_puzzle] 尺度归一未启用（ENABLE_SCALE_NORMALIZATION=False），位姿与既有验证版本一致")
    # 目标图模式没有相机模板帧，预览改画在原料实拍帧上，
    # 这样 plan_preview.png 直接能看到"图案会落在成品区的什么位置"。
    preview_canvas = raw_img if target_image_path else template_img
    preview = render_plan_preview(preview_canvas, raw_pieces, template_pieces, mapper, plans)
    _save_json(out_dir,"plans.json", {"plans":[asdict(p) for p in plans],
        "homography":mapper.homography.tolist(), "tool":TOOL,"user":USER,
        "suction_offset_xy":[float(v) for v in SUCTION_OFFSET_XY],
        "note":"Offline geometry only; model RMS is NOT measured robot error. Calibration remains unverified."})

    # raw_annotated / template_annotated 已在检测后立即写出，这里不再重复写同一文件。
    cv.imwrite(os.path.join(out_dir, "plan_preview.png"), preview)
    with open(os.path.join(out_dir, "plans.txt"), "w", encoding="utf-8") as f:
        for plan in plans:
            f.write(f"ID={plan.piece_id}({plan.color_name}) "
                    f"机器人平面原料角={plan.grasp_angle:.2f} 模板角={plan.place_angle:.2f} "
                    f"Δ={plan.delta_angle:.2f} 模型RMS={plan.fit_error_mm:.3f}mm 疑似翻面={plan.suspected_flip} "
                    f"旋转补偿=({plan.rotation_offset_mm[0]:.2f},{plan.rotation_offset_mm[1]:.2f})mm\n"
                    f"抓取位姿={[round(v, 3) for v in plan.grasp_pose]}\n"
                    f"放置位姿={[round(v, 3) for v in plan.place_pose]}\n\n")

    compensated = [p for p in plans if any(p.rotation_offset_mm)]
    if compensated:
        print(f"[planner] 已按吸盘偏心反补 {len(compensated)} 块的放置位姿（最大 "
              f"{max(abs(v) for p in compensated for v in p.rotation_offset_mm):.2f} mm）；"
              f"偏心参数来自 config.SUCTION_OFFSET_XY")

    print(f"[tangram_puzzle] 模板来源 {'目标图' if target_image_path else '实拍模板'}，"
          f"检出 {len(template_pieces)} 块；原料区检出 {len(raw_pieces)} 块，"
          f"生成 {len(plans)} 条抓放计划，结果已保存: {out_dir}")
    flagged=[p.piece_id for p in plans if p.suspected_flip or p.geometry_mismatch]
    if flagged:
        print(f"[planner] 几何疑点 IDs={flagged}，离线仅供检查，实机执行会被拦截")
    return plans


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="七巧板离线规划；实机需显式选择")
    parser.add_argument("template", nargs="?", help="模板原图（photo 模式离线用）")
    parser.add_argument("raw_image", nargs="?", help="原料原图（photo 模式离线用）")
    parser.add_argument("--raw", dest="raw_flag", help="原料帧路径（image 模式离线用）")
    parser.add_argument("--target-source", choices=("photo", "image"),
                        help="目标来源；缺省用 config.TARGET_SOURCE")
    parser.add_argument("--target-image", help="目标图路径；给了就按 image 模式跑")
    parser.add_argument("--execute", action="store_true", help="连接设备并执行；需现场完成标定及路径确认")
    parser.add_argument("--use-frame",
                        help="实机执行时改用这张已存在的帧（必须是在拍照位拍的），不现场抓帧")
    args = parser.parse_args()

    source = args.target_source or ("image" if args.target_image else TARGET_SOURCE)
    raw_path = args.raw_flag or args.raw_image

    if args.use_frame and not args.execute:
        parser.error("--use-frame 只用于实机执行；离线请用 --raw")
    if args.use_frame and raw_path:
        parser.error("--use-frame 与 --raw 不能同时给")
    if args.execute and (args.template or raw_path):
        parser.error("--execute 不与离线图片混用（要复用已有帧请用 --use-frame）")
    if source == "image":
        if args.template:
            parser.error("image 模式不接受模板位置参数；请用 --target-image 与 --raw")
        if not args.execute and not raw_path:
            parser.error("image 模式离线需要 --raw <原料帧>；实机请加 --execute")
    elif not args.execute and not (args.template and raw_path):
        parser.error("请提供模板原图和原料原图；默认不启动设备")

    run(args.template, raw_path, execute=args.execute,
        target_source=source, target_image_path=args.target_image,
        use_frame=args.use_frame)

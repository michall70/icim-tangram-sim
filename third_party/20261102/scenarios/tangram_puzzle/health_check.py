"""只读体检：机器人状态 + 相机成像 + 当前检测参数下的结果。

本脚本**不发送任何运动指令、不改变 DO、不写入控制器**：只读状态、抓一帧图、
用现有 DETECT_PARAMS 跑一次检测，把结果存成 JSON 报告，用于在实机上建立基线。

在实机上先跑它、而不是直接跑 --execute，原因：
- main.run --execute 会先回 HOME、到拍照位、松吸盘，已经是完整动作；
- 本脚本只有 RPC 读操作，机器人不会动，可以在任何安全状态下执行。

用法（项目根目录）:
    python -B -m scenarios.tangram_puzzle.health_check
    python -B -m scenarios.tangram_puzzle.health_check --no-robot
    python -B -m scenarios.tangram_puzzle.health_check --no-camera
    python -B -m scenarios.tangram_puzzle.health_check --image 某张原料图.png

说明：代码保持 Python 3.8 兼容（仓库里同时存在 cpython-38 与 cpython-314 的
字节码缓存，现场机器可能是 3.8），不使用 dict 合并运算符、match 语句等新语法。
"""

import argparse
import datetime
import json
import os
import sys

import cv2 as cv
import numpy as np

from common.calibration import CameraMapper
from scenarios.tangram_puzzle.config import (CAMERA_MATRIX, CAMERA_PARAMS, CAPTURE_POSE,
                                             DETECT_PARAMS, DIST_COEFFS, HOMOGRAPHY,
                                             ROBOT_IP, SDK_VERSION, TARGET_SOURCE, TOOL, USER)
from scenarios.tangram_puzzle.tangram_detector import TangramDetector

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# 拍照位一致性判据：超过则该次采集与标定/模板的前提已不成立（对应评审项 D04）。
CAPTURE_POS_TOL_MM = 0.3
CAPTURE_POS_TOL_DEG = 0.1

# 过曝判据：V>=250 视为可能饱和。现场托盘为浅色底纸，比黑色工位更容易饱和。
OVEREXPOSED_V = 250
LOW_SAT_S = 40
DARK_V = 30


def _six(value, label):
    """校验并转成 6 个 float；失败返回 None 而不是抛错（体检脚本不中断）。"""
    try:
        values = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if len(values) != 6 or not all(np.isfinite(values)):
        print("  [警告] %s 不是有效的六维数组: %r" % (label, value))
        return None
    return values


def _expand_do(do_hi, do_lo):
    """按 SDK 文档展开：hi 的 bit0~7 对应 co0~co7，lo 的 bit0~7 对应 do0~do7。"""
    state = {}
    for i in range(8):
        state["co%d" % i] = bool((int(do_hi) >> i) & 1)
    for i in range(8):
        state["do%d" % i] = bool((int(do_lo) >> i) & 1)
    return state


def check_robot():
    """只读读取机器人状态。任何失败都记录并返回 partial 结果，不抛异常。"""
    from common.robot_client import CobotClient

    report = {"ok": False, "steps": []}
    robot = CobotClient(ROBOT_IP, tool=TOOL, user=USER, sdk_version=SDK_VERSION)
    try:
        print("[1/3] 连接机器人 %s（只读，不动作）" % ROBOT_IP)
        robot.connect()
        report["controller_version"] = robot.controller_version
        report["sdk_version_requested"] = SDK_VERSION
        report["steps"].append("connect")

        ret = robot.rpc.GetActualTCPNum()
        report["active_tool"] = {"ret": list(ret) if isinstance(ret, (list, tuple)) else ret}
        print("  当前工具号: %r（代码要求 %d）" % (ret, TOOL))

        ret = robot.rpc.GetActualWObjNum()
        report["active_wobj"] = {"ret": list(ret) if isinstance(ret, (list, tuple)) else ret}
        print("  当前工件号: %r（代码要求 %d）" % (ret, USER))

        joints = _six(robot.rpc.GetActualJointPosDegree()[1], "当前关节角")
        report["joints_deg"] = joints
        if joints is not None:
            print("  当前关节角: [%s]" % ", ".join("%.3f" % v for v in joints))

        tcp = _six(robot.rpc.GetActualTCPPose()[1], "当前 TCP 位姿")
        report["tcp_pose"] = tcp
        if tcp is not None:
            print("  当前 TCP 位姿: [%s]" % ", ".join("%.3f" % v for v in tcp))

        flange = _six(robot.rpc.GetActualToolFlangePose()[1], "当前法兰位姿")
        report["flange_pose"] = flange
        if flange is not None:
            print("  当前法兰位姿: [%s]" % ", ".join("%.3f" % v for v in flange))

        # 拍照位一致性：只有机器人确实在拍照位时这一项才有意义，因此同时输出结论
        # 与"是否可能不在拍照位"的提示，不做成静默通过。
        ref = _six(CAPTURE_POSE, "配置 CAPTURE_POSE")
        if tcp is not None and ref is not None:
            delta = [tcp[i] - ref[i] for i in range(6)]
            report["tcp_minus_capture_pose"] = delta
            within = (all(abs(delta[i]) <= CAPTURE_POS_TOL_MM for i in range(3))
                      and all(abs(delta[i]) <= CAPTURE_POS_TOL_DEG for i in range(3, 6)))
            report["at_capture_pose"] = bool(within)
            print("  TCP - CAPTURE_POSE = [%s]" % ", ".join("%+.3f" % v for v in delta))
            if within:
                print("  → 与配置拍照位一致（%s mm / %s deg 内）"
                      % (CAPTURE_POS_TOL_MM, CAPTURE_POS_TOL_DEG))
            else:
                print("  → 未在配置拍照位；若这不是刻意为之，说明拍照位/标定前提已变，"
                      "此时的检测与映射结果不能直接采信")

        ret = robot.rpc.GetDO()
        report["do_raw"] = list(ret[1]) if isinstance(ret, (list, tuple)) and len(ret) > 1 else ret
        try:
            state = _expand_do(ret[1][0], ret[1][1])
            report["do_state"] = state
            report["suction_do0"] = state["do0"]
            print("  控制箱 DO 状态: CO=%s | DO=%s"
                  % ("".join("1" if state["co%d" % i] else "0" for i in range(8)),
                     "".join("1" if state["do%d" % i] else "0" for i in range(8))))
            print("  吸盘 DO0 当前电平 = %d（官方气动吸盘接控制箱 DO0；"
                  "有效电平可在 WebApp 的 DO0~DO7 有效状态里配置）"
                  % int(state["do0"]))
        except (IndexError, TypeError) as exc:
            print("  [警告] GetDO 返回格式异常，无法解析: %r (%s)" % (ret, exc))

        report["ok"] = True
    except Exception as exc:
        report["error"] = "%s: %s" % (type(exc).__name__, exc)
        print("  [失败] 机器人读取中断: %s" % report["error"])
        print("  提示: 本机 IP 为 %s；离线环境请加 --no-robot" % ROBOT_IP)
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass
    return report


def _roi_stats(hsv, roi, label):
    """统计 ROI 内亮度/饱和度分布，用于判断成像是否已进入饱和区。"""
    x, y, w, h = roi
    sub = hsv[y:y+h, x:x+w]
    sat = sub[:, :, 1]
    val = sub[:, :, 2]
    return {
        "label": label, "roi": [int(x), int(y), int(w), int(h)],
        "v_mean": float(val.mean()),
        "overexposed_pct": float((val >= OVEREXPOSED_V).mean() * 100.0),
        "dark_pct": float((val <= DARK_V).mean() * 100.0),
        "low_sat_pct": float((sat < LOW_SAT_S).mean() * 100.0),
    }


def _color_breakdown(img, label):
    """逐色统计掩模碎片数与面积，用于定位"掩模被高光切断"这类失败。

    这是对 tangram_detector._detect_color 的只读复算：阈值、形态学与面积门槛
    与现有实现一致，但额外输出中间量，便于判断失败发生在哪一步。
    """
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    min_area = DETECT_PARAMS["min_area"]
    morph_size = DETECT_PARAMS["morph_size"]
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_size, morph_size))
    rows = []
    for spec in DETECT_PARAMS["specs"]:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in spec["ranges"]:
            mask |= cv.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        raw_area = int(np.count_nonzero(mask))
        clean = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        clean = cv.morphologyEx(clean, cv.MORPH_CLOSE, kernel)
        contours, _ = cv.findContours(clean, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        areas = sorted((cv.contourArea(c) for c in contours), reverse=True)
        big = [a for a in areas if a >= min_area]
        rows.append({
            "id": spec["id"], "name": spec["name"], "shape": spec["shape"],
            "raw_mask_area_px": raw_area,
            "components_ge_min_area": len(big),
            "component_areas": [round(float(a), 1) for a in big[:6]],
            "split_suspect": bool(len(big) > 1),
        })
        flag = ""
        if len(big) > 1:
            flag = "  ← 掩模分裂（高光/阴影切断候选；现有实现会判 multiple candidates 并丢弃）"
        elif not big:
            flag = "  ← 无有效候选"
        print("     %-7s %-13s 面积%7d px  连通域(≥%d)=%d  最大=%s%s"
              % (spec["name"], spec["shape"], raw_area, min_area, len(big),
                 ("%.0f" % big[0]) if big else "-", flag))
    return rows


def _detect(img, mode, mapper, vis_save_base):
    """跑现有检测并保存标注图；返回结果与 issues。"""
    detector = TangramDetector(DETECT_PARAMS)
    annotated = img.copy()
    if mode == "raw":
        pieces = detector.detect_raw_material(img, annotated=annotated)
    else:
        pieces = detector.detect_template(img, annotated=annotated)
    ids = sorted(p.piece_id for p in pieces)
    cv.imwrite(vis_save_base, annotated)
    return {
        "mode": mode, "ids": ids, "count": len(ids),
        "issues": list(detector.last_issues),
        "pieces": [{"id": p.piece_id, "name": p.color_name, "shape": p.shape,
                    "grasp_point": [int(v) for v in p.grasp_point],
                    "angle": round(float(p.angle), 3)} for p in pieces],
        "annotated_path": vis_save_base,
    }, detector.last_issues


def check_camera(out_dir, image_path=None, already_undistorted=False):
    """抓图（或读入已有图）并统计成像与检测结果。"""
    mapper = CameraMapper(CAMERA_MATRIX, DIST_COEFFS, HOMOGRAPHY)
    report = {"ok": False}

    if image_path:
        print("[2/3] 读入图像（不访问相机）: %s" % image_path)
        img = cv.imread(image_path)
        if img is None:
            raise FileNotFoundError("无法读取图像: %s" % image_path)
        if not already_undistorted:
            img = mapper.undistort(img)
            report["input_already_undistorted"] = False
        else:
            report["input_already_undistorted"] = True
        report["source"] = image_path
        report["device_settings"] = None
    else:
        import pyrealsense2 as rs
        from common.realsense_camera import RealSenseCamera
        print("[2/3] 打开相机并采一帧（不移动机器人）")
        camera = RealSenseCamera(**CAMERA_PARAMS)
        try:
            camera.open()
            try:
                camera.print_current_settings()
                sensor = camera.profile.get_device().first_color_sensor()
                report["device_settings"] = {
                    "enable_auto_exposure": float(sensor.get_option(rs.option.enable_auto_exposure)),
                    "exposure": float(sensor.get_option(rs.option.exposure)),
                    "gain": float(sensor.get_option(rs.option.gain)),
                    "enable_auto_white_balance": float(
                        sensor.get_option(rs.option.enable_auto_white_balance)),
                    "white_balance": float(sensor.get_option(rs.option.white_balance)),
                }
            except Exception as exc:
                print("  [警告] 读取相机参数失败: %s" % exc)
            raw = camera.grab_color(warmup=30)
            cv.imwrite(os.path.join(out_dir, "capture_raw.png"), raw)
            img = mapper.undistort(raw)
            report["source"] = "camera"
        finally:
            try:
                camera.close()
            except Exception:
                pass

    cv.imwrite(os.path.join(out_dir, "undistorted.png"), img)
    h, w = img.shape[:2]
    report["image_size"] = [int(w), int(h)]
    if DETECT_PARAMS.get("image_size") and [w, h] != list(DETECT_PARAMS["image_size"]):
        print("  [警告] 图像尺寸 %dx%d 与配置 %s 不一致，检测会拒绝执行"
              % (w, h, DETECT_PARAMS["image_size"]))

    print("[3/3] 成像统计与检测")
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    stats = [_roi_stats(hsv, DETECT_PARAMS["raw_roi"], "raw_roi"),
             _roi_stats(hsv, DETECT_PARAMS["template_roi"], "template_roi")]
    report["exposure_stats"] = stats
    for row in stats:
        print("  %-11s 均值V=%5.1f  过曝(V>=%d)=%5.2f%%  暗(V<=%d)=%5.2f%%  低饱和(S<%d)=%5.2f%%"
              % (row["label"], row["v_mean"], OVEREXPOSED_V, row["overexposed_pct"],
                 DARK_V, row["dark_pct"], LOW_SAT_S, row["low_sat_pct"]))

    print("  原料区逐色掩模诊断:")
    report["raw_color_diagnostics"] = _color_breakdown(img, "raw")

    raw_res, raw_issues = _detect(img, "raw", mapper,
                                  os.path.join(out_dir, "raw_annotated.png"))
    report["raw_detection"] = raw_res
    report["target_source"] = TARGET_SOURCE

    print("  原料区检出 %d/7 块: %s" % (raw_res["count"], raw_res["ids"]))
    if raw_issues:
        print("  原料区 issues: %s" % raw_issues)

    required = set(DETECT_PARAMS["required_ids"])
    report["raw_complete"] = set(raw_res["ids"]) == required

    if TARGET_SOURCE == "image":
        # 目标图模式下不使用 TEMPLATE_PATH，也不需要在原料帧里检测成品区模板。
        # 仍统计 template_roi 的亮度（上面 stats 已做），但跳过模板检测，避免误报"模板 0/7"。
        report["template_detection"] = {"mode": "template", "skipped": True,
                                        "reason": "TARGET_SOURCE=image，目标位姿来自目标图文件",
                                        "ids": [], "count": 0, "issues": []}
        report["template_complete"] = None
        print("  模板区检测：已跳过（TARGET_SOURCE=image，目标位姿来自目标图文件）")
    else:
        template_res, template_issues = _detect(img, "template", mapper,
                                               os.path.join(out_dir, "template_annotated.png"))
        report["template_detection"] = template_res
        report["template_complete"] = set(template_res["ids"]) == required
        print("  模板区检出 %d/7 块: %s" % (template_res["count"], template_res["ids"]))
        if template_issues:
            print("  模板区 issues: %s" % template_issues)

    report["ok"] = True
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-robot", action="store_true", help="跳过机器人读取")
    parser.add_argument("--no-camera", action="store_true", help="跳过相机采集")
    parser.add_argument("--image", help="改为分析已有图像（跳过相机）")
    parser.add_argument("--already-undistorted", action="store_true",
                        help="配合 --image：输入图已去过畸变")
    parser.add_argument("--out", help="输出目录，默认 results/health_<时间戳>")
    parser.add_argument("--strict", action="store_true", help="任一项未通过时返回非零退出码")
    args = parser.parse_args(argv)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(RESULTS_DIR, "health_" + stamp)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("只读体检：不发送运动指令、不改变 DO、不写入控制器")
    print("输出目录: %s" % out_dir)
    print("=" * 72)

    report = {"timestamp": stamp, "read_only": True,
              "note": "Read-only diagnostic. No motion command, no DO change, no write to controller.",
              "config": {"robot_ip": ROBOT_IP, "tool": TOOL, "user": USER,
                         "sdk_version": SDK_VERSION,
                         "capture_pose": [float(v) for v in CAPTURE_POSE]}}

    if not args.no_robot:
        report["robot"] = check_robot()
    else:
        print("[1/3] 已跳过机器人读取")

    if not args.no_camera:
        try:
            report["camera"] = check_camera(out_dir, args.image, args.already_undistorted)
        except Exception as exc:
            report["camera"] = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
            print("  [失败] 相机与检测中断: %s" % report["camera"]["error"])
    else:
        print("[2/3][3/3] 已跳过相机与检测")

    path = os.path.join(out_dir, "health.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)

    print("-" * 72)
    print("报告: %s" % path)
    problems = []
    robot = report.get("robot") or {}
    if robot and not robot.get("ok"):
        problems.append("机器人读取失败")
    camera = report.get("camera") or {}
    if camera and camera.get("ok"):
        if not camera.get("raw_complete"):
            problems.append("原料区未检出完整七块")
        if camera.get("target_source") != "image" and not camera.get("template_complete"):
            problems.append("模板区未检出完整七块")
        for row in camera.get("exposure_stats") or []:
            # 目标图模式只关心原料区曝光；template_roi 只作为托盘锚点，不参与检测。
            if row["label"] == "template_roi" and camera.get("target_source") == "image":
                continue
            if row["overexposed_pct"] > 2.0:
                problems.append("%s 过曝 %.2f%%(>2%%)" % (row["label"], row["overexposed_pct"]))
        for row in camera.get("raw_color_diagnostics") or []:
            if row["split_suspect"]:
                problems.append("%s 掩模分裂" % row["name"])
    if problems:
        print("风险项（不一定是阻塞；请结合原料检出数判断）: %s" % "；".join(problems))
        print("  说明：检出 7/7 时，过曝/掩模分裂仍可能是风险提示；不要只看这一行。")
    else:
        print("未发现风险项（仅代表本次采集；不代表实机精度已验证）")

    if args.strict and problems:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

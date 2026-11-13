"""拼图前的两区取景框：把原料区、成品区各画成一个框，先确认框对，再跑拼图。

## 为什么要有这个工具

原料区/成品区的范围在代码里是两个写死的像素矩形（`config.DETECT_PARAMS`
的 `raw_roi` / `template_roi`）。换拍照位、换工位、换相机之后它们不会自己
跟着走，而检测器只会安静地把框外的东西当成不存在。过去几轮现场失效
（只检出 3 块、5 块）都属于"框没框住实物"，但终端里只能看到"检出 3 块"，
看不出是框偏了还是颜色判据坏了。

本工具把这件事变成一张图和一个结论：

- 把两个框画在画面上（绿色=原料区，橙色=成品区），带尺寸与毫米换算；
- 把画面里所有彩色实体（七色掩模并集）框出来，落在框外的标红；
- 给出"框是否在画面内、两框是否重叠、框外还有没有东西"的检查结论；
- 可选再跑一次现有检测，把检出块数与 `issues` 一并打印（框太紧时
  会出现 `touches ROI boundary`）。

## 用法（项目根目录）

    python -B -m scenarios.tangram_puzzle.tools.roi_frame --image 某帧.png
    python -B -m scenarios.tangram_puzzle.tools.roi_frame --image 某帧.png --already-undistorted
    python -B -m scenarios.tangram_puzzle.tools.roi_frame --camera          # 采一帧
    python -B -m scenarios.tangram_puzzle.tools.roi_frame --live            # 实时窗口，s 保存 / q 退出

只读：不连机器人、不发送运动指令、不改 config。默认输入按相机原始图处理
（内部去畸变），与 main.py 的口径一致；已经是去畸变图的用 `--already-undistorted`。
"""

import argparse
import datetime
import json
import os
import sys

import cv2 as cv
import numpy as np

from common.calibration import CameraMapper
from scenarios.tangram_puzzle.config import (CAMERA_MATRIX, CAMERA_PARAMS,
                                             DETECT_PARAMS, DIST_COEFFS, HOMOGRAPHY)
from scenarios.tangram_puzzle.tangram_detector import TangramDetector

SCENARIO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(SCENARIO_DIR, "results")
PROJECT_DIR = os.path.dirname(os.path.dirname(SCENARIO_DIR))

# 画面配色（BGR）。绿色/橙色在黑色工位与浅色托盘上都能看清。
RAW_COLOR = (0, 220, 0)
PLACE_COLOR = (0, 170, 255)
OUTSIDE_COLOR = (0, 0, 255)
TICK = 26               # 四角加粗刻度长度 px
FILL_ALPHA = 0.15       # 框内淡色填充，便于一眼看出范围
COVER_MIN_AREA = 3000   # 彩色实体最小面积 px²（约 500 mm²；最小三角块实测约 1 万 px²）


# ---------------------------------------------------------------- 纯几何

def _rect(roi):
    """(x, y, w, h) → (x0, y0, x1, y1)，右/下为开区间。"""
    x, y, w, h = [int(v) for v in roi]
    return x, y, x + w, y + h


def overlap_area(first, second):
    """两个 ROI 的重叠面积 px²。"""
    ax0, ay0, ax1, ay1 = _rect(first)
    bx0, by0, bx1, by1 = _rect(second)
    dx = min(ax1, bx1) - max(ax0, bx0)
    dy = min(ay1, by1) - max(ay0, by0)
    return max(0, dx) * max(0, dy)


def gap_between(raw_roi, template_roi):
    """两框之间的最小像素间隙；重叠时为 0。"""
    ax0, ay0, ax1, ay1 = _rect(raw_roi)
    bx0, by0, bx1, by1 = _rect(template_roi)
    dx = max(0, max(ax0, bx0) - min(ax1, bx1))
    dy = max(0, max(ay0, by0) - min(ay1, by1))
    return max(dx, dy)


def check_rois(image_size, params):
    """结构性检查：框必须在画面内、两框不能重叠。返回 (problems, notes)。"""
    problems = []
    notes = []
    width, height = int(image_size[0]), int(image_size[1])

    expected = params.get("image_size")
    if expected and [width, height] != [int(v) for v in expected]:
        problems.append("图像 %dx%d 与配置 image_size=%s 不一致，检测会直接拒绝执行"
                        % (width, height, list(expected)))

    for label, key in (("原料区", "raw_roi"), ("成品区", "template_roi")):
        roi = params.get(key)
        if not roi:
            problems.append("%s 未配置 %s" % (label, key))
            continue
        x, y, w, h = [int(v) for v in roi]
        if min(w, h) <= 0:
            problems.append("%s %s=(%d,%d,%d,%d) 宽高必须为正" % (label, key, x, y, w, h))
            continue
        if x < 0 or y < 0 or x + w > width or y + h > height:
            problems.append("%s %s=(%d,%d,%d,%d) 超出 %dx%d 画面"
                            % (label, key, x, y, w, h, width, height))

    raw_roi = params.get("raw_roi")
    template_roi = params.get("template_roi")
    if raw_roi and template_roi:
        area = overlap_area(raw_roi, template_roi)
        # 配置了成品区四角时，template_roi 不参与检测也不参与判定，
        # 两框重叠不再有实际危害 —— 降级为提示，避免吓人。
        quad_mode = _quad_configured()
        if area > 0:
            if quad_mode:
                notes.append("两框重叠 %d px²，但成品区用的是四角配置，template_roi 不参与，可忽略" % area)
            else:
                problems.append("两框重叠 %d px²：同一块实物会同时进入两个检测器，必须分开" % area)
        else:
            notes.append("两框不重叠，间隙 %d px" % gap_between(raw_roi, template_roi))
    return problems, notes


def _quad_configured():
    """config 里是否配置了成品区四角（TARGET_AREA_QUAD_MM）。"""
    try:
        from scenarios.tangram_puzzle.config import TARGET_AREA_QUAD_MM
    except Exception:
        return False
    return bool(TARGET_AREA_QUAD_MM) and len(TARGET_AREA_QUAD_MM) == 4


def roi_mm(roi, homography):
    """把像素框换算成机器人平面上的近似尺寸与中心坐标（mm）。

    单应是投影变换，这里取四条边长的平均值，仅用于和实物尺寸对照，
    不当作托盘实测尺寸。返回值第 4 项是该框在台面上相对机器人 X 轴的夹角（度）。
    """
    x, y, w, h = [int(v) for v in roi]
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
    mapped = cv.perspectiveTransform(corners.reshape(-1, 1, 2), homography).reshape(-1, 2)
    width_mm = 0.5 * (np.linalg.norm(mapped[1] - mapped[0]) + np.linalg.norm(mapped[2] - mapped[3]))
    height_mm = 0.5 * (np.linalg.norm(mapped[3] - mapped[0]) + np.linalg.norm(mapped[2] - mapped[1]))
    center = cv.perspectiveTransform(
        np.array([[[x + w / 2.0, y + h / 2.0]]], np.float32), homography).reshape(-1)
    edge = mapped[1] - mapped[0]
    yaw_deg = float(np.degrees(np.arctan2(edge[1], edge[0])))
    return float(width_mm), float(height_mm), (float(center[0]), float(center[1])), yaw_deg


# ---------------------------------------------------------------- 彩色实体

def color_components(img, params, min_area=COVER_MIN_AREA):
    """全画面彩色实体的连通域（七色掩模并集）。

    只用颜色、不做多边形拟合：这里要回答的是"东西在哪"，不是"是什么形状"。
    同一颜色只有一块实物，七色并集出现的连通域就是摆件本身；两块贴在一起
    会合并成一个连通域，对"是否落在框内"的判断没有影响。
    """
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for spec in params["specs"]:
        for lo, hi in spec["ranges"]:
            mask |= cv.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    found = []
    for contour in contours:
        area = float(cv.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv.boundingRect(contour)
        moments = cv.moments(contour)
        if moments["m00"] > 0:
            center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        else:
            center = (x + w / 2.0, y + h / 2.0)
        found.append({"area": round(area, 1), "bbox": [int(x), int(y), int(w), int(h)],
                      "center": [round(float(center[0]), 1), round(float(center[1]), 1)]})
    found.sort(key=lambda row: -row["area"])
    return found


def classify(component, params):
    """判断一个彩色实体属于哪个框：返回 (区域, 是否完整落在框内)。

    "完整落在框内"比"中心在框内"更严格：被框边切开的块正是检测器报
    `touches ROI boundary` 并整块丢弃的情况，必须单独暴露出来。
    """
    x, y, w, h = component["bbox"]
    center_x, center_y = x + w / 2.0, y + h / 2.0
    partial = None
    for name, key in (("raw", "raw_roi"), ("template", "template_roi")):
        roi = params.get(key)
        if not roi:
            continue
        x0, y0, x1, y1 = _rect(roi)
        if x >= x0 and y >= y0 and x + w <= x1 and y + h <= y1:
            return name, True
        if x0 <= center_x <= x1 and y0 <= center_y <= y1 and partial is None:
            partial = name
    return (partial or "outside"), False


def measure_coverage(img, params):
    """彩色实体 → 区域归属清单。"""
    rows = []
    for component in color_components(img, params):
        region, fully = classify(component, params)
        row = dict(component)
        row["region"] = region
        row["fully_inside"] = bool(fully)
        rows.append(row)
    return rows


# ---------------------------------------------------------------- 绘图

def _label(img, text, origin, color, scale=0.55, thickness=1):
    """带黑底的文字，避免落在浅色托盘上看不见。"""
    width, height = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    x, y = origin
    cv.rectangle(img, (x - 4, y - height - 6), (x + width + 4, y + 4), (0, 0, 0), -1)
    cv.putText(img, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv.LINE_AA)


def draw_region(img, roi, color, title, subtitle=None, thickness=3):
    """画一个取景框：淡色填充 + 粗边框 + 四角刻度 + 标题。"""
    x0, y0, x1, y1 = _rect(roi)
    overlay = img.copy()
    cv.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
    cv.addWeighted(overlay, FILL_ALPHA, img, 1.0 - FILL_ALPHA, 0.0, img)
    cv.rectangle(img, (x0, y0), (x1, y1), color, thickness, cv.LINE_AA)
    for px, py, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                           (x1, y1, -1, -1), (x0, y1, 1, -1)):
        cv.line(img, (px, py), (px + dx * TICK, py), color, thickness + 1, cv.LINE_AA)
        cv.line(img, (px, py), (px, py + dy * TICK), color, thickness + 1, cv.LINE_AA)

    # 标题放在框上方；框贴着画面顶边时改放到框内，避免被裁掉。
    baseline = y0 - 12
    if baseline < 64:
        baseline = y0 + 52
    _label(img, title, (x0 + 8, baseline - 26), color, 0.62, 2)
    if subtitle:
        _label(img, subtitle, (x0 + 8, baseline), color, 0.5, 1)


def draw_footer(img, text):
    height, width = img.shape[:2]
    cv.rectangle(img, (0, height - 30), (width, height), (0, 0, 0), -1)
    cv.putText(img, text, (10, height - 10), cv.FONT_HERSHEY_SIMPLEX, 0.55,
               (220, 220, 220), 1, cv.LINE_AA)


def annotate(img, params, coverage=None, homography=None):
    """把两个框、彩色实体归属与图例画到一张图上（返回新图，不改输入）。"""
    vis = img.copy()
    for key, color, title in (("raw_roi", RAW_COLOR, "RAW (source)"),
                              ("template_roi", PLACE_COLOR, "PLACE (target)")):
        roi = params.get(key)
        if not roi:
            continue
        subtitle = "x=%d y=%d w=%d h=%d" % tuple(int(v) for v in roi)
        if homography is not None:
            width_mm, height_mm, _, _yaw = roi_mm(roi, homography)
            subtitle += "  |  %.0fx%.0f mm" % (width_mm, height_mm)
        draw_region(vis, roi, color, title, subtitle)

    for row in coverage or []:
        region = row.get("region")
        color = {"raw": RAW_COLOR, "template": PLACE_COLOR}.get(region, OUTSIDE_COLOR)
        x, y, w, h = row["bbox"]
        cv.rectangle(vis, (x, y), (x + w, y + h), color,
                     2 if region == "outside" else 1, cv.LINE_AA)
        cv.circle(vis, (int(row["center"][0]), int(row["center"][1])), 4, color, -1, cv.LINE_AA)

    draw_footer(vis, "RAW=source  PLACE=target  thin box=colour blob (7-colour mask)  RED=outside both")
    return vis


# ---------------------------------------------------------------- 报告

def build_report(img, params, homography, *, detect=False):
    """汇总：框的尺寸/毫米换算、结构检查、彩色实体归属、可选检测结果。"""
    height, width = img.shape[:2]
    problems, notes = check_rois((width, height), params)

    regions = {}
    for key in ("raw_roi", "template_roi"):
        roi = params.get(key)
        if not roi:
            continue
        width_mm, height_mm, center, yaw_deg = roi_mm(roi, homography)
        entry = {"roi": [int(v) for v in roi],
                 "size_mm": [round(width_mm, 1), round(height_mm, 1)],
                 "center_robot_mm": [round(center[0], 2), round(center[1], 2)],
                 "yaw_vs_robot_x_deg": round(yaw_deg, 2)}
        # 若成品区配置了四角，顺手把"框中心 vs 四角中心"的偏差算出来：
        # 偏差小说明这个像素框确实套在成品区上；偏差大说明框与四角对不上。
        if key == "template_roi":
            try:
                from scenarios.tangram_puzzle.config import TARGET_AREA_QUAD_MM
            except Exception:
                TARGET_AREA_QUAD_MM = None
            if TARGET_AREA_QUAD_MM:
                quad = np.asarray(TARGET_AREA_QUAD_MM, float)
                if quad.shape == (4, 2):
                    anchor = quad.mean(axis=0)
                    dx = float(center[0]) - float(anchor[0])
                    dy = float(center[1]) - float(anchor[1])
                    entry["anchor_delta_mm"] = [round(dx, 2), round(dy, 2)]
                    entry["anchor_delta_len_mm"] = round((dx * dx + dy * dy) ** 0.5, 2)
        regions[key] = entry

    coverage = measure_coverage(img, params)
    outside = [row for row in coverage if row["region"] == "outside"]
    partial = [row for row in coverage if row["region"] != "outside"
               and not row["fully_inside"]]
    report = {
        "image_size": [int(width), int(height)],
        "regions": regions,
        "problems": problems,
        "notes": notes,
        "coverage": {
            "blobs": len(coverage),
            "raw": len([row for row in coverage if row["region"] == "raw"]),
            "template": len([row for row in coverage if row["region"] == "template"]),
            "partial": len(partial),
            "outside": len(outside),
            "details": coverage,
        },
    }

    if detect:
        detector = TangramDetector(params)
        raw_pieces = detector.detect_raw_material(img)
        raw_issues = list(detector.last_issues)
        template_pieces = detector.detect_template(img)
        template_issues = list(detector.last_issues)
        report["detection"] = {
            "raw": {"ids": sorted(p.piece_id for p in raw_pieces),
                    "count": len(raw_pieces), "issues": raw_issues},
            "template": {"ids": sorted(p.piece_id for p in template_pieces),
                         "count": len(template_pieces), "issues": template_issues},
        }
    return report


def print_report(report):
    """终端摘要：先给框的位置与毫米尺寸，再给检查结论与框外实体。"""
    width, height = report["image_size"]
    print("=" * 72)
    print("两区取景框自检（只读，不连机器人）")
    print("图像 %dx%d" % (width, height))
    for key, label in (("raw_roi", "原料区"), ("template_roi", "成品区")):
        row = report["regions"].get(key)
        if not row:
            print("  %s %s：未配置" % (label, key))
            continue
        x, y, w, h = row["roi"]
        print("  %s %-12s x=%4d y=%4d w=%4d h=%4d   约 %5.0f x %5.0f mm   中心→机器人(%+8.2f, %+8.2f) mm"
              % (label, key, x, y, w, h, row["size_mm"][0], row["size_mm"][1],
                 row["center_robot_mm"][0], row["center_robot_mm"][1]))
        if "anchor_delta_len_mm" in row:
            print("        └ 与四角中心 差 (%+.2f, %+.2f) mm，长度 %.2f mm %s"
                  % (row["anchor_delta_mm"][0], row["anchor_delta_mm"][1],
                     row["anchor_delta_len_mm"],
                     "（对得上）" if row["anchor_delta_len_mm"] <= 5.0 else "（⚠ 框与锚点不一致）"))
        if key == "template_roi":
            yaw = row.get("yaw_vs_robot_x_deg", 0.0)
            verdict = ("与机器人 X 轴平行（判定准确）" if abs(yaw) <= 2.0
                       else ("轻微倾斜，判定偏差可接受" if abs(yaw) <= 5.0
                             else "⚠ 明显倾斜：判定框会与托盘错位，建议把托盘摆正"))
            print("        └ 相对机器人 X 轴转 %.1f° → %s" % (yaw, verdict))
    for note in report["notes"]:
        print("  · %s" % note)

    coverage = report["coverage"]
    print("  彩色实体 %d 个：原料框内 %d，成品框内 %d，被框边切开 %d，框外 %d"
          % (coverage["blobs"], coverage["raw"], coverage["template"],
             coverage["partial"], coverage["outside"]))
    for row in coverage["details"]:
        if row["region"] != "outside":
            continue
        print("    [警告] 框外实体：面积 %7.0f px^2  中心(%6.0f,%6.0f)  包围盒 %s"
              % (row["area"], row["center"][0], row["center"][1], row["bbox"]))
    for row in coverage["details"]:
        if row["region"] == "outside" or row["fully_inside"]:
            continue
        print("    [警告] 压框边：%s 的实体被框边切开（bbox=%s，中心(%6.0f,%6.0f)）；"
              "检测会报 touches ROI boundary 并整块丢弃"
              % ("原料区" if row["region"] == "raw" else "成品区", row["bbox"],
                 row["center"][0], row["center"][1]))

    if "detection" in report:
        for key, label in (("raw", "原料区"), ("template", "成品区")):
            row = report["detection"][key]
            print("  %s检测：%d/7 块 %s" % (label, row["count"], row["ids"]))
            if row["issues"]:
                print("    issues: %s" % row["issues"])

    flags = list(report["problems"])
    if coverage["outside"]:
        flags.append("有 %d 个彩色实体落在两个框之外" % coverage["outside"])
    if coverage["partial"]:
        flags.append("有 %d 个实体被框边切开（框偏小或位置偏移）" % coverage["partial"])

    print("-" * 72)
    if flags:
        print("结论：[失败] 框与画面不匹配")
        for flag in flags:
            print("  - %s" % flag)
        print("  处理：按图上位置改 config.DETECT_PARAMS 的 raw_roi / template_roi，"
              "或把相机/工位恢复到拍这组图时的状态；不要直接放宽颜色阈值。")
    else:
        print("结论：[通过] 两个框都在画面内、不重叠、框外没有彩色实体")
        print("  注意：这只说明'框的位置对'，不代表颜色判据或实机精度已验证。")
    return bool(flags)


# ---------------------------------------------------------------- 输入输出

def _read_image(path, already_undistorted):
    img = cv.imread(path)
    if img is None:
        raise FileNotFoundError("无法读取图像: %s" % path)
    if already_undistorted:
        return img
    mapper = CameraMapper(CAMERA_MATRIX, DIST_COEFFS)
    return mapper.undistort(img)


def load_homography():
    """与 main.py 同源：直接使用 config.HOMOGRAPHY（不再读取 calib.json）。"""
    return HOMOGRAPHY, "config.HOMOGRAPHY"


def _out_dir(path=None):
    if path:
        return path
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(RESULTS_DIR, "roi_frame_" + stamp)


def _writable_dir(preferred):
    """优先用 results/；该目录不可写时退到项目根的 roi_frames/，并在终端说明。"""
    fallback = os.path.join(PROJECT_DIR, "roi_frames", os.path.basename(preferred))
    for candidate in (preferred, fallback):
        try:
            os.makedirs(candidate, exist_ok=True)
            probe = os.path.join(candidate, ".probe")
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write("")
            os.remove(probe)
            if candidate != preferred:
                print("  [提示] %s 不可写，改存到 %s" % (preferred, candidate))
            return candidate
        except OSError:
            continue
    raise PermissionError("没有可写的输出目录: %s" % preferred)


def _save(img, report, out_dir):
    out_dir = _writable_dir(out_dir)
    image_path = os.path.join(out_dir, "roi_frame.png")
    cv.imwrite(image_path, img)
    json_path = os.path.join(out_dir, "roi_frame.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print("  已保存: %s" % image_path)
    print("  已保存: %s" % json_path)
    return image_path


def _live(params, homography):
    """实时取景：窗口里叠加两个框，s 保存当前帧与结论，q/ESC 退出。"""
    from common.realsense_camera import RealSenseCamera

    mapper = CameraMapper(CAMERA_MATRIX, DIST_COEFFS, homography)
    camera = RealSenseCamera(**CAMERA_PARAMS)
    camera.open()
    window = "roi_frame"
    cv.namedWindow(window, cv.WINDOW_NORMAL)
    report = None
    frame_index = 0
    try:
        while True:
            img = mapper.undistort(camera.grab_color(warmup=0))
            frame_index += 1
            # 归属计算不必每帧都做；每 10 帧刷新一次，画面不卡。
            if report is None or frame_index % 10 == 1:
                report = build_report(img, params, homography)
            vis = annotate(img, params, report["coverage"]["details"], homography)
            flags = bool(report["problems"] or report["coverage"]["outside"]
                         or report["coverage"]["partial"])
            draw_footer(vis, "s=save  q=quit   |   RAW=source  PLACE=target   |   %s  raw=%d place=%d outside=%d"
                        % ("CHECK" if flags else "OK", report["coverage"]["raw"],
                           report["coverage"]["template"], report["coverage"]["outside"]))
            cv.imshow(window, vis)
            key = cv.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                out_dir = _out_dir()
                saved = annotate(img, params, report["coverage"]["details"], homography)
                print("[roi_frame] 保存到 %s" % out_dir)
                _save(saved, report, out_dir)
                print_report(report)
    finally:
        camera.close()
        cv.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="把原料区/成品区画成框并检查框与画面是否匹配（只读）")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="去畸变后的整帧图像（与拍模板/原料图同一拍照位）")
    group.add_argument("--camera", action="store_true", help="只采一帧，不移动机器人")
    group.add_argument("--live", action="store_true", help="实时窗口，s 保存 / q 退出")
    parser.add_argument("--already-undistorted", action="store_true",
                        help="配合 --image：输入图已去过畸变")
    parser.add_argument("--check", action="store_true",
                        help="再跑一次现有检测，打印检出块数与 issues")
    parser.add_argument("--out", help="输出目录，默认 results/roi_frame_<时间戳>")
    args = parser.parse_args(argv)

    params = DETECT_PARAMS
    homography, source = load_homography()
    print("[roi_frame] 单应来源: %s" % source)
    mapper = CameraMapper(CAMERA_MATRIX, DIST_COEFFS, homography)

    if args.live:
        _live(params, homography)
        return 0

    if args.image:
        img = _read_image(args.image, args.already_undistorted)
        print("[roi_frame] 读入图像: %s" % args.image)
    else:
        from common.realsense_camera import RealSenseCamera
        camera = RealSenseCamera(**CAMERA_PARAMS)
        print("[roi_frame] 采一帧（不移动机器人）")
        try:
            camera.open()
            img = mapper.undistort(camera.grab_color(warmup=30))
        finally:
            camera.close()

    report = build_report(img, params, homography, detect=args.check)
    failed = print_report(report)
    vis = annotate(img, params, report["coverage"]["details"], homography)
    _save(vis, report, _out_dir(args.out))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

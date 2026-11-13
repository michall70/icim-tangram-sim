"""手动拖框定 ROI：鼠标在图上拖出原料区/成品区，坐标实时显示，按 s 写回 config。

    python -B -m scenarios.tangram_puzzle.tools.roi_pick --image <实拍帧>

## 操作

| 键 | 作用 |
|---|---|
| **鼠标左键拖拽** | 在当前选中的区上画框 |
| `1` | 切到**原料区**（绿框） |
| `2` | 切到**成品区**（橙框） |
| `r` | 清掉当前区的框，重画 |
| `s` | 写回 `config.py`（自动备份 `config.py.bak_<时间戳>`），然后退出 |
| `p` | 只打印结果不动文件 |
| `q` / `ESC` | 不保存退出 |

右下角小字会显示：当前选中区、框的 `x y w h`、以及毫米尺寸。

## 成品区画多大

默认 `--template-mode usable`：**直接拖出"可用区"**（建议 180mm 那个范围，别贴着托盘边）。
如果更想拖**托盘内沿**、让工具自己换算成 180mm，用 `--template-mode tray --usable 180 --tray-mm 190`。

## 读图从哪来

    python -B -m scenarios.tangram_puzzle.capture_check        # 按 s 保存一帧

`capture_check` 存的是**原始帧**，本工具默认会做去畸变；若给的是
`results/<时间戳>/raw_frame.png`（已去畸变），加 `--already-undistorted`。
"""

import argparse
import datetime
import os
import re
import shutil

import cv2 as cv

from common.calibration import CameraMapper
from scenarios.tangram_puzzle.config import (CAMERA_MATRIX, DETECT_PARAMS, DIST_COEFFS,
                                             HOMOGRAPHY, TARGET_AREA_MM)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py")
RAW_COLOR = (60, 200, 60)
TPL_COLOR = (0, 165, 255)


def usable_from_tray(tray_box, tray_mm, usable_mm):
    """托盘内沿框 → 可用区框（同心、按 mm 比例缩放）。"""
    x, y, w, h = tray_box
    scale = float(usable_mm) / float(tray_mm)
    side_w = int(round(w * scale))
    side_h = int(round(h * scale))
    return (int(round(x + (w - side_w) / 2.0)), int(round(y + (h - side_h) / 2.0)),
            side_w, side_h)


def write_config(raw_box, tpl_box):
    """把两个框写回 config.py。返回备份文件名；没有任何变化时返回 None。

    注意：不能拿"文本有没有变化"当"有没有找到那两行"——如果画出来的框和现有配置一模一样，
    文本当然不变。这里先确认两行都存在，再判断是否真的需要改。
    """
    raw_pattern = re.compile(r"raw_roi=\([^)]*\)")
    tpl_pattern = re.compile(r"template_roi=\([^)]*\)")
    text = open(CONFIG_PATH, encoding="utf-8").read()
    if not raw_pattern.search(text) or not tpl_pattern.search(text):
        raise SystemExit("没在 config.py 里找到 raw_roi=... 或 template_roi=... 行，未写入")

    updated = raw_pattern.sub("raw_roi=%s" % (tuple(int(v) for v in raw_box),), text, count=1)
    updated = tpl_pattern.sub("template_roi=%s" % (tuple(int(v) for v in tpl_box),),
                              updated, count=1)
    if updated == text:
        return None                      # 与当前配置相同，不用写

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = CONFIG_PATH + ".bak_" + stamp
    shutil.copyfile(CONFIG_PATH, backup)
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        handle.write(updated)
    return backup


class Picker:
    def __init__(self, image, raw_box=None, tpl_box=None):
        self.image = image
        self.boxes = {"raw": raw_box, "template": tpl_box}
        self.active = "raw"
        self.start = None
        self.dragging = False
        self._preview = None
        self._last_status = None
        self.window = "roi_pick  (1=raw 2=template drag=draw s=save p=print q=quit)"

    def _clamp(self, x, y):
        height, width = self.image.shape[:2]
        return max(0, min(width - 1, x)), max(0, min(height - 1, y))

    def on_mouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.start = self._clamp(x, y)
            self.dragging = True
        elif event == cv.EVENT_MOUSEMOVE and self.dragging:
            self._preview = self._box_from(self.start, self._clamp(x, y))
        elif event == cv.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            box = self._box_from(self.start, self._clamp(x, y))
            if box[2] > 50 and box[3] > 50:
                self.boxes[self.active] = box

    @staticmethod
    def _box_from(a, b):
        x0, y0 = a
        x1, y1 = b
        return (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))

    def render(self):
        canvas = self.image.copy()
        for key, color in (("raw", RAW_COLOR), ("template", TPL_COLOR)):
            box = self.boxes.get(key)
            if not box:
                continue
            x, y, w, h = box
            thick = 3 if key == self.active else 1
            cv.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), color, thick)
            cv.putText(canvas, "%s %d,%d %dx%d" % (key, x, y, w, h), (x + 6, max(18, y - 8)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if self.dragging and getattr(self, "_preview", None):
            x, y, w, h = self._preview
            cv.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), (255, 255, 255), 1)
        return canvas

    def status(self, extra=""):
        parts = []
        for key in ("raw", "template"):
            box = self.boxes.get(key)
            parts.append("%s=%s" % (key, box if box else "-"))
        return "active=%s | %s %s" % (self.active, "  ".join(parts), extra)

    def run(self):
        cv.namedWindow(self.window, cv.WINDOW_NORMAL)
        cv.setMouseCallback(self.window, self.on_mouse)
        while True:
            cv.imshow(self.window, self.render())
            status = self.status()
            if status != self._last_status:      # 只在变化时打印，避免刷屏
                print("\r" + status, end="", flush=True)
                self._last_status = status
            key = cv.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                return None
            if key == ord("1"):
                self.active = "raw"
            elif key == ord("2"):
                self.active = "template"
            elif key == ord("r"):
                self.boxes[self.active] = None
            elif key in (ord("s"), ord("p")):
                cv.destroyAllWindows()
                return "save" if key == ord("s") else "print"


def main(argv=None):
    parser = argparse.ArgumentParser(description="手动拖框定 ROI（可写回 config）")
    parser.add_argument("--image", required=True)
    parser.add_argument("--already-undistorted", action="store_true")
    parser.add_argument("--template-mode", choices=("usable", "tray"), default="usable",
                        help="成品区拖的是可用区还是托盘内沿（tray 会自动换算）")
    parser.add_argument("--usable", type=float, default=float(TARGET_AREA_MM[0]))
    parser.add_argument("--tray-mm", type=float, default=190.0)
    args = parser.parse_args(argv)

    image = cv.imread(args.image, cv.IMREAD_COLOR)
    if image is None:
        raise SystemExit("无法读取图片: %s" % args.image)
    if not args.already_undistorted:
        image = CameraMapper(CAMERA_MATRIX, DIST_COEFFS, HOMOGRAPHY).undistort(image)

    picker = Picker(image, DETECT_PARAMS["raw_roi"], DETECT_PARAMS["template_roi"])
    action = picker.run()
    if action is None:
        print("\n未保存。")
        return 1

    raw_box, tpl_box = picker.boxes["raw"], picker.boxes["template"]
    if not raw_box or not tpl_box:
        print("\n两个区都要画（1=原料区，2=成品区）。未保存。")
        return 1
    if args.template_mode == "tray":
        tpl_box = usable_from_tray(tpl_box, args.tray_mm, args.usable)

    print("\n原料区 raw_roi      = %s" % (tuple(int(v) for v in raw_box),))
    print("成品区 template_roi = %s" % (tuple(int(v) for v in tpl_box),))
    if action == "save":
        backup = write_config(raw_box, tpl_box)
        if backup is None:
            print("这两个框与当前 config 完全相同，无需写入。")
        else:
            print("已写回 config.py（备份 %s）" % os.path.basename(backup))
    else:
        print("（只打印，未写文件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

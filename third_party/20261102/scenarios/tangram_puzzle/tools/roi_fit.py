"""按你从 roi_frame 图上读到的四条边，算出两个 ROI 并（可选）写回 config.py。

## 为什么是"你读数、我算数"

板/托盘是黑的、桌面也偏暗，全自动分割会被外景带偏（试过 Otsu、暗区连通域、
"框内最暗"搜索、暗区质心——都会滑到最暗的一侧）。而四条边在图上肉眼一看就有，
读数只要 10 秒。所以本工具把容易出错的那部分**算术**自动化：

    python -B -m scenarios.tangram_puzzle.tools.roi_fit ^
        --raw 150 100 744 699 ^
        --tray 758 154 1266 659

它会打印（下面的数字是 2026-10-31 实际写回 config 的一组）：

    raw_roi      = (148, 98, 598, 603)           # 板范围向外扩 2px
    template_roi = (772, 166, 481, 481)          # 180mm 可用区，中心=托盘中心

## 参数含义

- `--raw L T R B`：**黑板**的左右上下边（像素，去畸变后的 1280×720 坐标系）
- `--tray L T R B`：**托盘内沿**的四条边（不是可用区）
- `--tray-mm`：托盘内沿的物理边长（默认 190）
- `--usable`：想要的可用区边长（默认取 `config.TARGET_AREA_MM`，现在 180）
- `--apply`：写回 config.py（先备份成 `config.py.bak_<时间戳>`）

## 读数从哪来

    python -B -m scenarios.tangram_puzzle.tools.roi_frame --image <帧> --out <目录>

打开 `roi_frame.png`，鼠标悬停会显示坐标（或按网格目测）。
"""

import argparse
import datetime
import os
import re
import shutil

from scenarios.tangram_puzzle.config import DETECT_PARAMS, TARGET_AREA_MM

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py")
TRAY_PHYS_MM = 190.0
# 原料框比板边**向外**扩 2px：贴板边摆的块（如橙块）只要被切 1—2px 就会被判
# "touches ROI boundary" 整块丢弃（2026-10-31 实测：向内缩 2px → 7/7 变 6/7）。
RAW_MARGIN = 2


def compute(raw_edges, tray_edges, tray_mm, usable_mm):
    l, t, r, b = raw_edges
    if r - l < 200 or b - t < 200:
        raise SystemExit("原料区四边太小：%s" % (raw_edges,))
    raw_roi = (int(l - RAW_MARGIN), int(t - RAW_MARGIN),
               int(r - l + 2 * RAW_MARGIN), int(b - t + 2 * RAW_MARGIN))

    tl, tt, tr, tb = tray_edges
    if tr - tl < 150 or tb - tt < 150:
        raise SystemExit("托盘四边太小：%s" % (tray_edges,))
    px_per_mm = (tr - tl) / float(tray_mm)
    side = int(round(usable_mm * px_per_mm))
    cx = (tl + tr) / 2.0
    cy = (tt + tb) / 2.0
    template_roi = (int(round(cx - side / 2.0)), int(round(cy - side / 2.0)), side, side)

    report = {"raw_roi": raw_roi, "template_roi": template_roi,
              "px_per_mm": px_per_mm, "usable_mm": usable_mm,
              "tray_center": (cx, cy), "problems": []}
    if px_per_mm < 2.0 or px_per_mm > 4.0:
        report["problems"].append("px/mm=%.2f 异常（应 2.0—4.0），检查托盘四边" % px_per_mm)
    if not (280 <= side <= 620):
        report["problems"].append("可用区边长 %dpx 异常（应 280—620）" % side)
    overlap_x = min(raw_roi[0] + raw_roi[2], template_roi[0] + template_roi[2]) - max(raw_roi[0], template_roi[0])
    overlap_y = min(raw_roi[1] + raw_roi[3], template_roi[1] + template_roi[3]) - max(raw_roi[1], template_roi[1])
    if overlap_x > 0 and overlap_y > 0:
        report["problems"].append("两框重叠 %d×%d px" % (overlap_x, overlap_y))
    return report


def write_config(raw_roi, template_roi):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = CONFIG_PATH + ".bak_" + stamp
    shutil.copyfile(CONFIG_PATH, backup)
    text = open(CONFIG_PATH, encoding="utf-8").read()
    before = text
    text = re.sub(r"raw_roi=\([^)]*\)", "raw_roi=(%d, %d, %d, %d)" % raw_roi, text, count=1)
    text = re.sub(r"template_roi=\([^)]*\)",
                  "template_roi=(%d, %d, %d, %d)" % template_roi, text, count=1)
    if text == before:
        raise SystemExit("没有找到 raw_roi / template_roi 行，未写入")
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        handle.write(text)
    return backup


def main(argv=None):
    parser = argparse.ArgumentParser(description="按读数算两个 ROI，可写回 config")
    parser.add_argument("--raw", nargs=4, type=float, metavar=("L", "T", "R", "B"),
                        help="黑板四边（像素）")
    parser.add_argument("--tray", nargs=4, type=float, metavar=("L", "T", "R", "B"),
                        help="托盘内沿四边（像素）")
    parser.add_argument("--tray-mm", type=float, default=TRAY_PHYS_MM, help="托盘内沿物理边长 mm")
    parser.add_argument("--usable", type=float, default=float(TARGET_AREA_MM[0]),
                        help="可用区边长 mm，默认取 config.TARGET_AREA_MM")
    parser.add_argument("--apply", action="store_true", help="自检通过后写回 config.py（先备份）")
    args = parser.parse_args(argv)

    if not args.raw or not args.tray:
        parser.error("需要同时给 --raw L T R B 和 --tray L T R B（读数见 roi_frame.png）")

    report = compute(args.raw, args.tray, args.tray_mm, args.usable)
    print("托盘：%.0f×%.0f px，中心 (%.0f, %.0f)，%.3f px/mm"
          % (args.tray[2] - args.tray[0], args.tray[3] - args.tray[1],
             report["tray_center"][0], report["tray_center"][1], report["px_per_mm"]))
    print("原料区：旧 %s → 新 %s" % (DETECT_PARAMS["raw_roi"], report["raw_roi"]))
    print("成品区：旧 %s → 新 %s（%.0fmm 可用区）"
          % (DETECT_PARAMS["template_roi"], report["template_roi"], report["usable_mm"]))
    if report["problems"]:
        print("自检未通过：")
        for item in report["problems"]:
            print("  · %s" % item)
        if args.apply:
            print("未写回 config。")
        return 1
    print("自检通过")
    if args.apply:
        backup = write_config(report["raw_roi"], report["template_roi"])
        print("已写回 %s（备份 %s）" % (CONFIG_PATH, os.path.basename(backup)))
    else:
        print("未写回（加 --apply 才会写）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""九点标定验证：输入图像像素坐标 → 机器人 TCP x/y，人工核对末端位置。

用法（项目根目录执行）:
    python -m scenarios.tangram_puzzle.verify_calibration             # 终端输入像素坐标
    python -m scenarios.tangram_puzzle.verify_calibration <图像路径>   # 弹窗鼠标点选像素

口径说明：HOMOGRAPHY 标定于去畸变图像像素（手动 9 点脚本输出的坐标口径），
与 main.py 在线流程（undistort → 检测 → 映射）一致。因此验证像素也须取自去
畸变图像：点选模式传入相机原始采图，脚本内部自动去畸变后显示；终端输入模式
的像素坐标须来自去畸变图。

验证流程：
1. 选定一个特征明显的点，得到去畸变图像上的像素 (u, v)；
2. 本脚本换算出对应机器人 TCP x/y（mm）；
3. 手动示教将机器人末端移到该 x/y（姿态与标定点采集时一致，Z 也应同高，
   否则透视偏差会叠加进来）；
4. 目视检查末端是否正对该像素位置，偏差应与标定重投影误差同量级（≤2mm）。
"""

import sys

import cv2 as cv

from common.calibration import CameraMapper
from scenarios.tangram_puzzle.config import CAMERA_MATRIX, DIST_COEFFS, HOMOGRAPHY


def load_homography():
    """直接使用 config.HOMOGRAPHY（本仓库不包含一键九点/calib.json）。"""
    print("使用 config.HOMOGRAPHY（2026-09-10 手动 9 点标定）")
    return HOMOGRAPHY


def report(mapper, u, v):
    x, y = mapper.pixel_to_robot((u, v))
    print(f"  像素(u={u:.0f}, v={v:.0f})  →  机器人 TCP x={x:.3f}, y={y:.3f} mm"
          f"（示教移动到此位置后核对）")


def verify_by_input(mapper):
    print("\n连续输入像素坐标 u v（空格或逗号分隔），输入 q 退出")
    while True:
        text = input("像素坐标: ").strip().lower().replace(",", " ")
        if text in ("q", "quit", "exit"):
            return
        parts = text.split()
        try:
            if len(parts) != 2:
                raise ValueError
            u, v = float(parts[0]), float(parts[1])
        except ValueError:
            print("  输入格式有误，请输入两个数字，如: 640 360")
            continue
        report(mapper, u, v)


def verify_by_click(mapper, image_path):
    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    # 与标定口径一致：在去畸变图像上点选（若传入已是去畸变图，二次校正影响可忽略）
    img = mapper.undistort(img)
    vis = img.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            report(mapper, x, y)
            cv.circle(vis, (x, y), 4, (0, 0, 255), -1)
            cv.imshow("verify_calibration", vis)

    cv.namedWindow("verify_calibration", cv.WINDOW_NORMAL)
    cv.setMouseCallback("verify_calibration", on_mouse)
    cv.imshow("verify_calibration", vis)
    print("\n在窗口中点击要验证的像素点（可连续多点），按 q 或 ESC 退出")
    while cv.waitKey(100) not in (ord("q"), 27):
        pass
    cv.destroyAllWindows()


if __name__ == "__main__":
    homography = load_homography()
    if homography is None:
        raise SystemExit("HOMOGRAPHY 未标定，请先运行 python -m common.calibration 手动做 9 点标定")
    # 带内参构造：点选模式内部去畸变，保证像素口径与 HOMOGRAPHY 标定一致
    mapper = CameraMapper(CAMERA_MATRIX, DIST_COEFFS, homography)
    print("九点标定验证（去畸变图像像素 → TCP x/y）")
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg:
        verify_by_click(mapper, arg)
    else:
        verify_by_input(mapper)

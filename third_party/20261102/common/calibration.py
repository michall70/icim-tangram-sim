"""相机标定与坐标映射。

- CameraMapper：内参去畸变 + 单应性像素→机器人平面坐标（各场景传入自己的标定参数）
- 手动 9 点标定辅助脚本：python -m common.calibration [已有图像路径]
  流程：采图（或读入已有图）→ 窗口中鼠标依序点击 9 个标定点 →
  终端逐点输入对应的机器人 x y（从示教器读取）→ 求解单应矩阵并打印（可直接粘贴进 config）

标定操作建议：将标记物（如夹爪尖、笔尖）依次放在工作台 9 个位置
（3×3 网格，覆盖原料区与成品区），每个位置：机器人示教走位读取 x/y + 图像上点选像素。

本模块**不包含**一键九点自动标定（自动走网格、自动检靶标、生成 calib.json）。
现场如需重标，使用上面的手动交互脚本，把结果写回 `config.HOMOGRAPHY`。
"""

import cv2 as cv
import numpy as np


class CameraMapper:
    """相机内参去畸变 + 像素到机器人平面坐标的单应映射。

    camera_matrix/dist_coeffs/homography 任一为 None 表示该项未标定：
    未标定内参时 undistort 原样返回；未标定单应时 pixel_to_robot 返回像素坐标。
    """

    def __init__(self, camera_matrix=None, dist_coeffs=None, homography=None):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.homography = homography

    def undistort(self, img):
        if self.camera_matrix is None or self.dist_coeffs is None:
            return img
        return cv.undistort(img, self.camera_matrix, self.dist_coeffs)

    def pixel_to_robot(self, point):
        """像素 (u, v) → 机器人平面坐标 (x, y)，单位 mm。"""
        if self.homography is None:
            print("[calibration] 单应矩阵未标定，按像素坐标返回")
            return float(point[0]), float(point[1])
        pixel = np.array([[float(point[0]), float(point[1])]], dtype=np.float32).reshape(-1, 1, 2)
        robot_xy = cv.perspectiveTransform(pixel, self.homography).flatten()
        return float(robot_xy[0]), float(robot_xy[1])


def solve_homography(pixel_points, robot_points):
    """由标定点对（≥4 组）求解像素→机器人单应矩阵，并返回 (H, 重投影平均误差)。"""
    src = np.array(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.array(robot_points, dtype=np.float32).reshape(-1, 1, 2)
    h, _ = cv.findHomography(src, dst)
    if h is None:
        raise ValueError("单应矩阵求解失败，请检查标定点是否共线或坐标录入有误")
    proj = cv.perspectiveTransform(src, h).reshape(-1, 2)
    error = float(np.mean(np.linalg.norm(proj - dst.reshape(-1, 2), axis=1)))
    return h, error


def run_calibration(image_path=None, num_points=9):
    """交互式 9 点标定：鼠标点像素 + 输入机器人坐标 → 打印单应矩阵。"""
    if image_path is None:
        from common.realsense_camera import RealSenseCamera
        with RealSenseCamera(fps=15, depth=False) as camera:
            img = camera.grab_color()
    else:
        img = cv.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

    vis = img.copy()
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv.circle(vis, (x, y), 4, (0, 0, 255), -1)
            cv.putText(vis, str(len(points)), (x + 6, y - 6),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv.imshow("calibration", vis)

    print(f"请依序点击 {num_points} 个标定点的像素位置（顺序与之后录入的机器人坐标一致），可按 q 提前结束")
    cv.namedWindow("calibration", cv.WINDOW_NORMAL)
    cv.setMouseCallback("calibration", on_mouse)
    cv.imshow("calibration", vis)
    while len(points) < num_points:
        if cv.waitKey(100) in (ord("q"), 27):
            break
    cv.destroyAllWindows()
    if len(points) < 4:
        raise RuntimeError(f"标定点不足（至少 4 个，当前 {len(points)} 个）")

    robot_points = []
    for i, (px, py) in enumerate(points, 1):
        while True:
            text = input(f"第 {i}/{len(points)} 个点 像素({px},{py}) 对应机器人坐标 x y: ").split()
            if len(text) == 2:
                try:
                    robot_points.append((float(text[0]), float(text[1])))
                    break
                except ValueError:
                    pass
            print("输入格式有误，请输入两个数字，如: 208.749 -336.73")

    h, error = solve_homography(points, robot_points)
    print(f"\n重投影平均误差: {error:.3f} mm（>2mm 时建议重做标定）")
    print("\nHOMOGRAPHY = np.array(")
    for row in h:
        print(f"    [{row[0]:.9e}, {row[1]:.9e}, {row[2]:.9e}],")
    print("])")
    return h, error


if __name__ == "__main__":
    import sys
    run_calibration(sys.argv[1] if len(sys.argv) > 1 else None)

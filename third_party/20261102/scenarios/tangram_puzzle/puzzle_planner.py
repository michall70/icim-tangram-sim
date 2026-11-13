"""按颜色ID规划；位置与方向均在机器人XY平面计算。

当前仅支持同一固定拍照位的实拍模板，保留现有单应矩阵和高度。
112=给定位姿无法到达，不能仅靠角度归一化消除。
吸盘偏心在旋转后会让块中心偏离规划点，按 delta 反补（见 rotation_offset）。
"""
from dataclasses import dataclass
import math
import cv2 as cv
import numpy as np
from scenarios.tangram_puzzle.tangram_detector import TangramDetector


@dataclass
class PickPlacePlan:
    piece_id: int
    color_name: str
    grasp_pose: list
    place_pose: list
    grasp_angle: float       # 机器人XY平面方向，非图像角
    place_angle: float
    delta_angle: float       # 实际计划绕机器人Z轴旋转（含显式偏移）
    fit_error_mm: float = 0.0  # 预测顶点与模板顶点RMS；不是实机定位误差
    suspected_flip: bool = False
    mirror_fit_error_mm: float = 0.0
    geometry_mismatch: bool = False
    rotation_offset_mm: tuple = (0.0, 0.0)  # 已反补的旋转偏心位移，不是实测误差


def normalize_angle(angle, period=360.0):
    if not math.isfinite(angle) or not math.isfinite(period) or period <= 0:
        raise ValueError("角度必须有限，周期必须为正")
    result = (angle + period/2) % period - period/2
    return result + period if result <= -period/2 else result


def _homography(mapper):
    h = np.asarray(mapper.homography, dtype=float)
    if h.shape != (3,3) or not np.isfinite(h).all() or np.linalg.matrix_rank(h) != 3:
        raise ValueError("缺少有效单应矩阵，不能把像素当毫米规划")
    return h


def _transform(points, h):
    points = np.asarray(points, dtype=float).reshape(-1,2)
    if not np.isfinite(points).all():
        raise ValueError("坐标含非有限数值")
    homogeneous = np.column_stack((points,np.ones(len(points)))) @ h.T
    if np.any(np.abs(homogeneous[:,2]) < 1e-9):
        raise ValueError("单应映射分母接近零")
    result = homogeneous[:,:2] / homogeneous[:,2,None]
    if not np.isfinite(result).all():
        raise ValueError("坐标映射结果无效")
    return result


def _rotation(degrees):
    a = math.radians(degrees)
    return np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])


# 偏心超过该值几乎一定是录错（七巧板边缘距法兰轴不会这么远），用于拦截误填。
MAX_SUCTION_OFFSET_MM = 200.0


def _suction_offset(value):
    """校验吸盘偏心（mm）；(0,0) 表示不做补偿。"""
    if value is None:
        return np.zeros(2)
    offset = np.asarray(value, dtype=float).reshape(-1)
    if offset.size != 2 or not np.isfinite(offset).all():
        raise ValueError("吸盘偏心必须是两个有限的坐标值（mm）")
    if np.linalg.norm(offset) > MAX_SUCTION_OFFSET_MM:
        raise ValueError(f"吸盘偏心 {np.linalg.norm(offset):.1f} mm 超过 {MAX_SUCTION_OFFSET_MM:.0f} mm，请核对是否录错")
    return offset


def rotation_offset(delta_angle, suction_offset):
    """绕 Z 旋转 delta 后块中心相对规划点的位移 = (R(delta)-I)·offset（mm）。

    标定是在固定抓取姿态下做的，吸盘偏心已被吸收，因此抓取阶段（delta=0）
    不需要补偿；放置阶段姿态改变后该偏心才会变成平面上的位置误差。
    """
    offset = _suction_offset(suction_offset)
    if not offset.any():
        return np.zeros(2)
    return (_rotation(delta_angle) - np.eye(2)) @ offset


def _piece_geometry(piece, h):
    vertices = _transform(piece.vertices, h)
    want = 3 if piece.shape == "triangle" else 4
    if piece.shape not in ("triangle","square","parallelogram") or len(vertices) != want:
        raise ValueError(f"ID={piece.piece_id} 形状/顶点无效")
    if abs(cv.contourArea(vertices.astype(np.float32))) < 1e-6:
        raise ValueError("退化多边形")
    angle = TangramDetector._polygon_angle(vertices)
    center = _transform([piece.grasp_point], h)[0]
    return vertices, center, angle


def _rigid_fit(source, target):
    """遍历有向顶点循环对应，仅求旋转，不允许镜像或缩放。"""
    candidates=[]
    for k in range(len(target)):
        aligned=np.roll(target,k,axis=0)
        u,_,vt=np.linalg.svd(source.T@aligned)
        correction=np.eye(2)
        correction[-1,-1]=np.linalg.det(u@vt)
        row_rotation=u@correction@vt
        angle=math.degrees(math.atan2(row_rotation[0,1],row_rotation[0,0]))
        error=float(np.sqrt(np.mean(np.sum((source@row_rotation-aligned)**2,axis=1))))
        candidates.append((error,normalize_angle(angle)))
    return min(candidates,key=lambda item:item[0])


def validate_plan_geometry(plans):
    bad=[p.piece_id for p in plans if p.suspected_flip or p.geometry_mismatch]
    if bad:
        raise ValueError(f"规划几何未通过 IDs={bad}：疑似翻面或尺寸/轮廓/标定不一致，请检查预览；禁止自动翻面或强行执行")


def plan_puzzle(raw_pieces, template_pieces, mapper,
                tcp_rpy, grip_z, place_z, place_angle_offset,
                suction_offset=(0.0, 0.0)):
    """按颜色匹配；缺块/重复ID拒绝，禁止静默跳过。

    suction_offset: 吸盘中心相对法兰轴的偏心（mm，抓取姿态下），用于反补
    放置旋转带来的位置偏差；默认 (0,0) 与不补偿的旧行为完全一致。
    """
    h = _homography(mapper)
    if len(tcp_rpy) != 3 or not np.isfinite([*tcp_rpy,grip_z,place_z,place_angle_offset]).all():
        raise ValueError("抓放高度或姿态无效")
    offset = _suction_offset(suction_offset)
    raw_map = {p.piece_id:p for p in raw_pieces}
    template_ids = [p.piece_id for p in template_pieces]
    if len(raw_map)!=len(raw_pieces) or len(set(template_ids))!=len(template_ids):
        raise ValueError("检测包含重复ID")
    if not template_ids or set(template_ids)-set(raw_map):
        raise ValueError("模板为空或原料缺块，停止规划")
    rx,ry,rz = tcp_rpy
    plans=[]
    for tpl in template_pieces:
        raw=raw_map[tpl.piece_id]
        if raw.shape!=tpl.shape:
            raise ValueError(f"ID={tpl.piece_id} 原料与模板形状不一致")
        rv,g,ga = _piece_geometry(raw,h)
        tv,p,pa = _piece_geometry(tpl,h)
        period={"triangle":360.0,"parallelogram":180.0,"square":90.0}[tpl.shape]
        source=rv-g
        target=tv-p
        _,best_delta=_rigid_fit(source,target)
        delta=normalize_angle(normalize_angle(best_delta,period)+place_angle_offset)
        predicted=source @ _rotation(delta).T+p
        fit=min(float(np.sqrt(np.mean(np.sum((predicted-np.roll(tv,k,axis=0))**2,axis=1)))) for k in range(len(tv)))
        mirrored=source.copy()
        mirrored[:,0]*=-1
        mirror_fit,_=_rigid_fit(mirrored[::-1],target)
        radius=float(np.sqrt(np.mean(np.sum(target**2,axis=1))))
        # 相对几何门槛用于拦截明显不相容，非比赛毫米精度验收标准。
        suspected_flip=tpl.shape=="parallelogram" and fit>radius*0.10 and mirror_fit<fit*0.5
        mismatch=fit>radius*0.15
        # 抓取（delta=0）无需补偿；放置按实际旋转量反补吸盘偏心。
        correction=rotation_offset(delta,offset)
        place_xy=p-correction
        plans.append(PickPlacePlan(tpl.piece_id,tpl.color_name,
            [*map(float,g),float(grip_z),rx,ry,rz],
            [*map(float,place_xy),float(place_z),rx,ry,normalize_angle(rz+delta)],
            float(ga),float(pa),float(delta),fit,suspected_flip,mirror_fit,mismatch,
            tuple(float(v) for v in correction)))
    return plans


def predicted_piece_xy(plan):
    """块中心的预期落点 = 放置位姿 XY + 已反补的偏心位移（旋转补偿后两者相差该项）。"""
    return np.asarray(plan.place_pose[:2],float) + np.asarray(plan.rotation_offset_mm,float)


def render_plan_preview(template_img, raw_pieces, template_pieces, mapper, plans):
    """用实际规划角在机器人平面刚性搬运，再反投影；不缩放以掩盖误差。"""
    h=_homography(mapper)
    inverse=np.linalg.inv(h)
    overlay=template_img.copy()
    raw_map={p.piece_id:p for p in raw_pieces}
    template_map={p.piece_id:p for p in template_pieces}
    for i,plan in enumerate(plans):
        raw=raw_map[plan.piece_id]
        points=_transform(raw.vertices,h)
        predicted=(points-np.asarray(plan.grasp_pose[:2])) @ _rotation(plan.delta_angle).T+predicted_piece_xy(plan)
        image_points=_transform(predicted,inverse)
        cv.polylines(overlay,[np.rint(image_points).astype(np.int32)],True,(0,255,255),2)
        cv.polylines(overlay,[np.array(template_map[plan.piece_id].vertices,np.int32)],True,(0,190,0),1)
        cv.putText(overlay,f"ID {plan.piece_id}: {plan.delta_angle:.1f} deg; RMS {plan.fit_error_mm:.1f} mm" + (" | CHECK FLIP" if plan.suspected_flip else ""),
            (15,65+i*26),cv.FONT_HERSHEY_SIMPLEX,0.55,(0,0,255),1,cv.LINE_AA)
    cv.putText(overlay,"Yellow: predicted | Green: target | OFFLINE - calibration unverified",
        (15,30),cv.FONT_HERSHEY_SIMPLEX,0.55,(0,0,255),1,cv.LINE_AA)
    return overlay

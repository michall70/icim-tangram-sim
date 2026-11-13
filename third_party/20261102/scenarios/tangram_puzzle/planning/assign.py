"""解 ↔ 实物：把求解出的七个"槽位"指给现场实际的七块。

## 为什么不是"随便配"

七块里只有两类可以互换：两个大三角、两个小三角。中三角、正方形、平行四边形各自唯一。
所以指派只需要在同类内做双射；真正会出错的是这三件事：

1. **类别数量对不上**：求解器要 2L/1M/2T/1Q/1P，若现场只检到 1 个大三角，
   说明识别漏了，必须报错而不是硬凑；
2. **平行四边形手性**：实物平行四边形翻面后与另一面互为大镜像，靠旋转永远得不到。
   若检测到的平行四边形与件库的手性相反，物理上无法拼出该图案，必须拒绝；
3. **抓取可行性**：同类两块里，应优先选不被遮挡、抓取点离其他块更远的那块。

## 类别怎么认（不依赖颜色）

按形状 + 相对面积：
- 三角形 5 块，面积比 4:2:1（大:中:小）→ 排序取前二为 L、中间为 M、后二为 T；
- 四边形 2 块：四角都接近 90° → 正方形 Q；出现 45°/135° 角 → 平行四边形 P。
  边长比只作辅助下限，避免细长四边形被误判成正方形。

颜色仍可作为交叉校验，但**不作为身份依据**——这正是强光下橙色/咖啡串色时能救回来的原因。
"""

import math

import numpy as np

from scenarios.tangram_puzzle.tangram import geometry as geo

# 类别 → 期望件数
EXPECTED_COUNT = {"L": 2, "M": 1, "T": 2, "Q": 1, "P": 1}

# 面积比容差：中三角与正方形/平行四边形面积相同，靠形状区分；这里只用于三角内部分档
AREA_TOLERANCE = 0.15

# 正方形判定：
#   1) 内角：正方形 ≈90/90/90/90，平行四边形 ≈45/135/45/135 —— 差异极大，抗掩模变形；
#   2) 边长比只作辅助下限，避免把细长四边形误判成正方形。
SQUARE_ANGLE_TOLERANCE = 12.0   # 内角与 90° 的允许偏差（度）
SQUARE_EDGE_RATIO_MIN = 0.60    # 边长比的辅助下限（比旧阈值 0.85 宽松，主判据交给角度）


class AssignError(RuntimeError):
    """指派失败：类别不齐、手性不符或数量对不上。"""


def _kind_of(piece):
    """按形状给出"三角形/四边形"的粗分类，具体档位在 `classify_pieces` 里用面积定。"""
    if piece.shape in ("square", "parallelogram"):
        return "quad"
    return "tri"


def quad_kind(vertices):
    """四边形是正方形还是平行四边形：**按内角判**，边长比只作辅助。

    为什么不再只看边长比：掩模被高光/圆角咬掉一点时，方块的"最短/最长边"比会掉到
    0.85 以下，于是被判成平行四边形 —— 结果是 Q 缺一块、P 多一块，`assign_ids`
    直接报"类别数量不符"（2026-10-31 复现）。
    内角对这类变形稳定得多：正方形四角都在 90° 附近，平行四边形的角是 45°/135°。
    目标图解析（`planning.target_image.classify`）复用同一判据，避免两处口径漂移。
    """
    points = np.asarray(vertices, dtype=float).reshape(-1, 2)
    edges = geo.edge_lengths(points)
    if len(edges) != 4:
        return None
    angles = geo.interior_angles(points)
    if len(angles) != 4:
        return None
    near_right = sum(1 for angle in angles if abs(angle - 90.0) <= SQUARE_ANGLE_TOLERANCE)
    ratio = float(edges.min() / max(edges.max(), 1e-9))
    return "Q" if (near_right >= 3 and ratio >= SQUARE_EDGE_RATIO_MIN) else "P"


def classify_pieces(pieces):
    """把检出的块分成 L/M/T/Q/P，返回 [(piece, kind)]。

    分类只依赖相对关系（谁比谁大），因此不需要预先知道 S_phys。
    """
    triangles = [p for p in pieces if _kind_of(p) == "tri"]
    quads = [p for p in pieces if _kind_of(p) == "quad"]

    triangles = sorted(triangles, key=lambda p: -abs(geo.area(np.asarray(p.vertices, float))))
    labeled = []
    for rank, piece in enumerate(triangles):
        kind = "L" if rank < 2 else ("M" if rank == 2 else "T")
        labeled.append((piece, kind))
    for piece in quads:
        kind = quad_kind(piece.vertices)
        if kind is None:
            raise AssignError("ID=%s 的四边形顶点不足四个，无法判定是正方形还是平行四边形"
                              % getattr(piece, "piece_id", "?"))
        labeled.append((piece, kind))
    return labeled


def _fit_error(source, target):
    """两个顶点数相同的多边形，在**只允许旋转 + 平移**下的最小 RMS 残差。"""
    a = np.asarray(source, dtype=float)
    b = np.asarray(target, dtype=float)
    best = None
    for k in range(len(b)):
        aligned = np.roll(b, k, axis=0)
        u, _, vt = np.linalg.svd(a.T @ aligned)
        correction = np.eye(2)
        correction[-1, -1] = np.linalg.det(u @ vt)
        rotation = u @ correction @ vt
        error = float(np.sqrt(np.mean(np.sum((a @ rotation - aligned) ** 2, axis=1))))
        best = error if best is None else min(best, error)
    return best


def handedness_mismatch(vertices, kind="P", tolerance=0.35):
    """实物的手性是否与件库相反。

    做法：把实测顶点分别与"件库形状"和"件库形状的镜像"做旋转拟合，比残差。
    镜像拟合明显更好 ⇒ 实物是翻面状态，靠旋转拼不出该图案。
    这比"看某个叉积符号"稳，因为平行四边形的转角正负本来就是交替的。
    """
    library = {"P": np.array([[0.0, 0.0], [0.5, 0.0], [0.75, 0.25], [0.25, 0.25]])}[kind]
    measured = np.asarray(vertices, dtype=float)
    if len(measured) != len(library):
        return False, None, None
    # 必须先按面积把件库缩放到实测尺寸：件库是 S=1 的归一化形状，
    # 而实测顶点是毫米坐标，不缩放两边残差都会是几十毫米，比较毫无意义。
    measured_area = geo.area(measured)
    library_area = geo.area(library)
    if measured_area <= 0 or library_area <= 0:
        return False, None, None
    library = library * math.sqrt(measured_area / library_area)
    direct = _fit_error(library - library.mean(axis=0), measured - measured.mean(axis=0))
    mirrored = library.copy()
    mirrored[:, 0] *= -1
    flipped = _fit_error(mirrored - mirrored.mean(axis=0), measured - measured.mean(axis=0))
    return bool(flipped < direct * (1.0 - tolerance) + 1e-12), direct, flipped


def assign_ids(solution_pieces, detected_pieces, check_handedness=True):
    """给解出的槽位分配 piece_id，返回 (slot 序号 → piece_id, 报告 dict)。

    solution_pieces : `pattern.solution_to_pieces` 的输出（顺序即槽位顺序）
    detected_pieces : `TangramDetector.detect_raw_material` 的输出

    指派规则：同类内按"抓取点离其他块最远"优先（越不容易被邻块压住），
    其余同类随便配。类别数量不符时抛 AssignError。
    """
    labeled = classify_pieces(detected_pieces)
    by_kind = {}
    for piece, kind in labeled:
        by_kind.setdefault(kind, []).append(piece)

    counts = {kind: len(items) for kind, items in by_kind.items()}
    if counts != EXPECTED_COUNT:
        raise AssignError("检出类别数量不符：实际 %s，应为 %s（识别漏块或多块）"
                          % (counts, EXPECTED_COUNT))

    # 槽位分类：pattern.KIND_SHAPE 已把 kind 写进 color_name
    slot_kinds = []
    for piece in solution_pieces:
        kind = str(piece.color_name)[0]
        if kind not in EXPECTED_COUNT:
            raise AssignError("槽位类别无法识别：%r" % (piece.color_name,))
        slot_kinds.append(kind)

    mapping = {}
    report = {"counts": counts, "handedness": {}}
    for kind in EXPECTED_COUNT:
        slots = [i for i, k in enumerate(slot_kinds) if k == kind]
        candidates = list(by_kind[kind])
        if len(slots) != len(candidates):
            raise AssignError("类别 %s 的槽位 %d 个、实物 %d 块，无法一对一"
                              % (kind, len(slots), len(candidates)))
        # 同类内按"抓取点离其他块的距离"从大到小排，优先把难抓的块用在先放的槽位
        def isolation(piece):
            point = np.asarray(piece.grasp_point, dtype=float)
            others = [np.asarray(other.grasp_point, dtype=float)
                      for other in detected_pieces if other is not piece]
            return min((float(np.linalg.norm(point - other)) for other in others),
                       default=float("inf"))
        candidates.sort(key=isolation, reverse=True)
        for slot, piece in zip(slots, candidates):
            mapping[slot] = piece.piece_id

        if kind == "P" and check_handedness:
            mismatch, direct, flipped = handedness_mismatch(candidates[0].vertices)
            report["handedness"]["P"] = {"mismatch": mismatch,
                                         "direct_rms": None if direct is None else round(direct, 3),
                                         "mirror_rms": None if flipped is None else round(flipped, 3)}
            if mismatch:
                raise AssignError(
                    "平行四边形的实测手性与件库相反（旋转拟合 %.2f vs 镜像拟合 %.2f）："
                    "实物可能是翻面摆放，靠旋转无法拼出该图案"
                    % (direct, flipped))

    report["mapping"] = dict(mapping)
    return mapping, report


def describe(report):
    hand = report.get("handedness", {}).get("P")
    extra = ""
    if hand:
        extra = "；平行四边形旋转拟合 %.2f / 镜像拟合 %.2f（%s）" % (
            hand["direct_rms"], hand["mirror_rms"],
            "手性不符" if hand["mismatch"] else "手性正常")
    return "类别 %s；指派 %s%s" % (report.get("counts"), report.get("mapping"), extra)

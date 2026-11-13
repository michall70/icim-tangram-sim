"""七巧板平铺求解器（格点枚举版）：把"连续搜索"换成"离散枚举 + 精确覆盖"。

## 为什么要换算法

旧版本（`solver.py`）在每个节点只把**一块的一个顶点**对到"剩余区域最极值点"上，
然后递归。这个假设对单位正方形成立，但对公告 Fig 2 的三角形、以及现场给的 V 形都不成立：
实测锚点会在同一位置被反复选中（放下的块与轮廓的交点仍算"未覆盖"），所有分支死在那里，
即使 900 秒 / 200 万节点也穷尽无解。

本版本不再依赖那条假设：

1. **先把位姿枚举全**。七巧板所有顶点都落在两套格点上（边长步长 S/4 的方格，以及它旋转
   45° 的方格——因为 `(S/4)/√2 · √2 = S/4`，两套格点覆盖全部 8 个朝向）。于是"某块在
   某朝向下能放哪儿"是一个**有限集合**，可以一次性列出来。
2. **再用"必须覆盖最极值未覆盖点"作为剪枝**：任何完整拼法里，剩余区域最外面的那个点
   一定会被某块盖住；枚举所有**包含**该点的候选，而不是"顶点落在这里"的候选。
   这一条是完备的剪枝，不会误杀解法。
3. 七块面积和恒为 1（S=1 归一化），所以"两两不重叠 + 都在轮廓内"就等价于"恰好铺满"。

## 与旧版的关系

- `solver.py` 保留（单位正方形仍可解，作为回归基线）；
- 本模块是进阶主用版本；
- 共用 `geometry.py`、`verify.py`。
"""

import math
import time

import numpy as np

from scenarios.tangram_puzzle.tangram import geometry as geo

# 格点步长（S=1 单位）：S/4
LATTICE_STEP = 0.25

# 搜索方向：取一个不规整方向，避免多个点投影并列
SEARCH_DIRECTION = np.array([1.0, 0.37]) / math.hypot(1.0, 0.37)

ROTATIONS = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)

# 判定"点是否落在块内"时的边界容差
INSIDE_EPS = 1e-9


class Placement:
    """一个候选位姿：某块在某个朝向下、平移后的多边形。"""

    __slots__ = ("kind", "index", "angle", "polygon", "bbox")

    def __init__(self, kind, index, angle, polygon):
        self.kind = kind
        self.index = index
        self.angle = angle
        self.polygon = polygon
        self.bbox = geo.bounding_box(polygon)

    def contains(self, point):
        x0, y0, x1, y1 = self.bbox
        if point[0] < x0 - 1e-9 or point[0] > x1 + 1e-9 or \
                point[1] < y0 - 1e-9 or point[1] > y1 + 1e-9:
            return False
        return geo.point_in_polygon(point, self.polygon, eps=1e-7)


class SolveResult:
    def __init__(self, placements, nodes, seconds, candidates, target=None, basis=None,
                 snap_error=None):
        self.placements = placements
        self.nodes = nodes
        self.seconds = seconds
        self.candidates = candidates
        # 求解实际使用的轮廓（已吸附到格点）。自检必须拿它当基准：
        # 从图片提取的原轮廓与它相差千分之二量级，用原轮廓比对会把正确解判成"越界"。
        self.target = target
        self.basis = basis
        self.snap_error = snap_error

    def polygons(self):
        return [item["polygon"] for item in self.placements]


# ---------------------------------------------------------------- 件库形状

def piece_shapes():
    """七块在 S=1 下的本地多边形（质心在原点、逆时针）。与 solver.py 保持一致。"""
    large_leg = 1.0 / math.sqrt(2.0)
    small_leg = 1.0 / (2.0 * math.sqrt(2.0))

    def triangle(leg):
        return np.array([[0.0, 0.0], [leg, 0.0], [0.0, leg]])

    shapes = []
    for index in (1, 2):
        shapes.append({"kind": "L", "index": index, "poly": triangle(large_leg)})
    shapes.append({"kind": "M", "index": 1, "poly": triangle(0.5)})
    for index in (1, 2):
        shapes.append({"kind": "T", "index": index, "poly": triangle(small_leg)})
    shapes.append({"kind": "Q", "index": 1, "poly": np.array(
        [[0.0, 0.0], [small_leg, 0.0], [small_leg, small_leg], [0.0, small_leg]])})
    shapes.append({"kind": "P", "index": 1, "poly": np.array(
        [[0.0, 0.0], [0.5, 0.0], [0.75, 0.25], [0.25, 0.25]])})
    for shape in shapes:
        centered = geo.ensure_ccw(shape["poly"])
        shape["poly"] = centered - geo.centroid(centered)
    return shapes


# ---------------------------------------------------------------- 格点

def lattice_points(x0, y0, x1, y1, step=LATTICE_STEP, origin=None):
    """两套格点：正交方格，以及它旋转 45° 的方格。

    七巧板的边长只取 S、S/√2、S/2、S/(2√2) 四种；把步长取 S/4 时，
    正交朝向的顶点落在方格上，45° 朝向的顶点落在旋转方格上，两者合起来覆盖 8 个朝向。
    """
    # 格点必须从**图形自己的一个顶点**起铺：归一化会把图形平移到质心，
    # 顶点的绝对坐标会变成 -1/3 之类的非格点值，从全局 0 起铺就全错位了。
    origin = np.zeros(2) if origin is None else np.asarray(origin, dtype=float)
    x0, y0, x1, y1 = x0 - origin[0], y0 - origin[1], x1 - origin[0], y1 - origin[1]
    points = []
    i0 = int(math.floor(x0 / step)) - 1
    i1 = int(math.ceil(x1 / step)) + 1
    j0 = int(math.floor(y0 / step)) - 1
    j1 = int(math.ceil(y1 / step)) + 1
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            points.append((origin[0] + i * step, origin[1] + j * step))

    # 旋转 45°：基向量 (step/√2, step/√2) 与 (-step/√2, step/√2)
    unit = step / math.sqrt(2.0)
    for i in range(i0 - 1, i1 + 2):
        for j in range(j0 - 1, j1 + 2):
            points.append((origin[0] + unit * (i - j), origin[1] + unit * (i + j)))
    return points


def enumerate_placements(silhouette, shapes, step=LATTICE_STEP, log=None):
    """枚举所有"落在轮廓内"的候选位姿。"""
    x0, y0, x1, y1 = geo.bounding_box(silhouette)
    points = lattice_points(x0, y0, x1, y1, step, origin=geo.as_array(silhouette)[0])
    placements = []
    for shape in shapes:
        for angle in ROTATIONS:
            rotated = geo.rotate(shape["poly"], angle)
            reference = rotated[0]
            for point in points:
                candidate = rotated + (np.array(point, dtype=float) - reference)
                cx0, cy0, cx1, cy1 = geo.bounding_box(candidate)
                if cx0 < x0 - 1e-6 or cy0 < y0 - 1e-6 or \
                        cx1 > x1 + 1e-6 or cy1 > y1 + 1e-6:
                    continue
                if not geo.polygon_inside(candidate, silhouette):
                    continue
                placements.append(Placement(shape["kind"], shape["index"], angle, candidate))
    if log:
        log("枚举到 %d 个候选位姿" % len(placements))
    return placements



def snap_silhouette(polygon, step=LATTICE_STEP, tolerance=0.06):
    """把轮廓顶点吸附到格点上，返回 (吸附后的多边形, 用哪套格点, 最大偏差)。

    为什么必须做：从图片提取出来的轮廓总带千分之几的噪声（本例顶点是 0.501、0.247
    这类值），而候选位姿是按"轮廓首顶点 + 步长整数倍"精确生成的。差这么一点点，
    块边就会落在轮廓外一丁点，`polygon_inside` 判越界，合法位姿被整批丢掉——
    这正是 V 形明明可拼却枚举不到解的直接原因。

    七巧板的顶点只可能落在两套格点之一：正交方格，或它旋转 45° 的方格。
    这里两套都试，取吸附误差小的那套；两套误差都过大说明这不是七巧板图形。
    """
    polygon = np.asarray(polygon, dtype=float)
    origin = polygon[0]
    bases = {
        "正交": (np.array([step, 0.0]), np.array([0.0, step])),
        "45度": (np.array([step / math.sqrt(2.0)] * 2),
                 np.array([-step / math.sqrt(2.0), step / math.sqrt(2.0)])),
    }
    best = None
    for name, (axis_a, axis_b) in bases.items():
        matrix = np.column_stack((axis_a, axis_b))
        if abs(np.linalg.det(matrix)) < 1e-12:
            continue
        inverse = np.linalg.inv(matrix)
        snapped = []
        error = 0.0
        for point in polygon:
            coefficients = inverse @ (point - origin)
            rounded = np.rint(coefficients)
            moved = origin + matrix @ rounded
            error = max(error, float(np.linalg.norm(moved - point)))
            snapped.append(moved)
        if best is None or error < best[2]:
            best = (geo.ensure_ccw(np.array(snapped, dtype=float)), name, error)
    if best is None:
        return np.asarray(polygon, dtype=float), None, float("inf")
    return best

# ---------------------------------------------------------------- 覆盖判定

class _Coverage:
    """用一个粗栅格判断"还剩下哪些区域没被覆盖"，只用于选点，不参与精度。"""

    def __init__(self, silhouette, resolution=120):
        self._polygon = np.asarray(silhouette, dtype=float)
        x0, y0, x1, y1 = geo.bounding_box(silhouette)
        self.origin = np.array([x0, y0], dtype=float)
        span = max(x1 - x0, y1 - y0)
        self.step = span / float(resolution) if span > 0 else 1.0
        self.shape = (int(math.ceil((y1 - y0) / self.step)) + 1,
                      int(math.ceil((x1 - x0) / self.step)) + 1)
        self.inside = self._fill(silhouette)

    def _fill(self, polygon):
        grid = np.zeros(self.shape, dtype=bool)
        points = np.asarray(polygon, dtype=float)
        count = len(points)
        for row in range(self.shape[0]):
            y = self.origin[1] + (row + 0.5) * self.step
            crossings = []
            for i in range(count):
                x1, y1 = points[i]
                x2, y2 = points[(i + 1) % count]
                if (y1 > y) == (y2 > y):
                    continue
                crossings.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
            if len(crossings) < 2:
                continue
            crossings.sort()
            for k in range(0, len(crossings) - 1, 2):
                left = int(math.ceil((crossings[k] - self.origin[0]) / self.step - 0.5))
                right = int(math.floor((crossings[k + 1] - self.origin[0]) / self.step - 0.5))
                if right >= left:
                    grid[row, max(0, left):min(self.shape[1], right + 1)] = True
        return grid

    def paint(self, polygon):
        """把多边形盖到栅格上（凸多边形扫描线）。"""
        grid = np.zeros(self.shape, dtype=bool)
        points = np.asarray(polygon, dtype=float)
        count = len(points)
        rows = np.arange(self.shape[0])
        y = self.origin[1] + (rows + 0.5) * self.step
        low = np.full(self.shape[0], np.inf)
        high = np.full(self.shape[0], -np.inf)
        for i in range(count):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % count]
            if abs(y2 - y1) < 1e-15:
                continue
            t = (y - y1) / (y2 - y1)
            valid = (t >= 0.0) & (t <= 1.0)
            if not valid.any():
                continue
            xe = x1 + t * (x2 - x1)
            low = np.where(valid & (xe < low), xe, low)
            high = np.where(valid & (xe > high), xe, high)
        columns = self.shape[1]
        valid_rows = np.nonzero(np.isfinite(low) & np.isfinite(high) & (high >= low))[0]
        if valid_rows.size == 0:
            return grid
        diff = np.zeros((self.shape[0], columns + 1), dtype=np.int16)
        start = np.clip(np.ceil((low[valid_rows] - self.origin[0]) / self.step - 0.5),
                        0, columns).astype(int)
        end = np.clip(np.floor((high[valid_rows] - self.origin[0]) / self.step - 0.5),
                      -1, columns - 1).astype(int)
        np.add.at(diff, (valid_rows, start), 1)
        np.add.at(diff, (valid_rows, end + 1), -1)
        return np.cumsum(diff, axis=1)[:, :columns] > 0

    def extreme_uncovered(self, polygons):
        covered = np.zeros(self.shape, dtype=bool)
        for polygon in polygons:
            covered |= self.paint(polygon)
        open_cells = self.inside & ~covered
        if not open_cells.any():
            return None
        rows, cols = np.nonzero(open_cells)
        xs = self.origin[0] + (cols + 0.5) * self.step
        ys = self.origin[1] + (rows + 0.5) * self.step
        score = xs * SEARCH_DIRECTION[0] + ys * SEARCH_DIRECTION[1]
        order = np.argsort(score)
        # 栅格格子中心可能落在轮廓之外（格子跨在边界上），那样的点没有任何块能覆盖它，
        # 直接当作锚点会让搜索立刻判死（实测三角形就是这样 0.1 秒报无解）。
        # 因此按极值顺序找第一个"精确落在轮廓内"的中心。
        for index in order:
            point = np.array([xs[index], ys[index]])
            if geo.point_in_polygon(point, self._polygon):
                return point
        return None



def _frontier_ok(point, silhouette, placed, radius=0.01, samples=12):
    """点周围是否还有"在轮廓内且未被覆盖"的小块，避免选到已经填满的角。"""
    open_count = 0
    total = 0
    for k in range(samples):
        angle = 2.0 * math.pi * k / samples
        probe = point + radius * np.array([math.cos(angle), math.sin(angle)])
        if not geo.point_in_polygon(probe, silhouette):
            continue
        total += 1
        if not any(geo.point_in_polygon(probe, polygon) for polygon in placed):
            open_count += 1
    return total > 0 and open_count / float(total) > 0.02


def _anchor_candidates(silhouette, placed):
    """候选锚点：轮廓顶点 + 已放块顶点。这些点必定落在格点上。"""
    points = [np.asarray(p, dtype=float) for p in silhouette]
    for polygon in placed:
        for vertex in polygon:
            points.append(np.asarray(vertex, dtype=float))
    return points


def pick_anchor(silhouette, placed, coverage):
    """按极值顺序挑一个"仍未被覆盖、且周围还有空"的顶点作锚点。

    锚点必须是格点上的真实角点：栅格格子中心常常落在轮廓外或边界毛边上，
    没有任何块能覆盖它，搜索会立刻判死（三角形与 V 形都踩过这个坑）。
    """
    scored = []
    for point in _anchor_candidates(silhouette, placed):
        scored.append((float(np.dot(point, SEARCH_DIRECTION)),
                       round(float(point[0]), 9), round(float(point[1]), 9), point))
    scored.sort(key=lambda item: item[:3])
    for _, _, _, point in scored:
        if any(geo.point_in_polygon(point, polygon) and
               not _on_boundary(point, polygon) for polygon in placed):
            continue
        if not _frontier_ok(point, silhouette, placed):
            continue
        return point
    return coverage.extreme_uncovered([np.asarray(p, dtype=float) for p in placed])


def _on_boundary(point, polygon, eps=1e-9):
    count = len(polygon)
    for i in range(count):
        start, end = polygon[i], polygon[(i + 1) % count]
        direction = end - start
        length2 = float(np.dot(direction, direction))
        if length2 < 1e-18:
            continue
        t = float(np.dot(point - start, direction)) / length2
        if -1e-9 <= t <= 1.0 + 1e-9:
            projection = start + t * direction
            if np.linalg.norm(point - projection) <= eps:
                return True
    return False

# ---------------------------------------------------------------- 主求解

def solve(silhouette, max_nodes=200000, time_limit=120.0, log=None, step=LATTICE_STEP):
    """求解。silhouette 需已归一化到面积 1；返回 SolveResult 或 None。"""
    target, basis, snap_error = snap_silhouette(silhouette)
    if log:
        log("轮廓吸附：%s格点，最大偏差 %.4f" % (basis, snap_error))
    if abs(geo.area(target) - 1.0) > 0.05:
        return None

    shapes = piece_shapes()
    shapes.sort(key=lambda shape: -geo.area(shape["poly"]))
    candidates = enumerate_placements(target, shapes, step=step, log=log)
    if log:
        log("按块统计：%s" % {kind: sum(1 for c in candidates if c.kind == kind)
                             for kind in ("L", "M", "T", "Q", "P")})

    coverage = _Coverage(target)
    chosen = []
    state = {"nodes": 0, "start": time.time(), "timed_out": False}
    by_kind = {}
    for item in candidates:
        by_kind.setdefault(item.kind, []).append(item)

    def search(depth):
        if depth == len(shapes):
            return True
        state["nodes"] += 1
        if state["nodes"] >= max_nodes or time.time() - state["start"] > time_limit:
            state["timed_out"] = True
            return False

        anchor = pick_anchor(target, [item.polygon for item in chosen], coverage)
        if anchor is None:
            return depth == len(shapes)

        used = {(item.kind, item.index) for item in chosen}
        for shape in shapes:
            key = (shape["kind"], shape["index"])
            if key in used:
                continue
            for candidate in by_kind.get(shape["kind"], []):
                if candidate.index != shape["index"]:
                    continue
                if not candidate.contains(anchor):
                    continue
                if any(geo.convex_overlap(candidate.polygon, other.polygon) for other in chosen):
                    continue
                chosen.append(candidate)
                if search(depth + 1):
                    return True
                chosen.pop()
                if state["timed_out"]:
                    return False
        return False

    found = search(0)
    elapsed = time.time() - state["start"]
    if not found:
        return None
    placements = [{"kind": item.kind, "index": item.index, "angle": item.angle,
                   "polygon": item.polygon} for item in chosen]
    return SolveResult(placements, state["nodes"], elapsed, len(candidates),
                       target=target, basis=basis, snap_error=snap_error)

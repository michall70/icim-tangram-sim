"""七巧板平铺求解器：给定一条"目标轮廓"，求出七块该怎么摆。

**纯 numpy，禁止 import cv2**：必须能在无相机、无实物的条件下用合成图形穷举验证。

## 算法

目标是**精确覆盖**：七块面积和恒为 1（S=1），只要每块都落在轮廓内且两两不重叠，
它们的并集必然等于轮廓，覆盖性自动成立。于是问题化为"放得进去且不重叠"。

搜索用「极值未覆盖顶点」剪枝：

1. **候选位姿**只从精确几何产生：轮廓顶点、已放顶点、以及所有边所在**直线**的交点。
   拼块顶点一定落在这些点上（线段求交不够——顶点可能落在两条直线延长线的交点）。
2. **选锚点**：在这些候选点里，取对方向 (1, 0.37) 投影最小、且**仍未被覆盖**的那个。
   是否已被覆盖用一个 160×160 的布尔栅格判断，代价是常数级。
3. **生成子节点**：枚举「哪块 × 哪个顶点 × 8 个 45° 朝向」，把该顶点平移到锚点。
4. **逐条检查**：整块在轮廓内、与每块已放块不重叠、与已有结构有足够的共线接触。
5. 七块放完即得解。

三条约束都来自这套玩具有的几何事实，不是调参：
- **禁止镜像**：平行四边形是唯一手性块，反射会得到物理上不存在的零件；
- **45° 朝向**：所有边只沿 45° 的整数倍方向；
- **必须有共线接触**：只在角上碰到不算贴合，否则搜索会掉进大量死路（实测正方形解不出）。
"""

import math
import time

import numpy as np

from scenarios.tangram_puzzle.tangram import geometry as geo

# 搜索方向：故意取一个"不规整"的方向，避免多个点投影恰好并列
SEARCH_DIRECTION = np.array([1.0, 0.37]) / math.hypot(1.0, 0.37)

# 新放的一块与已有结构（轮廓或已放块）必须有的共线接触长度（单位 S）
MIN_CONTACT = 0.02

# 栅格分辨率：只用于判断"候选点是否还被覆盖"，不参与精度
RASTER_RESOLUTION = 160

# 每个节点最多把多少个候选锚点当作选择点。
# 实测：>1 会让分支按倍数膨胀——取 2 时连单位正方形都解不出（原 1164 节点）。
# 因此默认 1，即"只认最极值的那个未覆盖候选点"；要探索多锚点需同时大幅提高预算。
MAX_ANCHORS = 1

# 锚点周围"仍有未覆盖楔形"的最低比例。低于它说明这个角已经填满，必须跳过，
# 否则锚点会在同一个点上反复重选，搜索必死。
MIN_FRONTIER = 0.03

ROTATIONS = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)


class SolveResult:
    """求解结果。placements 为 [{'kind','index','angle','polygon'}]，坐标已归一化（S=1）。"""

    def __init__(self, placements, nodes, seconds, silhouette_area):
        self.placements = placements
        self.nodes = nodes
        self.seconds = seconds
        self.silhouette_area = silhouette_area

    def polygons(self):
        return [item["polygon"] for item in self.placements]

    def summary(self):
        return ", ".join("%s%d" % (item["kind"], item["index"]) for item in self.placements)


# ---------------------------------------------------------------- 件库本地形状

def piece_forms():
    """七块在 S=1 下的本地多边形（质心在原点、逆时针）。"""
    large_leg = 1.0 / math.sqrt(2.0)           # 大三角直角边
    small_leg = 1.0 / (2.0 * math.sqrt(2.0))   # 小三角直角边 = 正方形边长

    def triangle(leg):
        return np.array([[0.0, 0.0], [leg, 0.0], [0.0, leg]])

    forms = []
    for index in (1, 2):
        forms.append({"kind": "L", "index": index, "poly": triangle(large_leg)})
    forms.append({"kind": "M", "index": 1, "poly": triangle(0.5)})
    for index in (1, 2):
        forms.append({"kind": "T", "index": index, "poly": triangle(small_leg)})
    forms.append({"kind": "Q", "index": 1, "poly": np.array(
        [[0.0, 0.0], [small_leg, 0.0], [small_leg, small_leg], [0.0, small_leg]])})
    # 平行四边形：长边 1/2、短边 1/(2√2)、夹角 45°；手性固定
    forms.append({"kind": "P", "index": 1, "poly": np.array(
        [[0.0, 0.0], [0.5, 0.0], [0.5 + 0.25, 0.25], [0.25, 0.25]])})

    for form in forms:
        centered = geo.ensure_ccw(form["poly"])
        form["poly"] = centered - geo.centroid(centered)
    return forms


# ---------------------------------------------------------------- 栅格

class _Grid:
    """轮廓 + 已放块的布尔栅格，只用来判断候选点是否仍未被覆盖。"""

    def __init__(self, silhouette, resolution=RASTER_RESOLUTION):
        x0, y0, x1, y1 = geo.bounding_box(silhouette)
        span = max(x1 - x0, y1 - y0)
        if span <= 0:
            raise ValueError("轮廓尺寸为 0")
        self.origin = np.array([x0, y0], dtype=float)
        self.step = span / float(resolution)
        self.shape = (int(math.ceil((y1 - y0) / self.step)) + 1,
                      int(math.ceil((x1 - x0) / self.step)) + 1)
        self.silhouette = self._fill_polygon(silhouette)

    def _fill_polygon(self, polygon):
        """奇偶规则扫描线填充（轮廓可能非凸，只在构造时算一次）。"""
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

    def convex_mask(self, polygon):
        """凸多边形的向量化栅格化（每行只需最小/最大 x，用差分数组批量填充）。"""
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
            return np.zeros(self.shape, dtype=bool)
        diff = np.zeros((self.shape[0], columns + 1), dtype=np.int16)
        start = np.clip(np.ceil((low[valid_rows] - self.origin[0]) / self.step - 0.5),
                        0, columns).astype(int)
        end = np.clip(np.floor((high[valid_rows] - self.origin[0]) / self.step - 0.5),
                      -1, columns - 1).astype(int)
        np.add.at(diff, (valid_rows, start), 1)
        np.add.at(diff, (valid_rows, end + 1), -1)
        return np.cumsum(diff, axis=1)[:, :columns] > 0

    def covered_mask(self, placed):
        covered = np.zeros(self.shape, dtype=bool)
        for polygon in placed:
            covered |= self.convex_mask(polygon)
        return covered

    def is_uncovered(self, point, covered, radius=1):
        """点附近（2*radius+1 方格）是否存在"在轮廓内且未覆盖"的格子。

        锚点通常恰好落在已覆盖/未覆盖的交界上，只看单格会把它误判成已覆盖而整体跳过
        （实测：大三角形因此被判成无解）。看邻域是为了容下这种边界情形。
        """
        col = int(math.floor((point[0] - self.origin[0]) / self.step))
        row = int(math.floor((point[1] - self.origin[1]) / self.step))
        row0, row1 = max(0, row - radius), min(self.shape[0], row + radius + 1)
        col0, col1 = max(0, col - radius), min(self.shape[1], col + radius + 1)
        if row0 >= row1 or col0 >= col1:
            return False
        open_cells = self.silhouette[row0:row1, col0:col1] & ~covered[row0:row1, col0:col1]
        return bool(open_cells.any())


# ---------------------------------------------------------------- 候选点

def _line_intersection(point_a, direction_a, point_b, direction_b, eps=1e-9):
    """两条无限直线的交点；平行返回 None。"""
    denominator = direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    if abs(denominator) < eps:
        return None
    t = ((point_b[0] - point_a[0]) * direction_b[1] -
         (point_b[1] - point_a[1]) * direction_b[0]) / denominator
    return point_a + t * direction_a


def _edge_lines(silhouette, placed):
    lines = []

    def add_lines(polygon):
        count = len(polygon)
        for i in range(count):
            start = np.asarray(polygon[i], dtype=float)
            end = np.asarray(polygon[(i + 1) % count], dtype=float)
            vector = end - start
            length = float(np.linalg.norm(vector))
            if length > 1e-12:
                lines.append((start, vector / length))

    add_lines(silhouette)
    for polygon in placed:
        add_lines(polygon)
    return lines


def anchor_candidates(silhouette, placed):
    """候选锚点：轮廓顶点 ∪ 已放顶点 ∪ 所有边所在直线的交点。"""
    points = [np.asarray(p, dtype=float) for p in silhouette]
    for polygon in placed:
        for vertex in polygon:
            points.append(np.asarray(vertex, dtype=float))
    lines = _edge_lines(silhouette, placed)
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            hit = _line_intersection(lines[i][0], lines[i][1], lines[j][0], lines[j][1])
            if hit is not None:
                points.append(hit)
    return points


def _on_boundary(point, polygon, eps=1e-9):
    """点是否落在多边形边界上（含端点）。"""
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


def _frontier_fraction(point, silhouette, placed, radius=0.012, samples=16):
    """在 point 周围打一圈探针，返回"在轮廓内且尚未被覆盖"的角度占比。

    为什么必须有这一条：一块放下之后，它与轮廓的交点仍然是"边界点"，按定义算未覆盖；
    但它周围其实已经被填满了。若不检查局部还有没有空，锚点会永远停在同一个点上，
    每个分支都在那里死掉——实测 V 形与三角形就是这样被穷尽判成"无解"的。
    """
    covered = 0
    counted = 0
    for k in range(samples):
        angle = 2.0 * math.pi * k / samples
        probe = point + radius * np.array([math.cos(angle), math.sin(angle)])
        if not geo.point_in_polygon(probe, silhouette):
            continue
        counted += 1
        if any(geo.point_in_polygon(probe, polygon) for polygon in placed):
            covered += 1
    if counted == 0:
        return 1.0
    return 1.0 - covered / float(counted)


def anchor_sequence(candidates, grid, placed, silhouette, limit=MAX_ANCHORS):
    """按"对搜索方向投影从小到大"返回一串仍未被覆盖的候选锚点。

    为什么是"一串"而不是"一个"：只认最极值那一个点时，一旦该点不是某个块顶点应有的
    位置，整棵子树都会白搜（实测三角形与 V 形就是这样，900 秒穷尽仍无解）。把锚点本身
    也当成选择点，搜索才覆盖"顶点落在任意候选交点"的全部摆法。

    判定：先看栅格 3×3 邻域（锚点常落在已覆盖/未覆盖交界，单格判断会误杀），
    再用几何确认"不在任何已放块内部"；落在边界上算未覆盖——那正是要接着拼的缝。
    """
    scored = []
    for point in candidates:
        scored.append((float(np.dot(point, SEARCH_DIRECTION)),
                       round(float(point[0]), 9), round(float(point[1]), 9), point))
    scored.sort(key=lambda item: item[:3])

    covered = grid.covered_mask(placed)
    out = []
    for _, _, _, point in scored:
        if not grid.is_uncovered(point, covered):
            continue
        inside_some = False
        for polygon in placed:
            if geo.point_in_polygon(point, polygon) and not _on_boundary(point, polygon):
                inside_some = True
                break
        if inside_some:
            continue
        # 局部必须还有没被填满的楔形，否则这个点已经"用完"了
        if _frontier_fraction(point, silhouette, placed) <= MIN_FRONTIER:
            continue
        out.append(point)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------- 接触约束

def collinear_contact(poly_a, poly_b, eps=1e-6):
    """两块多边形之间共线重叠的边长之和（单位 S）。

    搜索的关键剪枝：拼块必须"贴着"已有结构，只在角上碰到不算贴。
    没有它，搜索会大量尝试仅角接触的摆法，实测连正方形都解不出。
    """
    total = 0.0
    points_a = np.asarray(poly_a, dtype=float)
    points_b = np.asarray(poly_b, dtype=float)
    count_a, count_b = len(points_a), len(points_b)
    for i in range(count_a):
        a1, a2 = points_a[i], points_a[(i + 1) % count_a]
        direction = a2 - a1
        length = float(np.linalg.norm(direction))
        if length < 1e-12:
            continue
        unit = direction / length
        normal = np.array([-unit[1], unit[0]])
        for j in range(count_b):
            b1, b2 = points_b[j], points_b[(j + 1) % count_b]
            other = b2 - b1
            other_length = float(np.linalg.norm(other))
            if other_length < 1e-12:
                continue
            if abs(abs(float(np.dot(unit, other / other_length))) - 1.0) > 1e-6:
                continue
            if abs(float(np.dot(b1 - a1, normal))) > eps or \
                    abs(float(np.dot(b2 - a1, normal))) > eps:
                continue
            t1 = float(np.dot(b1 - a1, unit))
            t2 = float(np.dot(b2 - a1, unit))
            low = max(0.0, min(t1, t2))
            high = min(length, max(t1, t2))
            if high > low:
                total += high - low
    return total


def contact_score(candidate, target, placed):
    """(总接触长度, 与轮廓的接触长度)。"""
    with_target = collinear_contact(candidate, target)
    total = with_target
    for polygon in placed:
        total += collinear_contact(candidate, polygon)
    return total, with_target


# ---------------------------------------------------------------- 主求解

def solve(silhouette, max_nodes=200000, time_limit=60.0):
    """求解。silhouette 需已归一化到面积 1；返回 SolveResult 或 None。"""
    target = geo.ensure_ccw(geo.as_array(silhouette))
    target_area = geo.area(target)
    if abs(target_area - 1.0) > 0.02:
        return None

    forms = piece_forms()
    forms.sort(key=lambda form: -geo.area(form["poly"]))
    variants = []
    for form in forms:
        per_angle = []
        for angle in ROTATIONS:
            rotated = geo.rotate(form["poly"], angle)
            vertices = [rotated[i] for i in range(len(rotated))]
            # 兜底锚点：边的 1/4、1/2、3/4 处。真实解里极值点可能落在某块的**边内部**
            # （T 型接缝），只锚顶点会漏掉这类摆法。这些点只在顶点锚定全失败后才启用。
            edge_points = []
            for i in range(len(rotated)):
                start, end = rotated[i], rotated[(i + 1) % len(rotated)]
                for fraction in (0.25, 0.5, 0.75):
                    edge_points.append(start + fraction * (end - start))
            per_angle.append((angle, rotated, vertices, edge_points))
        variants.append(per_angle)

    grid = _Grid(target)
    sx0, sy0, sx1, sy1 = geo.bounding_box(target)
    placed = []
    solution = []
    state = {"nodes": 0, "start": time.time(), "timed_out": False}

    def search(depth):
        if depth == len(forms):
            return True
        state["nodes"] += 1
        if state["nodes"] >= max_nodes or time.time() - state["start"] > time_limit:
            state["timed_out"] = True
            return False

        anchors = anchor_sequence(anchor_candidates(target, placed), grid, placed, target)
        if not anchors:
            return False


        used = {item["piece"] for item in solution}

        def try_place(anchor, wants_edges):
            """把某块的顶点（或边内部点）对到 anchor，递归下去。"""
            for index in range(len(forms)):
                if index in used:
                    continue
                for angle, rotated, vertices, edge_points in variants[index]:
                    for vertex in (edge_points if wants_edges else vertices):
                        candidate = rotated + (anchor - vertex)
                        x0, y0, x1, y1 = geo.bounding_box(candidate)
                        if x0 < sx0 - 1e-9 or y0 < sy0 - 1e-9 or \
                                x1 > sx1 + 1e-9 or y1 > sy1 + 1e-9:
                            continue
                        if not geo.polygon_inside(candidate, target):
                            continue
                        if any(geo.convex_overlap(candidate, other) for other in placed):
                            continue
                        total_contact, target_contact = contact_score(candidate, target, placed)
                        if total_contact < MIN_CONTACT:
                            continue
                        if not placed and target_contact < MIN_CONTACT:
                            continue
                        placed.append(candidate)
                        solution.append({"piece": index, "kind": forms[index]["kind"],
                                         "index": forms[index]["index"], "angle": angle,
                                         "polygon": candidate})
                        if search(depth + 1):
                            return True
                        placed.pop()
                        solution.pop()
                        if state["timed_out"]:
                            return None
            return False

        for anchor in anchors:
            for wants_edges in (False, True):
                outcome = try_place(anchor, wants_edges)
                if outcome is None:
                    return False
                if outcome:
                    return True
        return False

    found = search(0)
    elapsed = time.time() - state["start"]
    if not found:
        return None
    placements = [{"kind": item["kind"], "index": item["index"], "angle": item["angle"],
                   "polygon": item["polygon"]} for item in solution]
    return SolveResult(placements, state["nodes"], elapsed, target_area)

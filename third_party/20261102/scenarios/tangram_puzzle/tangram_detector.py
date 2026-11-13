"""当前固定工位七色检测。输入必须是与配置一致的去畸变图像。

保留完整像素坐标，按原料/目标 ROI 检测；不确定轮廓拒绝并写入 last_issues。
阈值仅在已记录样本验证，不代表任意光照、工位或实机精度保证。
"""

import math
from dataclasses import dataclass

import cv2 as cv
import numpy as np

ANNOTATE_BGR = {
    "Red": (60, 60, 255), "Orange": (60, 140, 255), "Yellow": (60, 255, 255),
    "Green": (60, 255, 60), "Blue": (255, 180, 60), "Purple": (255, 100, 200),
    "Coffee": (140, 90, 60),
}


@dataclass
class TangramPiece:
    piece_id: int
    color_name: str
    shape: str                # triangle / square / parallelogram
    center: tuple             # 轮廓质心（像素）
    grasp_point: tuple        # 拟合多边形质心，作为抓取点（像素）
    vertices: list            # 拟合多边形顶点（像素）
    angle: float              # 绕 Z 角度（deg，三角形=最长边方向+翻转修正，四边形=长对角线方向）
    contour: np.ndarray


class TangramDetector:
    """逐色 HSV 分割检测原料区/模板图案（7 色，阈值在 config.py DETECT_PARAMS）。"""

    def __init__(self, params: dict):
        self.p = dict(params)
        self.min_area = self.p.get("min_area", 500)
        self.morph_size = self.p.get("morph_size", 7)
        self.specs = self.p["specs"]
        self.last_issues = []
        # 非阻塞提示：不影响 validate_detection，但要让人看见"这里是靠兜底救回来的"
        self.last_notes = []
        self._roi = None
        # 判据加固开关（见 config.DETECT_PARAMS）：关掉即回到逐色硬阈值的旧行为
        self.use_shape_disambiguation = bool(self.p.get("use_shape_disambiguation", True))
        self.ambiguous_pairs = [tuple(pair) for pair in self.p.get("ambiguous_pairs", [])]

    # ---------- 检测入口（原料区与模板同一套逐色方案） ----------

    def detect_raw_material(self, img, annotated=None):
        return self._detect(img, annotated, self.p.get("raw_roi"))

    def detect_template(self, img, annotated=None):
        return self._detect(img, annotated, self.p.get("template_roi"))

    def _rescue_color(self, hsv, spec):
        """逐色规则拒掉后的兜底：把该色全部连通域合并，按已知件型拟合。

        为什么可以这么做：七块**颜色互不相同**，同一颜色出现多个连通域，只可能是
        同一块被高光/阴影切开——物理上不存在"两块同色"。所以对七巧板而言
        "multiple substantial candidates"不是歧义，而是**必然该合并**。

        只在正常检测失败后才调用，因此不会改变本来就能检出的那些块的行为。
        """
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in spec["ranges"]:
            mask |= cv.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = self._clean_mask(mask)
        # 只保留"与最大连通域相邻"的块：被高光劈开的两半会合回来，而远处被误捕的
        # 同色杂点（例如深色托盘上的反光）会被剔除——否则凸包会变成一个巨大异形，
        # 拟合度必然不过（实测紫色块就卡在这里）。
        count, labels, stats, centroids = cv.connectedComponentsWithStats(mask, 8)
        if count <= 1:
            return None
        areas = [(int(stats[i, cv.CC_STAT_AREA]), i) for i in range(1, count)]
        areas.sort(reverse=True)
        biggest_area, biggest = areas[0]
        if biggest_area < self.min_area:
            return None
        limit = max(30.0, 0.45 * math.sqrt(biggest_area))
        anchor = centroids[biggest]
        keep = np.zeros(mask.shape, np.uint8)
        for area, index in areas:
            if float(np.linalg.norm(centroids[index] - anchor)) <= limit:
                keep[labels == index] = 255
        points = cv.findNonZero(keep)
        if points is None or len(points) < max(30, self.min_area * 0.2):
            return None
        hull = cv.convexHull(points)
        if cv.contourArea(hull) < self.min_area:
            return None
        want = 4 if spec["shape"] in ("square", "parallelogram") else 3
        vertices = self._order_points(self._fit_polygon(hull, want))
        if vertices is None or len(vertices) != want:
            return None
        if not self._valid_edge_ratio(vertices):
            return None
        # 必须验拟合度：圆形/异形色块硬套三角也能凑出合法边长比，靠 IoU 挡掉。
        if self._polygon_iou(hull, vertices) < 0.90:
            self.last_issues.append(f"{spec['name']}: 兜底拟合度不足，拒绝")
            return None
        x, y, rw, rh = self._roi
        if (vertices[:, 0].min() <= x + 1 or vertices[:, 1].min() <= y + 1
                or vertices[:, 0].max() >= x + rw - 2 or vertices[:, 1].max() >= y + rh - 2):
            self.last_issues.append(f"{spec['name']}: 兜底方案也会触 ROI 边界，仍拒绝")
            return None
        grasp = self._polygon_centroid(vertices)
        if grasp is None:
            return None
        self.last_notes.append(
            f"{spec['name']}: 逐色规则未通过，已用'同色合并'兜底救回")
        return TangramPiece(
            piece_id=spec["id"],
            color_name=spec["name"],
            shape=spec["shape"],
            center=(int(round(grasp[0])), int(round(grasp[1]))),
            grasp_point=(int(round(grasp[0])), int(round(grasp[1]))),
            vertices=[(float(px_), float(py_)) for px_, py_ in vertices],
            angle=float(self._polygon_angle(vertices)),
            contour=hull,
        )

    def _detect(self, img, annotated=None, roi=None):
        self.last_issues = []
        self.last_notes = []
        h, w = img.shape[:2]
        if self.p.get("image_size") and (w, h) != tuple(self.p["image_size"]):
            raise ValueError("图像尺寸与当前工位配置不一致，禁止自动缩放后继续检测")
        self._roi = roi or (0, 0, w, h)
        x, y, rw, rh = self._roi
        if min(x, y) < 0 or min(rw, rh) <= 0 or x + rw > w or y + rh > h:
            raise ValueError("检测区域超出图像")
        masked = np.full_like(img, 255)
        masked[y:y+rh, x:x+rw] = img[y:y+rh, x:x+rw]
        hsv = cv.cvtColor(masked, cv.COLOR_BGR2HSV)
        prepared = self._prepare_masks(hsv)
        pieces = []
        for spec in self.specs:
            piece = self._detect_color(hsv, spec, prepared.get(spec["id"]))
            if piece is not None:
                pieces.append(piece)
        # 约束兜底：七块颜色互不相同 ⇒ 同色即同一块。被逐色规则拒掉的，用它救回来。
        found_ids = {piece.piece_id for piece in pieces}
        for spec in self.specs:
            if spec["id"] in found_ids:
                continue
            rescued = self._rescue_color(hsv, spec)
            if rescued is not None:
                pieces.append(rescued)
                found_ids.add(spec["id"])

        rejected = set()
        masks = []
        for piece in pieces:
            mask = np.zeros((h, w), np.uint8)
            cv.drawContours(mask, [piece.contour], -1, 1, cv.FILLED)
            masks.append(mask.astype(bool))
        for i in range(len(pieces)):
            for j in range(i):
                overlap = np.count_nonzero(masks[i] & masks[j]) / max(1, min(masks[i].sum(), masks[j].sum()))
                if overlap > self.p.get("conflict_overlap", 0.20):
                    rejected.update((i, j))
                    self.last_issues.append(f"conflict IDs {pieces[j].piece_id}/{pieces[i].piece_id}")
        pieces = [p for i, p in enumerate(pieces) if i not in rejected]
        if annotated is not None:
            cv.rectangle(annotated, (x,y), (x+rw-1,y+rh-1), (220,220,220), 1)
            for piece in pieces:
                self._annotate(annotated, piece)
        return pieces

    def _detect_color(self, hsv, spec, mask=None):
        """检测候选；触边、多实体歧义或拟合不良时拒绝并记录原因。

        mask 由 `_prepare_masks` 统一给出（已做歧义消解与断口桥接）；
        直接调用或关闭加固开关时会回退到本方法内的逐色硬阈值。
        """
        if mask is None:
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in spec["ranges"]:
                mask |= cv.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask = self._clean_mask(mask)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contours = [c for c in contours if cv.contourArea(c) >= self.min_area]
        if not contours:
            return None
        contours.sort(key=cv.contourArea, reverse=True)
        cnt = contours[0]
        if len(contours) > 1 and cv.contourArea(contours[1]) > 0.25 * cv.contourArea(cnt):
            self.last_issues.append(f"{spec['name']}: multiple substantial candidates")
            return None
        x, y, rw, rh = self._roi
        points = cnt.reshape(-1, 2)
        if (points[:,0].min() <= x+1 or points[:,1].min() <= y+1 or
                points[:,0].max() >= x+rw-2 or points[:,1].max() >= y+rh-2):
            self.last_issues.append(f"{spec['name']}: touches ROI boundary")
            return None

        want = 4 if spec["shape"] in ("square", "parallelogram") else 3
        vertices = self._order_points(self._fit_polygon(cnt, want))
        contour_mask = np.zeros(hsv.shape[:2], np.uint8)
        polygon_mask = np.zeros_like(contour_mask)
        cv.drawContours(contour_mask, [cnt], -1, 1, cv.FILLED)
        cv.fillPoly(polygon_mask, [np.rint(vertices).astype(np.int32)], 1)
        intersection = np.count_nonzero(contour_mask & polygon_mask)
        union = np.count_nonzero(contour_mask | polygon_mask)
        if intersection / max(union, 1) < self.p.get("min_fit_iou", 0.85):
            # 凹口修复：七块全部是凸多边形，掩模上的凹口只可能来自高光/阴影。
            # 用凸包重新拟合一次；要求「凸包拟合足够好」且「边长比合法」才采用，
            # 否则仍然拒绝——不能为了救回来而把两块粘连的伪形状当成一块。
            hull = cv.convexHull(cnt)
            hull_area = cv.contourArea(hull)
            solidity = cv.contourArea(cnt) / hull_area if hull_area > 0 else 0.0
            recovered = None
            if solidity < 0.92:
                candidate = self._order_points(self._fit_polygon(hull, want))
                if (self._polygon_iou(hull, candidate) >= 0.90
                        and self._valid_edge_ratio(candidate)):
                    recovered = candidate
            if recovered is None:
                self.last_issues.append(f"{spec['name']}: poor polygon fit")
                return None
            vertices = recovered
            self.last_notes.append(
                "%s: 掩模有凹口(实心度 %.2f)，已按凸包重拟合" % (spec["name"], solidity))
        m = cv.moments(cnt)
        center = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
        grasp = self._polygon_centroid(vertices) or center
        return TangramPiece(
            piece_id=spec["id"],
            color_name=spec["name"],
            shape=spec["shape"],
            center=center,
            grasp_point=grasp,
            vertices=[(int(x), int(y)) for x, y in vertices],
            angle=self._polygon_angle(vertices),
            contour=cnt,
        )

    # ---------- 掩模与几何拟合（兜底结果也必须通过质量检查） ----------

    def _clean_mask(self, mask):
        """开运算去散点/断细缝桥接 → 闭运算 → 按外轮廓重绘填洞（消除表面反光孔洞）。"""
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (self.morph_size, self.morph_size))
        m = cv.morphologyEx(mask, cv.MORPH_OPEN, k)
        m = cv.morphologyEx(m, cv.MORPH_CLOSE, k)
        contours, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        clean = np.zeros_like(m)
        cv.drawContours(clean, contours, -1, 255, cv.FILLED)
        return clean

    # ---------- 判据加固：形状仲裁与断口桥接 ----------
    #
    # 两种现场失效都源于"身份判据绑在照明相关的单点阈值上"：
    #   ① 橙(三角形,V>=200) 与 咖啡(正方形,V<=199) 只差 1 个灰阶 —— 暗光下整块橙被判成咖啡；
    #   ② 高光带把一块切成两段 —— 触发"multiple substantial candidates"并整块丢弃。
    # 两者的共同出路是：**形状没有歧义**（橙是三角形、咖啡是正方形），用形状仲裁。

    def _prepare_masks(self, hsv):
        """为每个颜色生成最终掩模：先按形状消解相邻色歧义，再桥接高光断口。"""
        raw = {}
        for spec in self.specs:
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in spec["ranges"]:
                mask |= cv.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            raw[spec["id"]] = mask
        if not self.use_shape_disambiguation:
            self.last_resolved = []
            return {sid: self._clean_mask(m) for sid, m in raw.items()}

        by_id = {spec["id"]: spec for spec in self.specs}
        for first, second in self.ambiguous_pairs:
            if first in raw and second in raw:
                raw[first], raw[second] = self._resolve_ambiguous(
                    raw[first], raw[second], by_id[first], by_id[second])

        prepared = {}
        for sid, mask in raw.items():
            prepared[sid] = self._bridge_split(self._clean_mask(mask))
        return prepared

    @staticmethod
    def _vertices_of(spec):
        return 4 if spec["shape"] in ("square", "parallelogram") else 3

    def _resolve_ambiguous(self, mask_a, mask_b, spec_a, spec_b):
        """两种颜色的掩模互相渗色时，按形状决定每块归属。

        只在两种颜色"难以用色相/亮度分开"时启用（config 的 ambiguous_pairs）。
        若某块形状既不像三角形也不像四边形（通常是两块粘连），则退回按像素占比，
        不做任何合并——宁可让它继续被拒绝，也不要把两块并成一块。
        """
        self.last_resolved = getattr(self, "last_resolved", [])
        union = cv.bitwise_or(mask_a, mask_b)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        union = cv.morphologyEx(union, cv.MORPH_CLOSE, kernel)
        contours, _ = cv.findContours(union, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        out_a = np.zeros_like(mask_a)
        out_b = np.zeros_like(mask_b)
        want_a, want_b = self._vertices_of(spec_a), self._vertices_of(spec_b)
        for contour in contours:
            if cv.contourArea(contour) < 0.5 * self.min_area:
                continue
            component = np.zeros_like(mask_a)
            cv.drawContours(component, [contour], -1, 255, cv.FILLED)
            count_a = int(np.count_nonzero(mask_a & component))
            count_b = int(np.count_nonzero(mask_b & component))
            total = count_a + count_b
            if total == 0:
                continue
            # 只有一方几乎不沾（<2%）时才可以不仲裁；否则一律交给形状——
            # 这一对颜色之所以被列为歧义对，就是因为色相/亮度分不开它们。
            # 早期版本用 85% 占比做捷径，结果暗光下橙块 87% 落在咖啡色里，
            # 直接被判给咖啡，形状仲裁形同虚设。
            if count_a <= 0.02 * total:
                out_b |= component
                continue
            if count_b <= 0.02 * total:
                out_a |= component
                continue
            vote = self._shape_vote(contour)
            if vote == want_a and want_a != want_b:
                out_a |= component
                self.last_resolved.append("%s: 形状仲裁(三角形)" % spec_a["name"])
            elif vote == want_b and want_a != want_b:
                out_b |= component
                self.last_resolved.append("%s: 形状仲裁(四边形)" % spec_b["name"])
            else:
                # 形状无法判定：保持各自原始像素，不合并
                out_a |= (mask_a & component)
                out_b |= (mask_b & component)
        return out_a, out_b

    def _shape_vote(self, contour):
        """判断轮廓更像三角形还是四边形；两者都拟合不良时返回 0。

        两条约束缺一不可：
        - **边长比**：七巧板任何一块的最短边/最长边都不小于 0.707，所以拟合出的多边形
          若有一条几乎为零的边，就是"把三角形的一个角切成两个点"的退化四边形。
          不加这条，三角形会被投票成四边形（实测 IoU 0.965 vs 0.970，只差 0.005）。
        - **少顶点优先**：三角形只要不比四边形明显差（0.02 以内）就选三角形。
        """
        scores = {}
        for want in (3, 4):
            polygon = self._fit_polygon(contour, want)
            if polygon is None:
                continue
            points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
            if len(points) < 3:
                continue
            edges = sorted(np.linalg.norm(points[(i + 1) % len(points)] - points[i])
                           for i in range(len(points)))
            if edges[-1] <= 0 or edges[0] < 0.62 * edges[-1]:
                continue
            scores[want] = self._polygon_iou(contour, polygon)
        if not scores:
            return 0
        best = max(scores, key=lambda key: scores[key])
        if 3 in scores and scores[3] >= scores[best] - 0.02:
            best = 3
        if scores[best] < self.p.get("min_fit_iou", 0.85):
            return 0
        return best

    def _bridge_split(self, mask):
        """高光带把一块切成两段时，用略大的闭运算把断口接上。

        只在本色掩模内操作：7 色互不相同，同一颜色不可能对应两块实物，
        所以"合起来仍是一块有效形状"时合并是安全的；合不成有效形状就原样返回，
        让后续的"多候选"检查继续拒绝它。
        """
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if len([c for c in contours if cv.contourArea(c) >= self.min_area]) <= 1:
            return mask
        for grow in (4, 10, 16):
            size = self.morph_size + grow
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))
            merged = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
            found, _ = cv.findContours(merged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            found = [c for c in found if cv.contourArea(c) >= self.min_area]
            if len(found) != 1:
                continue
            if self._shape_vote(found[0]):
                out = np.zeros_like(mask)
                cv.drawContours(out, found, -1, 255, cv.FILLED)
                return out
        return mask

    @staticmethod
    def _valid_edge_ratio(points, minimum=0.62):
        """七巧板任何一块的最短边/最长边都不小于 0.707；低于 minimum 视为伪多边形。"""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 3:
            return False
        edges = sorted(np.linalg.norm(pts[(i + 1) % len(pts)] - pts[i]) for i in range(len(pts)))
        return edges[-1] > 0 and edges[0] >= minimum * edges[-1]

    @staticmethod
    def _polygon_iou(contour, polygon):
        """多边形与轮廓掩模的交并比。"""
        x, y, w, h = cv.boundingRect(contour)
        pad = 3
        origin = np.array([x - pad, y - pad], dtype=np.float32)
        size = (h + 2 * pad, w + 2 * pad)
        actual = np.zeros(size, np.uint8)
        cv.drawContours(actual, [(contour - origin).astype(np.int32)], -1, 1, cv.FILLED)
        fitted = np.zeros(size, np.uint8)
        points = np.asarray(polygon, dtype=np.float32) - origin
        cv.fillPoly(fitted, [np.rint(points).astype(np.int32)], 1)
        union = np.count_nonzero(actual | fitted)
        return np.count_nonzero(actual & fitted) / union if union else 0.0

    @staticmethod
    def _fit_polygon(contour, want):
        """比较多种拟合的掩模交并比，避免第一个顶点数正确的粗拟合胜出。"""
        hull = cv.convexHull(contour)
        peri = cv.arcLength(hull, True)
        candidates = []
        for eps in (0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12):
            approx = cv.approxPolyDP(hull, eps * peri, True)
            if len(approx) == want:
                polygon = approx.reshape(-1, 2).astype(np.float32)
                candidates.append(polygon)
                refined = TangramDetector._refine_edges(contour, polygon)
                if refined is not None:
                    candidates.append(refined)
        if want == 3 and not candidates:
            area, tri = cv.minEnclosingTriangle(hull)
            if area > 0 and tri is not None:
                candidates.append(tri.reshape(-1, 2).astype(np.float32))
        elif want == 4 and not candidates:
            candidates.append(cv.boxPoints(cv.minAreaRect(hull)))
        # 局部画布，包含所有候选，随后仍由调用方执行最低质量门槛。
        all_points = np.vstack([contour.reshape(-1, 2)] + candidates)
        origin = np.floor(all_points.min(axis=0)).astype(int) - 2
        size = np.ceil(all_points.max(axis=0)).astype(int) - origin + 3
        actual = np.zeros((size[1], size[0]), np.uint8)
        cv.fillPoly(actual, [contour.reshape(-1, 2)-origin], 1)
        def score(points):
            fitted = np.zeros_like(actual)
            cv.fillPoly(fitted, [np.rint(points-origin).astype(np.int32)], 1)
            return np.count_nonzero(actual & fitted) / max(1, np.count_nonzero(actual | fitted))
        return max(candidates, key=score)

    @staticmethod
    def _refine_edges(contour, polygon):
        """用每条边中段的真实轮廓点拟合直线，交点恢复圆角遮掉的顶点。"""
        points = contour.reshape(-1, 2).astype(np.float32)
        starts = polygon
        ends = np.roll(polygon, -1, axis=0)
        vectors = ends - starts
        lengths2 = (vectors * vectors).sum(axis=1)
        if np.any(lengths2 < 1):
            return None
        t = ((points[:,None,:] - starts) * vectors).sum(axis=2) / lengths2
        projection = starts + np.clip(t, 0, 1)[:,:,None] * vectors
        distances = np.linalg.norm(points[:,None,:] - projection, axis=2)
        nearest = distances.argmin(axis=1)
        lines = []
        for i in range(len(polygon)):
            selected = points[(nearest == i) & (t[:,i] > 0.15) & (t[:,i] < 0.85)]
            if len(selected) < 4:
                return None
            vx, vy, x, y = cv.fitLine(selected, cv.DIST_HUBER, 0, 0.01, 0.01).flatten()
            lines.append((np.array([x,y]), np.array([vx,vy])))
        corners = []
        for i in range(len(lines)):
            a, u = lines[i-1]
            b, v = lines[i]
            matrix = np.column_stack((u,-v))
            if abs(np.linalg.det(matrix)) < 0.1:
                return None
            point = a + np.linalg.solve(matrix, b-a)[0]*u
            if np.linalg.norm(point-polygon[i]) > 0.2*np.sqrt(lengths2).max():
                return None
            corners.append(point)
        result = np.array(corners, np.float32)
        return result if cv.isContourConvex(result.reshape(-1,1,2)) else None

    @staticmethod
    def _order_points(pts):
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        return pts[np.argsort(angles)]

    @staticmethod
    def _polygon_centroid(pts):
        area2 = cx = cy = 0.0
        n = len(pts)
        for i in range(n):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % n]
            cross = x0 * y1 - x1 * y0
            area2 += cross
            cx += (x0 + x1) * cross
            cy += (y0 + y1) * cross
        if abs(area2) < 1e-6:
            mid = np.mean(pts, axis=0)
            return int(round(mid[0])), int(round(mid[1]))
        return int(round(cx / (3.0 * area2))), int(round(cy / (3.0 * area2)))

    @staticmethod
    def _polygon_angle(pts):
        """三角形=最长边方向+第三顶点叉积修正 180°翻转；四边形=长对角线方向。"""
        n = len(pts)
        if n == 3:
            max_len = 0.0
            idx1 = idx2 = -1
            for i in range(3):
                p1, p2 = pts[i], pts[(i + 1) % 3]
                length = math.hypot(*(p2 - p1))
                if length > max_len:
                    max_len = length
                    idx1, idx2 = i, (i + 1) % 3
            p1, p2 = pts[idx1].copy(), pts[idx2].copy()
            p3 = pts[3 - idx1 - idx2]
            if p1[0] > p2[0] or (p1[0] == p2[0] and p1[1] > p2[1]):
                p1, p2 = p2, p1
            dx, dy = float(p2[0] - p1[0]), float(p2[1] - p1[1])
            angle = math.degrees(math.atan2(dy, dx))
            if dx * float(p3[1] - p1[1]) - dy * float(p3[0] - p1[0]) > 0:
                angle += 180.0
            while angle > 180.0:
                angle -= 360.0
            while angle <= -180.0:
                angle += 360.0
            return angle
        if n == 4:
            d1, d2 = pts[2] - pts[0], pts[3] - pts[1]
            vec = d1 if math.hypot(*d1) >= math.hypot(*d2) else d2
            return math.degrees(math.atan2(vec[1], vec[0]))
        return 0.0

    # ---------- 标注 ----------

    @staticmethod
    def _annotate(vis, piece):
        color = ANNOTATE_BGR.get(piece.color_name, (128, 128, 128))
        cv.polylines(vis, [np.array(piece.vertices, np.int32)], True, color, 2)
        # OpenCV 5 对参数类型很严：center 必须是 python int 元组，
        # 传 numpy 浮点会报 "Can't parse 'center'"，把整次检测打断。
        for x, y in piece.vertices:
            cv.circle(vis, (int(round(float(x))), int(round(float(y)))), 3, color, -1)
        gx, gy = int(round(float(piece.grasp_point[0]))), int(round(float(piece.grasp_point[1])))
        cv.circle(vis, (gx, gy), 5, (0, 0, 0), -1)
        cv.circle(vis, (gx, gy), 3, (255, 255, 255), -1)
        rad = np.deg2rad(piece.angle)
        cv.arrowedLine(vis, (gx, gy), (int(gx + 70 * np.cos(rad)), int(gy + 70 * np.sin(rad))),
                       (0, 0, 255), 2, tipLength=0.25)
        label = f"{piece.color_name} id{piece.piece_id} {piece.angle:.1f}"
        width = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        label_x = max(5, min(gx + 12, vis.shape[1] - width - 5))
        cv.putText(vis, label, (label_x, max(20, gy - 10)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)


def validate_detection(pieces, required_ids, issues=()):
    """完整任务的检测门槛；不把少块/重复身份/拒绝候选当作成功。"""
    ids = [p.piece_id for p in pieces]
    if issues or len(ids) != len(set(ids)) or set(ids) != set(required_ids):
        raise ValueError(f"检测未通过: expected={sorted(required_ids)}, actual={ids}, issues={list(issues)}")

"""七巧板件库：以正方形边长 S 为单位的标称几何、实测拟合与形状归类。

本模块**不依赖 cv2 / 相机 / 机器人**，只做几何，便于用合成数据穷举验证。

标称几何（全部由 S 唯一确定，没有自由参数）：

    大三角 L ×2 : 直角边 S/√2,       斜边 S,        面积 S^2/4
    中三角 M ×1 : 直角边 S/2,        斜边 S/√2,     面积 S^2/8
    小三角 T ×2 : 直角边 S/(2√2),    斜边 S/2,      面积 S^2/16
    正方形 Q ×1 : 边 S/(2√2),                       面积 S^2/8
    平行四边形 P ×1 : 长边 S/2, 短边 S/(2√2), 夹角 45°, 面积 S^2/8

两条必须记住的不变量：

1. **面积和恰为 S^2**。因此轮廓面积可直接给出 S = sqrt(A)，尺度不需要额外标定；
   反过来，若「轮廓面积 / S_phys^2」偏离 1 超过几个百分点，说明那张图不是七块拼满的图形。
2. **所有边长只取四个值**：S·{1, 0.7071, 0.5, 0.3536}；所有内角都是 45° 的整数倍。
   因此件身份可以用「边长档位」判别，不必依赖颜色。

实测经验（2026-10-26，本项目那副七巧板，卡尺 0.1mm）：
   最小二乘 S = 115.84 mm，残差 RMS 0.91 mm、最大 1.26 mm；
   「本应等长」的小三角直角边 / 正方形边长 / 平行四边形短边实测为 39.7 / 42.2 / 39.7，
   极差 6.3%。**结论：实物不是数学精确件**，件库必须允许按实测覆盖，
   并且规划时不要追求零缝（详见 scripts 与 README 中的「外轮廓优先」约定）。
"""

import math

SQRT2 = math.sqrt(2.0)

# kind -> 每条边的长度系数（乘以 S 即得 mm）
EDGE_COEF = {
    "L": {"leg": 1.0 / SQRT2, "hyp": 1.0},
    "M": {"leg": 0.5, "hyp": 1.0 / SQRT2},
    "T": {"leg": 1.0 / (2.0 * SQRT2), "hyp": 0.5},
    "Q": {"side": 1.0 / (2.0 * SQRT2)},
    "P": {"long": 0.5, "short": 1.0 / (2.0 * SQRT2)},
}

# kind -> 面积系数（乘以 S^2 即得 mm^2）
AREA_COEF = {"L": 0.25, "M": 0.125, "T": 0.0625, "Q": 0.125, "P": 0.125}

# kind -> 件数
COUNT = {"L": 2, "M": 1, "T": 2, "Q": 1, "P": 1}

# kind -> 中文名与顶点数（求解器与标注共用）
KIND_NAME = {"L": "大三角", "M": "中三角", "T": "小三角", "Q": "正方形", "P": "平行四边形"}
KIND_VERTICES = {"L": 3, "M": 3, "T": 3, "Q": 4, "P": 4}
KIND_HANDED = {"L": False, "M": False, "T": False, "Q": False, "P": True}

# 边长档位：leg/S 的四个可能取值，用于把实测边长归到某个类
LEG_RATIOS = {"T": 1.0 / (2.0 * SQRT2), "M": 0.5, "L": 1.0 / SQRT2}


def library(scale, overrides=None):
    """按尺度 S（mm）展开七块，返回 [{'kind','index','edges'(mm),'area'(mm^2)}]。

    overrides: 可选 {kind: {边名: mm}}，用实测值覆盖标称值。现场换一副块时，
    先用 tools/measure_pieces.py 量出来再传进来，件库就跟着实物走。
    """
    if not (scale > 0) or not math.isfinite(scale):
        raise ValueError("S 必须为正的有限数")
    pieces = []
    for kind, count in COUNT.items():
        for index in range(1, count + 1):
            edges = {}
            for name, coef in EDGE_COEF[kind].items():
                key = (overrides or {}).get(kind, {}).get(name)
                edges[name] = float(key) if key else coef * scale
            pieces.append({"kind": kind, "index": index,
                           "name": "%s%d" % (kind, index),
                           "edges": edges,
                           "area": AREA_COEF[kind] * scale * scale})
    return pieces


def total_area(scale):
    """七块面积和，理论上恒等于 S^2。"""
    return scale * scale


def scale_from_silhouette(area_mm2):
    """由轮廓面积反推 S：轮廓若是七块拼满的图形，S = sqrt(面积)。"""
    if not (area_mm2 > 0):
        raise ValueError("面积必须为正")
    return math.sqrt(area_mm2)


def scale_consistency(silhouette_area_mm2, scale, tolerance=0.05):
    """轮廓面积与 S_phys 是否自洽；返回 (偏差比例, 是否通过)。

    偏差超过 tolerance 时，通常意味着：这张轮廓不是七巧板拼法、图像被裁过、
    或 S_phys 量错了。此时应拒绝求解，而不是给一个近似解。
    """
    if not (scale > 0):
        raise ValueError("S 必须为正")
    ratio = silhouette_area_mm2 / (scale * scale)
    return ratio - 1.0, abs(ratio - 1.0) <= tolerance


def estimate_scale(measurements):
    """由若干条实测边长做最小二乘，估计 S。

    measurements: [(kind, 边名, 实测mm), ...]；边名不在该类的 EDGE_COEF 里会报错。
    返回 (S, 残差列表)；残差为 (kind, 边名, 理论mm, 实测mm, 偏差比例)。
    """
    if not measurements:
        raise ValueError("没有实测数据")
    num = 0.0
    den = 0.0
    for kind, edge_name, value in measurements:
        coef = EDGE_COEF[kind][edge_name]
        num += coef * value
        den += coef * coef
    scale = num / den
    residuals = []
    for kind, edge_name, value in measurements:
        theory = EDGE_COEF[kind][edge_name] * scale
        residuals.append((kind, edge_name, theory, float(value),
                          (value - theory) / theory if theory else float("nan")))
    return scale, residuals


def residual_stats(residuals):
    """残差统计：返回 (RMS mm, 最大绝对 mm, 最大绝对比例)。"""
    if not residuals:
        return 0.0, 0.0, 0.0
    diffs = [abs(measured - theory) for _, _, theory, measured, _ in residuals]
    ratios = [abs(ratio) for _, _, _, _, ratio in residuals]
    rms = math.sqrt(sum(d * d for d in diffs) / len(diffs))
    return rms, max(diffs), max(ratios)


def classify(edges_mm, scale, tolerance=0.08):
    """按边长把一块实物归到类（L/M/T/Q/P），不依赖颜色。

    edges_mm: 该块的全部边长（三角形 3 条，四边形 4 条）。
    返回 (kind 或 None, 说明)。None 表示形状对不上任何一类。
    """
    if not edges_mm or not (scale > 0):
        raise ValueError("边长与 S 都必须有效")
    length = len(edges_mm)
    ordered = sorted(float(v) for v in edges_mm)
    if length == 3:
        leg, hyp = ordered[0], ordered[2]
        if abs(hyp / leg - SQRT2) > tolerance:
            return None, "非等腰直角三角形（斜边/直角边=%.3f，应为1.414）" % (hyp / leg)
        ratio = leg / scale
        best = min(LEG_RATIOS, key=lambda kind: abs(LEG_RATIOS[kind] - ratio))
        error = abs(LEG_RATIOS[best] - ratio) / LEG_RATIOS[best]
        if error > tolerance:
            return None, "直角边/ S = %.3f，与任何三角类不符" % ratio
        return best, "直角边/S=%.3f（标称 %.3f，偏差 %.1f%%）" % (
            ratio, LEG_RATIOS[best], 100 * error)
    if length == 4:
        longest, shortest = ordered[-1], ordered[0]
        if longest / shortest <= 1.0 + tolerance:
            ratio = longest / scale
            if abs(ratio - EDGE_COEF["Q"]["side"]) / EDGE_COEF["Q"]["side"] <= tolerance:
                return "Q", "四边等长，边长/S=%.3f" % ratio
            return None, "四边等长但边长/S=%.3f 与正方形标称不符" % ratio
        long_ratio = longest / scale
        short_ratio = shortest / scale
        want_long = EDGE_COEF["P"]["long"]
        want_short = EDGE_COEF["P"]["short"]
        if (abs(long_ratio - want_long) / want_long <= tolerance and
                abs(short_ratio - want_short) / want_short <= tolerance):
            return "P", "长/短边-S=%.3f/%.3f（标称 %.3f/%.3f）" % (
                long_ratio, short_ratio, want_long, want_short)
        return None, "四边形边长比不符（长/短边-S=%.3f/%.3f）" % (long_ratio, short_ratio)
    return None, "顶点数 %d 不属于七巧板任何一块" % length

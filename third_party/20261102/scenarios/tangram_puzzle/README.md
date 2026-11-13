# 七巧板拼图场景（tangram_puzzle）

基于 **FAIRINO FR3-C + RealSense D435i（臂端）+ 气动吸盘** 的七巧板识别与拼装。
从原料区抓取散乱摆放的七块，按目标图逐块放到成品区。

> 项目级状态、精度链与现场步骤请先读：
> [README](../../README.md) ｜ [项目现状与路线](../../docs/项目现状与路线.md)
> ｜ [技术条件与精度链](../../docs/技术条件与精度链.md)
> ｜ [现场操作手册](../../docs/现场操作手册.md)。
> 本文件只讲场景内部的代码结构、命令与限制。

---

## 1. 当前能力（2026-10-31）

| 能力 | 状态 |
|---|---|
| 目标图 → 七块布局 → 7 条计划（离线） | ✅ 可用 |
| 原料区检测 7/7（当前基准帧） | ✅ 可用 |
| 照片模板通道（回退） | ⚠️ 保留，赛规不允许赛前拼拍 |
| 实机抓放当前配置 | ❌ 未验收 |
| 白色剪影 → 计划 | ⚠️ 模块可用，`main` 无入口 |
| 成品复拍 / 落点闭环 | ❌ 未实现 |

**离线通过 ≠ 实机可用。** 详细证据见项目级文档。

---

## 2. 流程

```text
相机拍照位 → 去畸变 → 原料区 ROI 检测（7 色 HSV + 形状/面积仲裁）
                         │
目标来源 ────────────────┤
  A. 目标图（image，主）  │  → 七块布局 → S_phys 缩放 → 成品区锚点
  B. 实拍模板（photo）    │  → 模板块检测 → 目标位姿
                         ▼
              槽位 ↔ 实物指派（assign：同类双射 + P 手性）
                         ▼
              plan_puzzle：抓取/放置六维位姿
                         ▼
              几何门禁 → 42 端点 IK 预检 → 逐块抓放
```

### 目标图模式的关键点

目标图**不是相机帧**，所以：

- 不套单应；像素坐标只用于提取形状；
- 实物尺度由 `S_PHYS_MM` 决定；
- 放在台面哪由 `TARGET_ANCHOR_MM`（或 `template_roi` 中心）与 `TARGET_YAW_DEG` 决定；
- 放不放得下由成品区四边形 + `TARGET_AREA_MM` 判断。

照片模板模式仍保留：模板块与原料块共用同一个单应，前提是模板就是相机在拍照位拍的帧。

---

## 3. 模块地图

| 文件/目录 | 职责 |
|---|---|
| `main.py` | 入口：离线规划 / `--execute` 实机；目标图与照片两条通道 |
| `config.py` | **唯一参数入口**：点位、ROI、HSV、内参/单应、目标图、`S_phys` |
| `tangram_detector.py` | HSV 分割、形状/面积仲裁、断口桥接、多边形拟合、检测门禁 |
| `puzzle_planner.py` | ID 匹配、抓放位姿、吸盘偏心反补、预览、几何门禁 |
| `planning/target_image.py` | 目标图 → 七块 → 归一化 → 缩放/锚定/朝向 → 像素 |
| `planning/pattern.py` | 归一化求解结果 → 台面毫米位姿 → 像素 |
| `planning/assign.py` | 槽位 ↔ 实物：L/M/T 面积分档、Q/P 内角判据、P 手性 |
| `planning/scale.py` | 按 `S_phys` 做尺度归一（默认关闭） |
| `tangram/geometry.py` | 纯 NumPy 几何：面积/质心/内角、点在多边形内、SAT、45° 吸附 |
| `tangram/piece_set.py` | 件库标称几何、边长档位、面积系数 |
| `tangram/lattice_solver.py` | 格点枚举 + 精确覆盖（主求解器） |
| `tangram/solver.py` | 旧版顶点锚定 DFS（保留作回归基线） |
| `tangram/verify.py` | 几何层 + 栅格层自检 |
| `vision/silhouette.py` | 白色剪影提取、背景估计、45° 吸附 |
| `tools/` | 标定、ROI、颜色、尺寸、求解、目标图核对等工具 |
| `data/` | 测试夹具与基准输入（`legacy/` 为历史样本） |
| `results/` | 运行输出（不作为源代码依赖） |

---

## 4. 常用命令

所有命令在**仓库根**执行。

### 4.1 离线：目标图 + 原料帧

```powershell
# 只看目标图解析 + 原料检测 + 7 条计划
python -B -m scenarios.tangram_puzzle.tools.target_check `
  --target scenarios\tangram_puzzle\data\target_samples\target_mine.png `
  --raw    scenarios\tangram_puzzle\results\capture_check\exp312_gain0_wb3300_125104.png

# 走主流程（写 results/，不连设备）
python -B -m scenarios.tangram_puzzle.main `
  --target-image scenarios\tangram_puzzle\data\target_samples\target_mine.png `
  --raw          scenarios\tangram_puzzle\results\capture_check\exp312_gain0_wb3300_125104.png
```

### 4.2 离线：照片模板

```powershell
python -B -m scenarios.tangram_puzzle.main <模板原图> <原料原图>
```

### 4.3 实机

```powershell
python -B -m scenarios.tangram_puzzle.health_check        # 只读体检
python -B -m scenarios.tangram_puzzle.main --execute      # 真实运动
```

`--execute` 会回 HOME、到拍照位、松吸盘、逐块抓放；执行前请按
[现场操作手册](../../docs/现场操作手册.md) 完成体检与授权。

### 4.4 工具

```powershell
python -B -m scenarios.tangram_puzzle.capture_check              # 调曝光/增益/白平衡
python -B -m scenarios.tangram_puzzle.tools.roi_frame --image <帧>   # 画两个 ROI
python -B -m scenarios.tangram_puzzle.tools.roi_fit --raw L T R B --tray L T R B --apply
python -B -m scenarios.tangram_puzzle.tools.tune_colors --image <帧> --area raw [--apply]
python -B -m scenarios.tangram_puzzle.tools.diag_colors --image <帧> --area raw
python -B -m scenarios.tangram_puzzle.tools.roi_autofit --image <帧>  # 近似估算（只打印）
python -B -m scenarios.tangram_puzzle.tools.measure_pieces --image <帧>
python -B -m common.calibration        # 手动 9 点重标（交互）
python -B -m scenarios.tangram_puzzle.tools.solve_check --image <剪影> --scale 115.84
python -B -m scenarios.tangram_puzzle.tools.check_geometry --template <图> --raw <图>
python -B -m scenarios.tangram_puzzle.detect_check --mode raw --image <图>
python -B -m scenarios.tangram_puzzle.verify_calibration [图像]
```

---

## 5. 关键配置键（`config.py`）

| 键 | 当前值 | 说明 |
|---|---|---|
| `ROBOT_IP` / `TOOL` / `USER` | `192.168.58.2` / 0 / 0 | 位姿原点是法兰 |
| `VEL` / `APPROACH_DZ` | 20% / 30 mm | 运动速度、接近高度 |
| `HOME_JOINT` / `CAPTURE_POSE` | 见文件 | 同一点（P22） |
| `TCP_RPY` | `[-178.633, 0.416, 139.234]` | 抓放固定姿态 |
| `GRIP_Z` / `PLACE_Z` | 348.860 / 351.500 | 法兰 z |
| `S_PHYS_MM` | 115.84 | 正方形边长，现场换件需重测 |
| `ENABLE_SCALE_NORMALIZATION` | False | 未实机验证，保持关闭 |
| `SUCTION_OFFSET_XY` | (0,0) | 未测占位 |
| `CAMERA_PARAMS` | 312 / 0 / 3300 | 曝光/增益/白平衡 |
| `raw_roi` / `template_roi` | (148,98,598,603) / (772,166,481,481) | 像素框 |
| `TARGET_SOURCE` | `"image"` | 目标图为主，`"photo"` 回退 |
| `TARGET_IMAGE_PATH` | `target_mine.png` | 目标图 |
| `TARGET_YAW_DEG` | None | 自动：先 0°，放不下再对齐成品区 |
| `TARGET_ANCHOR_MM` | None | 用 `template_roi` 中心 |
| `TARGET_AREA_MM` | (180,180) | 可用区尺寸 |

---

## 6. 数据目录

```text
data/
├─ stage3_20261025/      回归测试夹具（1_Color、template_Color_Color、blank_Color）★ 不要删
├─ stage3_20261029b/     当前照片模板（回退通道）
├─ glare_20261026/       三档光照实验（偏暗/正常/强光）
├─ raw_samples/          原料区基准帧
├─ target_samples/       目标图样本（target_mine / square / triangle）
├─ silhouette_tests/     剪影实验集（含 answers）
├─ silhouette_sample.png 单张剪影样本
└─ legacy/               历史模板与旧样本（不参与运行）
```

`data/README.md` 与 `data/raw_samples/README.md` 有各自的使用说明。

---

## 7. 测试

```powershell
python -B -m unittest `
  scenarios.tangram_puzzle.test_planning `
  scenarios.tangram_puzzle.test_stage3 `
  scenarios.tangram_puzzle.test_solver `
  scenarios.tangram_puzzle.test_roi_frame `
  scenarios.tangram_puzzle.test_target
```

5 个模块、75 项，**不连接相机/机器人**。覆盖：

- 规划数学、透视旋转、偏心反补、42 端点 IK 预检（mock RPC）；
- 阶段三检测门禁（缺块/触边/跨色/错误分辨率/圆形拒绝）；
- 件库几何、格点求解、几何自检、pattern/assign；
- ROI 工具、目标图解析。

**不覆盖**：真实相机、真实机器人、真实抓放、仿真。

---

## 8. 已知限制

1. **实机未验收**：当前参数组合没有完整 7 块落点数据。
2. **强光 5/7**：镜面高光会把掩模咬掉，代码兜底只能到 5/7。
3. **单块失败即终止**：`validate_detection` 要求 7/7，没有重拍/跳过。
4. **无落点闭环**：放完不复拍，偏差无法逐块量化。
5. **标定绑定本机**：只有 2026-09-10 的手动 9 点 `HOMOGRAPHY`，与拍照位/内参强绑定；一键九点已移除。
6. **目标图余量 0.2 mm**：实机贴边风险，需改 yaw/换图/改可用区。
7. **剪影无主流程入口**：求解、pattern、assign 已有，但 `main` 只走彩色/照片。
8. **照片模板不可用于赛场**：赛规不允许赛前拼拍，`photo` 仅回退。

---

## 9. 相关文档

- [项目现状与路线](../../docs/项目现状与路线.md)
- [技术条件与精度链](../../docs/技术条件与精度链.md)
- [现场操作手册](../../docs/现场操作手册.md)
- [仿真对接说明](../../docs/仿真对接说明.md)
- [代码评审与问题清单](../../DeepSeek.md)

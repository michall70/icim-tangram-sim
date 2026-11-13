# 七巧板机器人拼装（ICIM 2026）

用 **FAIRINO FR3-C 协作机械臂 + Intel RealSense D435i（臂端）+ 气动吸盘**，
识别工作台上随机平铺的七巧板，并按给定目标图逐块抓取、搬运、拼装。

赛题有两个难度：

- **基础任务**：目标图是**七彩色块示意图**，程序做"识别—匹配—抓放"；
- **进阶任务**：目标图是**白色剪影**（无颜色、无内部分界线），程序必须先自己求出
  七块怎么摆，再执行拼装。

本仓库是**真机执行与总集成**这一侧的实现：视觉识别、目标图解析、抓放规划、
机器人通信、标定工具与现场调试工具。三维仿真由队友另行负责，**不在本仓库内**
（见 [仿真对接说明](docs/仿真对接说明.md)）。

---

## 当前状态（2026-10-31）

| 能力 | 状态 | 依据 / 缺口 |
|---|---|---|
| 彩色基础任务 · 离线 | **可用** | 目标图 + 原料帧 → 原料 7/7 → 7 条计划；75 项离线测试通过 |
| 彩色基础任务 · 实机 | **未通过验收** | 2026-10-28 曾完整跑通 7 块但放置偏移大；当前参数组合尚未实机复验 |
| 目标图通道（赛场方案） | **离线可用** | 不依赖赛前拼拍模板；当前 `target_mine.png` 布局余量仅约 0.2 mm，偏紧 |
| 白色剪影 · 求解 | **离线部分可用** | 公告 Fig 2 三角形可解；26 张合法剪影 8 s 内 20/26；`main` 尚无剪影入口 |
| 换机可迁移性 | **未验证** | 只有 2026-09-10 的手动 9 点 `HOMOGRAPHY`；一键九点已移除，换机需手动重标 |
| 三维仿真 | **不在本仓库** | 仅提供接口与数值约定；队友侧状态**待确认** |

> **不要把离线通过当成实机可用。** 离线只证明"能在保存的图上检出七块并生成计划"，
> 不证明逆解、路径、吸取、落点。硬件功能凡未实测，本文档与子文档都会明确标注。

---

## 快速开始（不接硬件）

要求：Python 3.8 及以上（本机实测 3.14.6）、OpenCV、NumPy、SciPy、
`pyrealsense2`（只用相机时需要）。现场机器人 SDK 已随仓库附带，无需安装。

```powershell
# 1) 安装依赖（真实相机采集还需要 Intel RealSense SDK 2.x）
python -m pip install -r requirements.txt

# 2) 离线目标图 + 原料帧：七块 → 指派 → 7 条抓放计划
python -B -m scenarios.tangram_puzzle.tools.target_check `
  --target scenarios\tangram_puzzle\data\target_samples\target_mine.png `
  --raw    scenarios\tangram_puzzle\results\capture_check\exp312_gain0_wb3300_125104.png

# 3) 走主流程（离线，只写 results/，不连设备）
python -B -m scenarios.tangram_puzzle.main `
  --target-image scenarios\tangram_puzzle\data\target_samples\target_mine.png `
  --raw          scenarios\tangram_puzzle\results\capture_check\exp312_gain0_wb3300_125104.png

# 4) 全部离线回归（不连相机、不连机器人）
python -B -m unittest `
  scenarios.tangram_puzzle.test_planning `
  scenarios.tangram_puzzle.test_stage3 `
  scenarios.tangram_puzzle.test_solver `
  scenarios.tangram_puzzle.test_roi_frame `
  scenarios.tangram_puzzle.test_target
```

注意：`data/raw_samples/raw_20261031b.png` 的绿块手性与目标图相反，会在几何门禁处
被拦下（`suspected_flip=True`）。它用于记录"绿块需要物理翻面"，不是成功基准；
成功基准是上面的 `exp312_gain0_wb3300_125104.png`。详见
[现场操作手册](docs/现场操作手册.md)。

---

## 真实机器人怎么用

**前置条件（缺一不可）**：机器人已上电并处于安全状态、相机链路可达
（优先 USB3；当前 USB2 降级 @15 fps 单彩色流）、现场光照与颜色阈值已整定、
已确认 `CAPTURE_POSE` 未被改动。

```powershell
# 1) 只读体检：读机器人状态 + 抓一帧 + 检测，不发送任何运动指令
python -B -m scenarios.tangram_puzzle.health_check

# 2) 检查取景框与颜色（只读）
python -B -m scenarios.tangram_puzzle.tools.roi_frame --image <原始帧>
python -B -m scenarios.tangram_puzzle.tools.diag_colors --image <原始帧> --area raw

# 3) 离线确认 7 条计划
python -B -m scenarios.tangram_puzzle.tools.target_check --target <目标图> --raw <原始帧>

# 4) 实机执行（会回 HOME、去拍照位、松吸盘、逐块抓放）
python -B -m scenarios.tangram_puzzle.main --execute
```

`--execute` 是**真实运动**入口，不是"只验证可达性"。执行前请完整阅读
[现场操作手册](docs/现场操作手册.md)。本仓库的自动化工具不会主动连接机器人，
除非你显式运行 `main --execute` 或 `health_check` 等设备入口。

---

## 仿真怎么用

本仓库**没有** MuJoCo / Unity 仿真代码，也没有可运行的仿真场景文件。
仿真侧需要与本仓库对齐的接口、坐标系、相机内参、吸盘长度、目标图与颜色编码，
全部整理在 [仿真对接说明](docs/仿真对接说明.md)。

如果仿真程序已经存在，推荐按以下方式接入，而不是复制本仓库的视觉/规划代码：

- 模拟 `common.realsense_camera.RealSenseCamera`，返回 BGR `numpy` 数组；
- 模拟 `common.robot_client.CobotClient` 的最小接口（见对接文档），
  或在 `main.py` 用工厂函数切换真机/仿真客户端；
- 目标图直接读取图片文件，不为仿真单独写一套目标格式。

> 仿真状态（是否已跑通、是否录过视频）本仓库无法确认，**待队友回填**。

---

## 目录结构

```text
20261015/
├─ README.md                     本文件：项目入口与导航
├─ AGENTS.md                     对话记录与文档维护约定
├─ DeepSeek.md                   代码评审与问题清单（D01–D15）
├─ 备赛对话记录.md                按轮追加的项目日志、问题台账与时间线
├─ requirements.txt              运行依赖
├─ common/                       设备与标定层（跨场景）
│  ├─ calibration.py             去畸变、单应映射、手动 9 点标定辅助
│  ├─ realsense_camera.py        D435i 采集（当前 USB2 单彩色流；可选深度对齐）
│  ├─ robot_client.py            CobotClient：SDK 版本自动匹配、IK 预检、抓放
│  ├─ gripper.py                 RobotDOGripper：控制箱 DO 驱动吸盘
│  └─ robot_sdk/                 厂商 SDK v3.8.2 / v3.9.3（未改动）
├─ scenarios/tangram_puzzle/     七巧板场景
│  ├─ main.py                    入口：离线规划 / --execute 实机
│  ├─ config.py                  唯一参数入口（ROI、阈值、点位、标定）
│  ├─ tangram_detector.py        HSV 分割 + 形状/面积仲裁 + 多边形拟合
│  ├─ puzzle_planner.py          ID 匹配、抓放位姿、偏心补偿、预览
│  ├─ planning/                  目标图解析、图案→台面、尺度归一、Q/P 指派
│  ├─ tangram/                   纯 NumPy 几何、件库、格点求解、自检
│  ├─ vision/silhouette.py       白色剪影提取
│  ├─ tools/                     标定、ROI、颜色、尺寸、求解等只读工具
│  ├─ data/                      测试与基准输入（含 legacy/ 历史样本）
│  └─ results/                   运行输出（不作为源代码依赖）
├─ docs/                         技术文档（见下表）
├─ external_refs/                第三方求解器快照、对比数据与诊断证据
└─ archive/                      历史备份（旧代码/旧文档），不参与运行
```

---

## 文档导航

| 文档 | 记录什么 | 适合谁 |
|---|---|---|
| [项目现状与路线](docs/项目现状与路线.md) | 哪些功能已验证/未验证、L0–L8 路线、门禁与优先级 | 所有人，先读 |
| [技术条件与精度链](docs/技术条件与精度链.md) | 赛题/硬件事实、七巧板几何、坐标与单位、标定与误差预算 | 做算法、标定、汇报的人 |
| [现场操作手册](docs/现场操作手册.md) | 到场后的相机、ROI、标定、体检、执行、回退与故障处理 | 上机操作的人 |
| [仿真对接说明](docs/仿真对接说明.md) | 给仿真侧的接口、坐标系、内参、数值与验收清单 | 仿真队友 |
| [场景说明](scenarios/tangram_puzzle/README.md) | 场景内部模块、命令、配置键与限制 | 改代码的人 |
| [代码评审与问题清单](DeepSeek.md) | D01–D15：问题、证据、状态、下一步 | 维护代码的人 |
| [备赛对话记录](备赛对话记录.md) | 按日期追加的需求、决定、检查结论与时间线 | 追溯"为什么这样做" |
| [汇报材料交接说明](docs/汇报材料交接说明-给报告负责人.docx) | 15 分钟汇报的叙事、素材与 Q&A | 报告/PPT 负责人 |

---

## 已知问题（摘要）

1. **实机精度未验收**：当前参数组合没有一次完整的 7 块实机落点数据；
   2026-10-28 的完整运行放置偏移大，根因未定量。
2. **标定绑定本机**：`HOMOGRAPHY`、内参、`CAPTURE_POSE`、ROI 都只对当前工位成立；
   一键九点已移除，换机后只能手动重做 9 点标定，否则程序会"照常跑完但每块都错"。
3. **高度链缺一个实测数**：桌面→法兰距离未用卷尺确认，吸盘长 252.5 mm 来自工具链模型。
4. **强光仍是已知未解项**：三档光照中强光档 5/7；镜面高光会把掩模咬掉。
5. **目标图余量偏紧**：当前 `target_mine.png` 在 180 mm 可用区内最小内距约 0.2 mm。
6. **无落点闭环**：放完不复拍，偏差无法逐块量化。
7. **进阶无主流程入口**：求解、pattern、assign 已有，但 `main` 只走彩色/照片通道。

详细状态、证据与优先级见 [项目现状与路线](docs/项目现状与路线.md) 与
[DeepSeek.md](DeepSeek.md)。

---

## 数据与证据边界

- `scenarios/tangram_puzzle/data/stage3_20261025/` 是回归测试夹具，**不要删**；
- `data/glare_20261026/` 是三档光照实验数据；
- `data/raw_samples/` 是原料区基准帧；`data/target_samples/` 是目标图样本；
- `results/` 是运行输出，可重新生成；仅保留被文档引用的关键证据；
- `external_refs/` 含第三方代码快照与对比结果，**不是运行依赖**；
- `archive/` 是历史备份，不参与运行。

第三方参考项目的许可证与使用边界见 [external_refs/README.md](external_refs/README.md)。

---

## 安全与授权

- 本仓库的离线命令不会连接机器人；`health_check` 只读状态；`--execute` 会产生**真实运动**。
- 未经明确授权，不要运行 `main --execute` 或任何会驱动夹爪/吸盘的动作。
- 相机标定文件、内参、ROI、点位参数属于本机关键数据，修改前先备份并在文档中记录。

## 新读者导读

[猫娘读物](docs/猫娘读物.md)：按当前代码整理的项目结构、使用入口、坐标与规划原理、参数及结果文件说明。

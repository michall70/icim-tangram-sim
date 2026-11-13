# 接口对照：你们的代码 ↔ 仿真

目标：**你们一行代码都不改**。做法是在导入期把硬件类换掉，其余照旧。
入口在 `icim_sim/devices.py::bootstrap_env()`。

## 0. 你们的目录结构不需要固定

仿真**不假设** `common/`、`scenarios/` 这些名字——`20261030` 也只是你们某次快照的日期。
`bootstrap_env()` 会扫描 `--team-repo` 指定的目录（不给就在 `third_party/` 下自动挑一个），
按**文件内容**认角色：

| 角色 | 认的标志 |
|---|---|
| `robot_client` | 有 `class CobotClient` |
| `camera` | 有 `class RealSenseCamera` |
| `calibration` | 有 `class CameraMapper`（可选，只有 dry-run 用） |
| `config` | 有全局 `GRIP_Z` + `CAMERA_MATRIX` |
| `main` | 文件名叫 `main.py` 且定义了 `run(...)` |
| `detector` | 有 `class TangramDetector` |
| `planner` | 有 `def plan_puzzle` |

跑起来会先打印一张表告诉你们认到了什么：

```
[team] 队友代码：.../third_party/20261030
    calibration  common.calibration
    camera       common.realsense_camera
    config       scenarios.tangram_puzzle.config
    detector     scenarios.tangram_puzzle.tangram_detector
    main         scenarios.tangram_puzzle.main
    planner      scenarios.tangram_puzzle.puzzle_planner
    robot_client common.robot_client
```

认错了/认不全，就在仓库根目录放一个 `team_layout.json` 手工指定（角色 -> 模块名）：

```json
{"robot_client": "utils.robot", "camera": "utils.camera",
 "config": "app.cfg", "main": "app.main"}
```

所以下一版快照改名字、或者把 `common/` 并进别处，都不用动仿真。

## 1. 被替换的东西

| 你们 import 的 | 仿真里换成 | 说明 |
|---|---|---|
| `...robot_client.CobotClient` | `icim_sim.robot.SimCobotClient` | 方法名/参数/返回值/异常语义一致 |
| `...realsense_camera.RealSenseCamera` | `icim_sim.camera.SimCamera` | 只实现 `open/close/grab_color/get_aligned_frames` |
| `pyrealsense2` | 空壳模块 | 只为让 import 过；真路径已被上面顶替 |
| `opcua` | 空壳模块 | 同上（`RobotDOGripper` 走 `rpc.SetDO`，不需要 OPC UA） |
| `RobotDOGripper` | **不动** | `rpc.SetDO(0,1)` 在仿真里就是"吸住" |

**为什么不用改 import**：Python 的 `from X import Y` 是运行时去模块对象上取属性。
我们先改 `X.Y`，再导入你们的 `main`，所以你们那行 import 取到的就是替身。
顺序在 `devices.py` 里写死了：加路径 → 换 CobotClient → 导 config → 换 RealSenseCamera → 导 main。

相机在 `camera.open()` 时**不要求机器人已连接**——你们的 `main.py` 是先 `camera.open()`
再 `robot.connect()`，所以真正绑定发生在第一次取图。

## 2. `CobotClient` 支持的方法

| 方法 | 仿真行为 |
|---|---|
| `connect()` / `disconnect()` | 加载 `assets/scene.xml`；断开后**保留**最终状态（还等着读结果） |
| `check_coordinate_context()` | 支持 tool 0 / user 0，其它直接报错 |
| `move_home(joint, vel)` | MoveJ 到给定关节角 |
| `move_to(pose, vel)` | 逆解 + MoveJ；`pose=[x,y,z,rx,ry,rz]`，mm/度，原点是**法兰面中心** |
| `move_above(pose, dz, vel)` | 先抬高再走 |
| `pick(pose, gripper, approach_dz, vel)` | 上方接近 → 到位 → `gripper.grip()` → 抬升 |
| `place(pose, gripper, approach_dz, vel)` | 上方接近 → 到位 → `gripper.release()` → 抬升 |
| `preflight_pick_place(plans, dz)` | 逐点逆解，不动机器人；返回每个端点的关节解 |
| `rpc.GetActualJointPosDegree()` | 当前六个关节角（度），与 `qpos` 一一对应 |
| `rpc.SetDO(i, v)` / `GetDO()` | DO0 是吸盘：1=吸、0=放 |

**没有**实现：`MoveL`（直线）、`MoveC`、速度倍率覆写等。用到会抛 `RuntimeError` 说明，
而不是静默给出错误结果。

## 3. 会被替换的常量

`run.py` 在调 `main.run()` 之前把下面这些**只对本工位成立**的数换成模型算出来的值，
并打印改动前后（`apply_overlay()`）：

> ⚠️ 他们的 `main.py` 是 `from ...config import 常量`，所以同一个名字**在 `config`
> 和 `main` 两个模块上各有一份**。只改 `config` 不生效——`apply_overlay` 两处都改。

| 常量 | 含义 | 仿真里怎么算 |
|---|---|---|
| `HOMOGRAPHY` | 像素 → 基座 xy | 由模型相机的真实投影求逆（板面高度上） |
| `CAPTURE_POSE` | 拍照位 | 读包根目录 `capture_pose.json` 里固定的位姿（托盘和相机没动就不用重算）；文件缺失才现场搜 |
| `GRIP_Z` / `PLACE_Z` | 抓放法兰 z | 板面高度 + 法兰→吸盘口长度（+2 mm 放置余量） |
| `TCP_RPY` | 抓放姿态 | 吸盘轴竖直；绕竖直轴的转角由 `run.py` 挑一个"逆解到得了"的（**不能随便换**：会改变板子在吸盘里的相对朝向，实测差 15 mm；也**不能逐块换**，见 §7.2） |
| `HOME_JOINT` | 收尾/起始姿态 | 改成**拍照位对应的关节**（他们示教的那个在本模型里臂会穿过自己） |
| `DETECT_PARAMS['raw_roi']` | 原料区像素框 | **原料托盘**的投影外接框再外扩 6 mm（不是"这一摞板子的外接框"——那样换个布局就得重标） |
| `DETECT_PARAMS['template_roi']` | 成品区像素框 | 以目标托盘中心为心、边长 = **托盘垫片**的正方形投影（2026-11-02 起垫片 260 mm） |
| `TARGET_SOURCE` | 目标来源 | 固定 `"image"`（`--target-source photo` 可回退到旧链路） |
| `TARGET_IMAGE_PATH` | 目标图案图片 | `--target` 指定；默认 `runs/target_sim.png`（仿真按模型尺度画的经典正方形） |
| `TARGET_ANCHOR_MM` | 图案形心落在台面哪 | 目标托盘中心（基座系 mm）→ 不再是 `None` |
| `TARGET_AREA_MM` | "放不下"的判据 | 托盘垫片边长，260×260 mm |
| `S_PHYS_MM` | 图案缩放的实物尺度 | 七块面积和开根 = **116.364 mm**（他们的 115.84 是件套公差，差 0.45%） |
| `TEMPLATE_PATH` | （photo 通道才用）相机模板照片 | `--template` 指定；image 通道下不读 |
| `RESULTS_DIR` | 结果输出目录 | 改到 `runs/results/`，**不写你们的仓库** |

你们的算法（检测、匹配、规划、异常处理）**原样运行**，仿真不介入。

## 4. 吸盘模型（唯一"物理判据"的地方）

`rpc.SetDO(0, 1)` 时，同时满足三条才吸住：

| 判据 | 阈值 | 可调 |
|---|---|---|
| 吸盘口在板子顶面轮廓内（凸包内） | 硬性 | — |
| 吸盘口离板面（正数=还有缝） | ≤ 8 mm，压进 ≤ 4 mm | `robot.suction_max_gap_mm` / `suction_max_press_mm` |
| 工具轴与板面法向夹角 | ≤ 20° | `robot.suction_max_tilt_deg` |

不满足就返回 `False` 并打印原因（`还差 12.1 mm 才碰到板面`）。
吸住之后板子由每帧的运动学跟随（刚性吸附），放开即自由下落。

## 5. 初始布局（七块板 + 关节角）

```python
from icim_sim import layout

layout.list_layouts()            # layouts/ 下所有 .json
layout.choose()                  # 按 命令行/active.txt/default/唯一 的规则挑一份
layout.read(path)                # 读成 dict
layout.apply(robot.m, robot.d, data)      # 套到场景上（板子 + 关节）
layout.apply_to(robot, 我的)    # 直接按名字套用
layout.save_from(robot, 我的)   # 把当前状态存成 layouts/我的.json
```

命令行（`run.py` / `play.py` / `demo.py` 都支持）：

    --layout 名字     用哪份（auto=按规则挑，none=保持场景原样）
    --list-layouts    列出现有布局

`SimCobotClient(layout="auto")` 会在 `connect()` 里套用默认布局——所以队友的
`main.run()` 一取图，板子已经在布局位置上。

## 6. 多份队友代码

```
third_party/
  20261101/       每份都是一整个仓库，名字随便
  20261030/
  active.txt      一行：默认用哪份
```

    python run.py --list
    python run.py --team-repo 20261101
    python run.py --set-active 20261101

规则：显式指定 → active.txt → 唯一候选 → 报错列清单。

## 7. 目标图案通道（`TARGET_SOURCE="image"`）与两条踩过的坑

### 7.1 谁负责什么

```
目标图（赛方给的一张彩色图）
  → 他们的 planning/target_image.py：按颜色聚类切七块 → 3/4 边形 → 按面积分类 L/M/T/Q/P
  → 归一到 S=1 → 乘 S_PHYS_MM → 平移到 TARGET_ANCHOR_MM → 反投影回像素 → plan_puzzle
```

仿真这边只提供三样：**图**（`--target`）、**锚点**（目标托盘中心）、**尺度与可用区**
（`S_PHYS_MM` / `TARGET_AREA_MM`）。图案怎么摆、怎么匹配、怎么排顺序，全是他们的。

### 7.2 坑一：抓取朝向不能逐块改（我们自己的 bug，已修）

`pick()` 里曾经有个 `_free_yaw()`：抓取时按 90° 整数倍换法兰朝向，理由是"吸盘是圆的，
怎么转都吸得住，物理等价"。**不成立**：

    板子的净转角 = 放置 rz − 抓取 rz

抓取朝向改 90k°，板子最后就少转 90k°——位置不变、朝向错。实测（用 `plans.json`
按执行顺序复现）：M1/T1/T2 各少 90°，P1 少 270°，成品是一堆歪掉的块。
想省 J6 只能**全局**选一个抓取朝向（`run.pick_grasp_yaw()`，所有块同一个角）。

### 7.3 坑二：平行四边形有**两条**手性判据，口径不同

| 判据 | 在哪算 | 报什么 | 我们怎么处理 |
|---|---|---|---|
| `assign.handedness_mismatch()` | **像素**空间（检测到的顶点 vs y 向上的件库） | `AssignError`，整批中止 | `icim_sim/chirality.py::check()` |
| `plan_puzzle.suspected_flip` | **毫米**空间（块轮廓 vs 模板多边形的最小二乘） | `ValueError`（`--execute` 才拦） | `chirality.pattern_handedness()` |

两条判据互相独立，**经常一条说没事、另一条说要翻**：拿他们 `data/target_samples/`
三张图实测，都是"assign 正常、plan 说要翻"（拟合 12.2–12.5 mm / 镜像 1.1–1.3 mm）。
所以 `run.py` 两条都算，任一说要翻就在开局把那块板翻过来（七巧板是等厚棱柱，
翻面后外形是镜像、落回托盘还是平的，物理上合法）。

判据本身不重写：`pattern_handedness()` 会把模型里这块的真轮廓造一个 `TangramPiece`，
**调一次他们的 `plan_puzzle`**（只喂这一块），看它返回的 `suspected_flip`。

### 7.4 已知差异：图案会上下镜像（他们那边的口径，我们不改）

`target_image.normalize()` 交出来的还是像素坐标（v 向下），`pattern.transform_polygon()`
只做缩放/旋转/平移、**不翻 y**，所以落到台面的图案是图片的上下镜像。正方形看不出来，
不对称的图形会反着拼。细节和证据在 `docs/对接问题清单_1102.md` §4。
仿真侧的落点比对因此**对着 `plans.json` 的 place 点**，不跟图案 JSON 比（那才叫
"摆放 vs 规划"），见 `run.report_final()`。

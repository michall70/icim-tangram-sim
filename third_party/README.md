# third_party —— 队友的代码放这里

这个目录**可以有任意多份**，每个子目录都是一份完整的队友仓库（一次快照）：

```
third_party/
  20261030/          ← 当前这份（交付包里自带，只含 .py 代码）
  20261030/          ← 10 月 30 日那份（你自己拷进来的）
  active.txt         ← 一行：默认用哪份（可选）
```

## 用哪份，优先级从上到下

1. 命令行指定：`python run.py --team-repo 20261030`（写目录名或绝对路径都行）
2. `third_party/active.txt` 里写的一行目录名
3. 只有一份候选 → 就用它
4. 多份且没指定 → **列出来让你选**，不猜
5. 一份都没有 → 报错，告诉你怎么放

看有哪些、认清了几项：

```powershell
python run.py --list
```

设默认份：

```powershell
python run.py --set-active 20261030
```

## 里面的目录结构不用迁就我们

仿真**不假设** `common/`、`scenarios/` 这些名字（`20261030` 也只是当初快照的日期）。
它会扫每份仓库里的 `.py`，按**内容**认角色：

| 角色 | 认的标志 |
|---|---|
| `robot_client` | 有 `class CobotClient` |
| `camera` | 有 `class RealSenseCamera` |
| `calibration` | 有 `class CameraMapper`（可选） |
| `config` | 有全局 `GRIP_Z` + `CAMERA_MATRIX` |
| `main` | 文件名叫 `main.py` 且定义了 `run(...)` |
| `detector` / `planner` | 有 `TangramDetector` / `def plan_puzzle` |

跑起来会先打印认到的结果。认错了就在那份仓库根目录放 `team_layout.json` 手工指定：

```json
{"robot_client": "utils.robot", "camera": "utils.camera",
 "config": "app.cfg", "main": "app.main"}
```

## 交付包自带的那份只含代码

`20261030/` 是作者放的**样例快照**，只保留 `common/`、`scenarios/`。
为了体积去掉了：

| 去掉了 | 为什么 |
|---|---|
| `data/`（真机拍的模板照片，7.7 MB） | 仿真用自己的渲染图当模板，`run.py` 会替换 `TEMPLATE_PATH` |
| `archive/` | 他们自己的历史备份 |
| `results/`、`__pycache__/` | 运行产物 |

所以这份**只够在仿真里跑**。要用他们自己的离线脚本（`detect_check.py` 之类）读 `data/`，
把完整仓库拷进来替换掉它即可。

## 换成你的新代码

把新版整个拷成 `third_party/<新名字>/`，然后：

```powershell
python run.py --list                  # 确认认到了 7 项
python run.py --team-repo <新名字>     # 用它跑
python run.py --set-active <新名字>    # 想把它设成默认
```

旧的那份留着不碍事，仿真只读选中的那一份。

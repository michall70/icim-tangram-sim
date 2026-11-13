# 交付怎么做（作者侧）

这个包是给队友用的**运行环境**，不是我的工作间。工作间在 `../icim-tangram-workshop`，
负责从 STEP 生成场景和网格；它不再保存运行时代码。

## 一、要发什么

```
icim_sim/            ← 发这个文件夹（或者打成的 zip）
```

**不发**：`.venv`、`runs/`、`.git`、`__pycache__`。
**要发**：`third_party/` 下**当前那一份快照整份带上**（含 `data/` 里的样例图、
`results/` 里的历史图，队友自己认路）；旧快照不带（`tools/deliver.py` 里的
`SHIP_SNAPSHOT` 决定是哪一份，现在是 `20261101`）。
唯一不带的是他们仓库里的 `.git`（两百多 MB 历史，打进去只会让嵌套仓库更难解压）。

| 不发 | 为什么 |
|---|---|
| `.venv` | venv 里写死了绝对路径和 Python 版本，拷到别人机器上大概率起不来；他们自己建一个更可靠 |
| `runs/` | 是运行产物（图片、录像、规划结果），他们跑一次就有了 |
| `third_party/<旧快照>/` | 只保留一份，避免"到底用哪份"的歧义；`active.txt` 指的就是它 |
| `third_party/20261101/.../.git` | 他们的提交历史，四百 MB 里占两百；解压后还会变成嵌套仓库 |
| `.git`（本包） | 可选，加了体积大，一般没必要 |

## 二、打包命令（在工作间里跑）

```powershell
cd E:\Michall\Projects\Robotic_arm
.venv\Scripts\python.exe tools\deliver.py --verify
```

出来 `..\dist\icim_sim_v<版本>.zip`（0.2.0 这版约 **226 MB / 878 个文件**——大头是
`third_party/20261101` 整份快照 200 MB + `assets/` 的网格 44 MB）。脚本会打印打进去多少文件、
最大的是哪几个，方便确认没有混进 `runs/` 或 `third_party/`。

参数：

| 参数 | 作用 |
|---|---|
| `--verify` | 打包前先在包内跑 `verify.py`，不过就不打 |
| `--with-wheels` | 顺便 `pip download` 依赖，打进去后可以**离线安装** |
| `--with-git` | 连 `.git` 一起打（想让对方看到提交历史时用） |
| `--out DIR` | 换输出目录 |

## 三、他们那边怎么装

有网：

```powershell
cd icim_sim
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

没网（你打了 `--with-wheels`）：

```powershell
.venv\Scripts\python.exe -m pip install --no-index --find-links wheels -r requirements.txt
```

装完先跑三条确认仿真活着（不需要他们的代码）：

```powershell
.venv\Scripts\python.exe verify.py     # 24/24
.venv\Scripts\python.exe demo.py       # 0.00 mm
.venv\Scripts\python.exe play.py       # 手玩
```

## 四、以后更新

| 改了什么 | 怎么更新 |
|---|---|
| 场景/网格/初始姿态/托盘材质 | 在工作间跑 `build_scene.py` → `publish_assets.py`，再打一次包 |
| 仿真的代码（相机、吸盘、标定） | 直接在 `icim_sim/` 里改，改完 `verify.py` 必须还是 24/24 |
| 队友换了新代码 | **不用动包**，覆盖 `third_party/` 下的目录即可（按内容自动识别） |

## 五、作者侧的检查清单

- [ ] `tools/publish_assets.py --check` 报"已一致"（场景和网格同步）
- [ ] `icim_sim` 里 `verify.py` 24/24、`demo.py` < 1 mm
- [ ] `runs/` 和 `third_party/` 没有混进 zip（看 `deliver.py` 打印的清单）
- [ ] `third_party/README.md` 还在（告诉他们代码放哪）
- [ ] 版本号：改 `icim_sim/__init__.py` 里的 `__version__`

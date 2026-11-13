# 场景数据目录说明

本目录区分**测试夹具**、**基准输入**、**实验数据**与**历史样本**。
程序运行时只读取 `config.py` 指向的具体文件；其余用于回归、复现与对照。

| 路径 | 用途 | 能否删除 |
|---|---|---|
| `stage3_20261025/` | 回归测试夹具：`1_Color.png`、`template_Color_Color.png`、`blank_Color.png` | **不能**，6 个测试模块中的两个直接读它 |
| `stage3_20261029b/` | 照片模板回退通道的当前模板图 | 不能（`TEMPLATE_PATH` 指向） |
| `glare_20261026/` | 三档光照实验（偏暗/正常/强光），用于复现 5/7–7/7–5/7 | 建议保留 |
| `raw_samples/` | 原料区基准帧；使用规则见该目录 `README.md` | 保留 |
| `target_samples/` | 目标图样本：`target_mine.png`（当前默认）、`target_square.png`、`target_triangle.png` | 保留 |
| `silhouette_tests/` | 剪影实验集：20 张、5 张、V 形与答案 JSON | 建议保留 |
| `silhouette_sample.png` | 单张剪影样本 | 保留 |
| `legacy/` | 历史模板与旧工位样本，已被 `stage3_20261029b` 等替代 | 可删；当前保留供追溯 |

## 命名规则

- 相机导出帧保留原始命名（如 `1_Color.png`、`00_Color.png`），便于与日志对照；
- 基准帧用 `raw_<日期>.png`，目标图用 `target_<描述>.png`；
- 新实验放独立子目录 `silhouette_tests/<日期>_<主题>/`，不要散落在根目录。

## 重要提醒

- `results/<时间戳>/raw_frame.png` 是**已去畸变**的图；`main --raw` 会再去畸变一次，
  不要直接喂给它。`target_check` 对这类图要加 `--already-undistorted`。
- `data/` 下不要放运行产生的中间图；运行输出统一进 `scenarios/tangram_puzzle/results/`。

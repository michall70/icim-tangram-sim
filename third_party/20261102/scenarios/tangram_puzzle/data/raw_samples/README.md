# 原料区实拍帧放这里

## 为什么要单独放

`main` 与 `tools/target_check` 的 `--raw` 参数吃的是**相机原始帧**（带镜头畸变），
程序内部会自己做一次去畸变。所以：

| 图片来源 | 是不是原始帧 | 怎么用 |
|---|---|---|
| `capture_check` 按 `s` 存的图 | 是 | 直接当 `--raw` |
| `results/<时间戳>/raw_frame.png` | 否，已去畸变 | `target_check` 要加 `--already-undistorted`；`main` 没有这个开关，**别用** |

## 当前基准帧

| 文件 | 说明 | 能否直接通过 |
|---|---|---|
| `raw_20261031.png` | 2026-10-31 首次 7/7 的帧（旧 ROI 下） | 检测 7/7；与当前目标图搭配时注意手性 |
| `raw_20261031b.png` | 曝光 312、重画 ROI 后的帧 | 检测 7/7，但**绿块手性与目标图相反**，几何门禁会拦下 |
| `results/capture_check/exp312_gain0_wb3300_125104.png` | 用户自测帧，当前 `target_check` 通过 | **是**（7 条计划、无翻面） |

> `raw_20261031b.png` 不是失败样本，而是"绿块需要物理翻面"的证据。
> 不要把它的结果当成"代码坏了"。

## 怎么拍新帧

1. 跑 `python -B -m scenarios.tangram_puzzle.capture_check`
2. 七块全部摆进原料框（离框边 ≥15 px），ROI 内 V 中位数调到 60—80
3. 按 `s` 保存，文件落在 `results/capture_check/exp<曝光>_gain<增益>_wb<白平衡>_<时间戳>.png`
4. 复制到本目录并改名，例如 `raw_20261031c.png`

## 怎么验证这张图

```powershell
# 目标图 + 原料帧 → 七条计划
python -B -m scenarios.tangram_puzzle.tools.target_check `
  --target scenarios\tangram_puzzle\data\target_samples\target_mine.png `
  --raw    scenarios\tangram_puzzle\data\raw_samples\<新帧>.png

# 走主流程
python -B -m scenarios.tangram_puzzle.main `
  --target-image scenarios\tangram_puzzle\data\target_samples\target_mine.png `
  --raw          scenarios\tangram_puzzle\data\raw_samples\<新帧>.png
```

通过判据：原料 7/7、`[assign]` 七个映射且手性正常、7 条计划、落点全在成品区内、
每块拟合 <5 mm、无 `suspected_flip`。

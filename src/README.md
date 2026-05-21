# `src/` 目录说明

`src/` 现在不再承担 `iid VCTR` 的 Python 复现任务，而是明确转向 paired-eye VCTR 的主开发线。

## 当前定位

- 主线目标：实现双眼 paired 的 VCTR。
- 默认参考：`VaryingCoefPLM.pdf`。
- 不再兼容：旧的 `iid VCTR` Python 主线。

旧的 `iid VCTR` Python 移植代码已归档到：

- `archive/python_iid_vctr/`

该归档用于：

- 保留历史移植痕迹；
- 方便后续与 MATLAB `iid` 基线核对；
- 避免和 paired 主线混在一起。

## 当前目录结构

```text
src/
  data/
  features/
  metrics/
  models/
  experiments/
  utils/
```

各模块职责：

- `data/`
  - paired-eye 数据容器。
  - 默认按 subject-level 与 eye-level 两层组织数据。

- `features/`
  - paired 主线的张量分块与投影特征构造接口。
  - 当前只保留骨架，等待 paired 仿真和真实数据流程定型。

- `metrics/`
  - paired 主线仍可复用的误差指标与结构识别指标。

- `models/`
  - paired-eye VCTR 的核心模型接口与实现位置。
  - 当前只保留 `PairedEyeVCTRModel` 骨架。

- `experiments/`
  - paired 仿真与 paired 真实数据分析入口。
  - 当前为占位入口，尚未接通完整主链。

- `utils/`
  - 从旧 `iid` 主线中提炼出来的通用数值工具。
  - 当前保留 kernel、spline、penalty 相关函数。

## 当前结论

`src/` 现在的任务不是“继续完善 iid 复现”，而是为 paired-eye VCTR 建立清晰、单一、可扩展的主开发线。

# `hpc/` 目录说明

这个目录存放 NUS HPC 相关说明文件，以及历史 PBS 脚本归档。

## 文件列表

### `archive/`

- 历史 PBS 脚本归档目录
- 当前包含：
  - `archive/paired_case2_reduced24_5parts.pbs`
  - `archive/paired_case2_full_parallel.pbs`
- 这两个脚本由于对应旧的模型/DGP 阶段，现统一作为历史参考保留

### `README.md`

- 本目录说明文件
- 只负责介绍 `hpc/` 目录当前结构与定位

## 当前定位

- `hpc/` 现在是仓库顶层的 HPC 入口目录
- 如果后续有新的正式 PBS 模板，建议直接放在 `hpc/` 下
- 旧模板统一放在 `hpc/archive/`，避免与现行脚本混淆

## 使用建议

- 如果你只是想了解 HPC 目录结构，先看本文件
- 如果你需要回看旧的批处理写法或较早期的 Case 2 方案，查看 `hpc/archive/`

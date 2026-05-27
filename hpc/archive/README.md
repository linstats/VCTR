# `hpc/archive/` 目录说明

这个目录存放历史 HPC PBS 脚本模板。

当前归档内容：

- `paired_case2_reduced24_5parts.pbs`
  - 旧的 24 核、5 份拆分 Case 2 提交模板
- `paired_case2_full_parallel.pbs`
  - 更早期的 full parallel Case 2 提交模板

这些脚本保留的目的主要是：

- 作为旧实验提交方式的记录
- 便于回看此前在 NUS HPC 上的参数组织和 conda 激活写法

由于当前 paired 模型与 DGP 已重新调整，这两个脚本默认都不再视为现行生产模板。

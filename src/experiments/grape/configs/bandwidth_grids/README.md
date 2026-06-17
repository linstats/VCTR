# GRAPE bandwidth grid 配置

本目录保存 GRAPE 实证 bandwidth 设计。每次重新设计 `h` / `hbar` 时，新增一个 JSON 文件，不覆盖旧版本。

当前第一版：

- `v1_adaptive_support.json`

该版本先给出较宽的候选 `signal h`，再按 `(S,R)` 的局部线性参数量和 5-fold 训练 support 自动过滤过小的 `h`。

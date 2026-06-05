# `hpc/` 目录说明

这个目录存放 NUS HPC 相关说明文件，以及历史 PBS 脚本归档。

## NUS HPC 环境

- 可靠直接入口：
```bash
ssh e0829076@atlas9.nus.edu.sg
```
- 登录后 home 目录为 `/home/svu/e0829076`。
- HPC 上的仓库副本为 `/home/svu/e0829076/2026-tensor`。
- `hopper.nus.edu.sg` 曾从当前网络路径超时；本项目优先使用 `atlas9.nus.edu.sg`。
- `hpcportal.nus.edu.sg` 也曾不稳定；除非明确需要 portal，否则优先走直接 SSH。

## Python 环境

本项目不依赖集群默认 Python。当前可用的用户 conda 环境为：

```bash
/home/svu/e0829076/conda-envs/vctr-py310
```

交互式激活方式：

```bash
module purge
module load miniconda/4.12
source activate $HOME/conda-envs/vctr-py310
```

这个环境已验证可以运行：

```bash
python src/experiments/paired_case1_altbase_repetition.py --help
```

PBS 脚本不要假设继承交互 shell 环境；每个批处理脚本都应显式执行 `module purge`、`module load miniconda/4.12` 和 `source activate $HOME/conda-envs/vctr-py310`。

## 队列与资源

已确认的队列约束：

- `serial`: 1 CPU，内存范围约 `2gb` 到 `15gb`。
- `parallel`: 最少 `12` CPUs，最多 `96` CPUs，最大内存约 `540gb`。

实际使用建议：

- full paired repetition run 不要用 `serial`。
- 多进程 Python 作业提交到 `parallel` 时，通常请求 `12` CPUs 并设置 `--n-jobs 12`。
- 如果并发提交超过实际 CPU 上限，PBS 会自动排队。
- `ridge = 1e-4` 可作为数值稳定化选择；论文默认公式仍应描述为 `ridge = 0`。

## 文件列表

### `archive/`

- 历史 PBS 脚本归档目录
- 当前包含：
  - `archive/paired_case2_reduced24_5parts.pbs`
  - `archive/paired_case2_full_parallel.pbs`
- 这两个脚本由于对应旧的模型/DGP 阶段，现统一作为历史参考保留

### `paired_case1_altbase_varsigma_parallel.pbs`

- 当前正式的 `paired_case1_altbase` varying-sigma HPC 模板
- 直接放在 `hpc/` 根目录
- 默认资源：
  - `parallel`
  - `12` CPUs
  - `24gb` memory
  - `36h` walltime
- 默认实验：
  - `n_subject = 1000 2000`
  - `coef_types = base1 base2 base3 base4`
  - `rho_values = 0.0 0.3 0.6 0.9`
  - `R = 4`, `S = 25`, `p0 = 4`
  - `beta = 2.0,1.0,-1.0,0.5`
  - `covariance_mode = exchangeable_varying_sigma`
  - `signal_bandwidth = 0.18`
  - `variance_bandwidth = 0.18`
  - `ridge = 1e-4`

### `paired_case1_altbase_varying_var_parallel.pbs`

- 当前 `paired_case1_altbase` time-varying DGP variance HPC 模板
- 结果默认写到：
  - `src/experiments/paired_case1_altbase_repetition/hpc_varying_var/<sigma2_function>/part<id>/`
- 默认资源：
  - `parallel`
  - `12` CPUs
  - `24gb` memory
  - `36h` walltime
- 默认实验配置沿用 `hpc_const_var/run_config.json`：
  - `n_subject = 1000 2000`
  - `coef_types = base1 base2 base3 base4`
  - `rho_values = 0.0 0.3 0.6 0.9`
  - `R = 4`, `S = 25`, `p0 = 4`
  - `beta = 2.0,1.0,-1.0,0.5`
  - `sigma2 = 1.0`
  - `covariance_mode = exchangeable_varying_sigma`
  - `signal_bandwidth = 0.18`
  - `variance_bandwidth = 0.18`
  - `ridge = 1e-4`
- 新增 DGP variance 参数：
  - `SIGMA2_FUNCTION = sin | sin2 | mixed`

### `submit_paired_case1_altbase_varying_var_8parts.sh`

- 提交 `paired_case1_altbase_varying_var_parallel.pbs` 的便捷脚本
- 默认提交 `3` 个 `sigma2_function`，每个拆成 `8` 个 part
- 每个 part 使用 `12` workers
- `30` 次 repetition 拆分为：
  - `4,4,4,4,4,4,3,3`
- seed 覆盖：
  - part 1: `123-126`
  - part 2: `127-130`
  - part 3: `131-134`
  - part 4: `135-138`
  - part 5: `139-142`
  - part 6: `143-146`
  - part 7: `147-149`
  - part 8: `150-152`
- 可先用 dry run 核实命令：
```bash
DRY_RUN=1 bash hpc/submit_paired_case1_altbase_varying_var_8parts.sh
```

### `paired_case1_altbase_base56_allsigma_parallel.pbs`

- 当前 `paired_case1_altbase` A5/A6 all-sigma HPC 补跑模板
- 结果默认写到：
  - `src/experiments/paired_case1_altbase_repetition/hpc_base56_allsigma/part<id>/`
- 默认资源：
  - `parallel`
  - `12` CPUs
  - `12gb` memory
  - `40h` walltime
- 默认实验配置：
  - `n_subject = 1000 2000`
  - `coef_types = base5 base6`
  - `sigma2_functions = constant sin sin2 mixed`
  - `rho_values = 0.0 0.3 0.6 0.9`
  - `R = 4`, `S = 25`, `p0 = 4`
  - `beta = 2.0,1.0,-1.0,0.5`
  - `sigma2 = 1.0`
  - `covariance_mode = exchangeable_varying_sigma`
  - `signal_bandwidth = 0.18`
  - `variance_bandwidth = 0.18`
  - `ridge = 1e-4`
- 这套 `8` part 任务现已完成；本地合并结果目录为：
  - `src/experiments/paired_case1_altbase_repetition/hpc_base56_allsigma/`

### `submit_paired_case1_altbase_base56_allsigma_8parts.sh`

- 提交 `paired_case1_altbase_base56_allsigma_parallel.pbs` 的便捷脚本
- 默认补跑 `base5 base6` 和 `constant/sin/sin2/mixed` 四种 DGP variance
- `30` 次 repetition 拆分为：
  - `4,4,4,4,4,4,3,3`
- seed 覆盖：
  - part 1: `123-126`
  - part 2: `127-130`
  - part 3: `131-134`
  - part 4: `135-138`
  - part 5: `139-142`
  - part 6: `143-146`
  - part 7: `147-149`
  - part 8: `150-152`
- 可先用 dry run 核实命令：
```bash
DRY_RUN=1 bash hpc/submit_paired_case1_altbase_base56_allsigma_8parts.sh
```

正式提交：

```bash
bash hpc/submit_paired_case1_altbase_base56_allsigma_8parts.sh
```

- 当前这批提交已完成，并已用于生成 Case 1 `A_5/A_6` 的合并结果与 LaTeX 表格。

### `paired_case2_altbase_varsigma_parallel.pbs`

- 当前正式的 `paired_case2_altbase` varying-sigma HPC 模板
- 直接放在 `hpc/` 根目录
- 默认资源：
  - `parallel`
  - `12` CPUs
  - `16gb` memory
  - `36h` walltime
- 默认实验：
  - `n_subject = 2000 5000`
  - `coef_types = base1 base2 base3 base4`
  - `rho_values = 0.0 0.3 0.6 0.9`
  - `R = 6`, `S = 27`, `p0 = 4`
  - `beta = 2.0,1.0,-1.0,0.5`
  - `covariance_mode = exchangeable_varying_sigma`
  - `signal_bandwidth = 0.20`
  - `variance_bandwidth = 0.20`
  - `ridge = 1e-4`
- 默认提交组织方式：
  - `10` parts
  - 每个 part `3` reps
  - `12` CPUs + `16gb` + `36h`

### `paired_case2_altbase_remaining_parallel.pbs`

- 当前用于 Case 2 剩余任务补齐的正式 PBS 模板
- 配合：
  - `hpc/submit_paired_case2_altbase_remaining_8parts.sh`
  - `src/experiments/paired_case2_altbase_repetition/build_remaining_manifests.py`
  - `src/experiments/paired_case2_altbase_repetition/paired_case2_altbase_manifest_run.py`
- 当前目标是只补齐：
  - A1-A4 的 `sin/sin2/mixed`
  - A5-A6 的 `constant/sin/sin2/mixed`
- 默认资源：
  - `parallel`
  - `12` CPUs
  - `16gb` memory
  - `72h` walltime
- 默认实验配置：
  - `n_subject = 2000 5000`
  - `rho_values = 0.0 0.3 0.6 0.9`
  - `R = 6`, `S = 27`, `p0 = 4`
  - `beta = 2.0,1.0,-1.0,0.5`
  - `sigma2 = 1.0`
  - `covariance_mode = exchangeable_varying_sigma`
  - `signal_bandwidth = 0.18`
  - `variance_bandwidth = 0.18`
  - `ridge = 1e-4`
  - `a_eval_mode = anchor_grid`
  - `a_eval_num_points = 500`
  - `a_eval_grid = quantile`
  - `a_interp = linear`
- 运行方式：
  - 每个 PBS job 通过 `MANIFEST_PATH` 读取一份精确任务清单
  - 不再按 `n_rep` 自动展开整块笛卡尔积

### `submit_paired_case2_altbase_remaining_8parts.sh`

- 当前用于提交 Case 2 剩余任务的便捷脚本
- 先本地生成 `8` 份 manifest，再提交 `8` 个 PBS jobs
- `8` 个 part 是严格等工作量分配，不是近似平均：
  - 每个 part `600` 个 fit 任务
  - 每个 part 含 `75` 个 `(coef_type, sigma2_function, seed)` bundle
  - 其中：
    - `45` 个 bundle 来自 A1-A4 varying-var
    - `30` 个 bundle 来自 A5-A6 all-sigma
- 默认输出：
  - manifest：`src/experiments/paired_case2_altbase_repetition/remaining_manifests_anchor500_h018_R6/`
  - run root：`src/experiments/paired_case2_altbase_repetition/hpc_case2_remaining_anchor500_h018_R6/`

### `paired_case2_altbase_backfill_parallel.pbs`

- `paired_case2_altbase` 精确补跑的 PBS 模板
- 配合：
  - `src/experiments/paired_case2_altbase_repetition/audit_case2_hpc_parts.py`
  - `src/experiments/paired_case2_altbase_repetition/paired_case2_altbase_backfill.py`
- 默认资源：
  - `parallel`
  - `12` CPUs
  - `16gb` memory
  - `72h` walltime
- 通过环境变量传入：
  - `MANIFEST_PATH`
  - `RUN_NAME`
  - `N_JOBS`
- manifest 每行一个缺失任务，字段固定为：
  - `part,n_subject,coef_type,rho_true,rep,seed`

### `README.md`

- 本目录说明文件
- 只负责介绍 `hpc/` 目录当前结构与定位

## 当前定位

- `hpc/` 现在是仓库顶层的 HPC 入口目录
- 如果后续有新的正式 PBS 模板，建议直接放在 `hpc/` 下
- 旧模板统一放在 `hpc/archive/`，避免与现行脚本混淆

## 常用监控命令

队列查看：

```bash
qstat -Q
qstat -Qf serial
qstat -Qf parallel
qstat -u e0829076
qstat -f <job_id>
qstat -fx <job_id>
```

Case 1 repetition 输出监控：

```bash
wc -l ~/2026-tensor/src/experiments/paired_case1_altbase_repetition/<run_name>/results/raw_results.csv
tail -n 5 ~/2026-tensor/src/experiments/paired_case1_altbase_repetition/<run_name>/results/raw_results.csv
find ~/2026-tensor/src/experiments/paired_case1_altbase_repetition/<run_name> -maxdepth 2 -type f | sort
```

运行状态解释：

- `run_config.json` 和 `results/raw_results.csv` 通常会较早创建。
- `progress.json` 要等至少一个 fit 完成后才出现。
- `raw_results.csv` 只有 header 行表示任务已启动但还没有任何单次 fit 完成。

## 使用建议

- 如果你只是想了解 HPC 目录结构，先看本文件
- 如果你需要回看旧的批处理写法或较早期的 Case 2 方案，查看 `hpc/archive/`

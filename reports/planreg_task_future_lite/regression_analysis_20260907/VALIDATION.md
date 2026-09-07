# 本次只读复盘的验证记录

日期：2026-09-07。代码起点 b32b1dee06fb42003a2f69490366cc46db419195。
以下测试为本次新运行；完整 Navtest 的 GPU 导出/官方 CPU 评分是同日此前完成的固定 artifact 评测，不冒称本次又做了一次 GPU 全量推理。

## 测试

```bash
env PYTHONNOUSERSITE=1 \
  PYTHONPATH=/mnt/project/DriveVLA-M0-planreg-task-future-lite:/mnt/project/DriveVLA-M0-env/lib/python3.9/site-packages \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  /mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/navsim_py39_exact/bin/python -m pytest -q \
  tests/test_task_future_regression_analysis.py \
  tests/test_current_only_navtest_export.py \
  tests/test_drivor_scorer_parity.py \
  tests/test_student_checkpoint_export.py \
  tests/test_global_local_register_readout.py \
  tests/test_task_future_loss.py \
  tests/test_same_batch_gradient_diagnostics.py
```

**26 passed, 14 warnings in 33.43s；exit code 0。**
warnings 全部为 Matplotlib/Pyparsing 弃用提示，无 skipped。

新增统计测试明确验证：两组重复 token 可有很高总方差但 centered rank=1；固定 slot identity 不可冒充跨场景内容；rank 的尺度不变性与振幅应分别检查；NaN 不可默默接受。

这些是统计/回归单元测试，不替代真实模型表现。真实读出测量来自两版各 12,146 场景的既有 FP32 forward 输出，梯度来自正式32-rank训练保存的1,408条同批记录。

## scorer 功能审计

运行 `python scripts/audit_drivor_scorer_parity.py`，exit code 0，`passed=true`。

- 固定上游：valeoai/DrivoR@fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a。
- 六个 component logits max_abs_diff 均为0。
- 聚合 PDM score max_abs_diff=0。
- selected indices 相同：[61,39]。
- proposal.grad=None，proposal grad norm=0。
- scene feature grad norm=18.2012882232666。
- 独立 scorer decoder=4层，state_dict shapes 一致，b2d shape 一致。
- TTC invalid mask：loss=0.6931471805599453，reference差0，invalid梯度0。

本次审计记录的本地源文件 SHA：

| 文件 | SHA-256 |
|---|---|
| action_decoder.py | 9e2a5944e40fc7f66e2f06fbc35fcd594dc210252c1c2ea255bd5b467ffbbba1 |
| episode_drive_loss.py | c1e9d1229f68e1ed18a23c696f4f6d92c66da817bc0431685de798152851916b |
| scorer.py | cb82e2442c708d155bf5c5627956606c2152e83cdfe353ff39b2140f2d45c3be |
| transformer_decoder.py | e59d25a2ba4da64adc4c593cbfd9c30e23475da0df48274a30373c16d4bb5834 |

## 只读边界及静态检查

- 新分析脚本完整执行成功，输出 evidence/analysis.json、CSV和3张图；每份输入 bank、事件文件、梯度审计文件、shared init 都记录SHA。
- `python -m py_compile`：新分析脚本、此前的 current-only exporter/repack helper 与新统计测试通过。
- `git diff --check` 通过；提交前另对 staged 新文件执行 cached diff check。
- 模型路径 `navsim/agents/EpisodeDrive` 相对起点无功能修改。
- 没有删除、改写权重/候选缓存/metric cache；没有启动训练或使用 Navtest 产生训练清单。
- GPU 压力脚本保留运行。

## 未运行

全历史243项、重新进行真实模型GPU干预/融合后token导出、分任务梯度、新的teacher learnability试验、完整六头oracle替换、额外seed/epoch：本次是报告和已存真实证据分析。缺少的实验在主报告中明确为NOT_RUN，不以单元测试替代，也不声称根因贡献已被单变量定量证明。

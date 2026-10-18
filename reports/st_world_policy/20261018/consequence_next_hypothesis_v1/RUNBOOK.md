在本分支根目录运行：

```bash
timeout 120s env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python reports/st_world_policy/20261018/consequence_next_hypothesis_v1/analyze_existing.py
```

实际运行exit0，9个合成控制通过。结果已有时脚本拒绝覆盖；独立复核应在独立工作树/输出命名空间执行，保留已有不可变结果。不要删除原结果来重跑。本工具读取硬绑定的C1 boundary3584既存反馈，未实现任何训练、评分、采样或模型调用。

`RESULTS.json` 保存plan/code/data hashes、各读取路径/字节数、32scene汇总、全部分母、合成反例和资源实测。torch只用于CPU解码已有小反馈，以及原float32四候选中心化算术。

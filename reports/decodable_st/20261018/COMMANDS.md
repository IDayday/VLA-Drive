# 已实际执行的命令与恢复

本页命令对应本轮已有run_id/receipt。`RUNROOT`是受控共享存储中的本轮artifact根；`REPO`是本任务worktree；`PYTHON`是已审计的学生环境Python绝对路径。三者的实际解析值留在私有resolved config/启动日志，公开仓库不提交机器敏感路径。不要在新机器上仅凭报告推断资源授权。

先读取当前dispatch policy，选择获授权空闲GPU并设置CUDA_VISIBLE_DEVICES。教师使用各自隔离环境；原生评分使用既有canonical环境。不要改变其他训练的环境，也不要在已完成run目录重跑覆盖计时收据。

## 同步campaign

```bash
cd "$REPO"
"$PYTHON" -m tools.decodable_st.configure_local \
  --artifact-root "$RUNROOT" \
  --output "$RUNROOT/audits/resolved_campaign_v2.json" --python "$PYTHON"
"$PYTHON" -m tools.decodable_st.run_campaign \
  --config "$RUNROOT/audits/resolved_campaign_v2.json" \
  --phase all --execute-ready-stages --continue-independent-on-failure \
  --artifact-root "$RUNROOT/campaign_execution_v3"
```

这组已执行：按SHA复用真实完成收据，audit/teacher/tiny为INCONCLUSIVE，representation/planning为BLOCKED，export为PASS。不是新建shell后声称训练完成。`--phase audit/teacher/tiny/representation/planning/export`、`--dry-run`、`--resume`、`--arm`、`--seed`、`--max-updates`、`--max-exposures`和`--device`均由同一CLI处理；不能用参数绕过科学前置条件或改写已锁定receipt。

## 原生教师与容量（对应既有结果的复现形式）

```bash
PYTHONPATH="$RUNROOT/external/MoGe:$RUNROOT/external/ViGT" \
  "$RUNROOT/envs/vigt/bin/python" -m tools.decodable_st.audit_teacher \
  --config "$RUNROOT/audits/vigt_job.json" --mode vigt-real \
  --output "$RUNROOT/vigt_train64_native_v1"
"$RUNROOT/envs/sgdrive/bin/python" -m tools.decodable_st.audit_sgdrive_native \
  --config "$RUNROOT/audits/sgdrive_job.json" --limit 64 \
  --output "$RUNROOT/sgdrive_train64_native_v1"
"$PYTHON" -m tools.decodable_st.train_compression \
  --config "$RUNROOT/audits/compression_job.json" \
  --output "$RUNROOT/runs/compression32_patch_frozen_v2" \
  --grid 32 --readout patch --max-updates 1000
```

原生ViGT的固定MoGe/ViGT源码通过隔离环境中的已登记路径加载；完整教师配置、权重及依赖hash在lock和私有job文件。容量16/32 bilinear源码4249f651；干净patch复跑使用`execution/final_cff7ce29`。不能从最新报告树恢复旧run并仍写相同训练源码。

## 两个学生与无损状态边界

```bash
cd "$RUNROOT/execution/rs_629e0f1f"
"$PYTHON" -m tools.decodable_st.train_representation \
  --config "$RUNROOT/audits/tiny_job.json" \
  --output "$RUNROOT/runs/RS_gradient_v1" --arm RS --max-updates 1000 --resume
"$PYTHON" -m tools.decodable_st.train_representation \
  --config "$RUNROOT/audits/tiny_job.json" \
  --output "$RUNROOT/runs/Rfrozen_tiny64_v1" --arm Rfrozen --max-updates 1000
```

RS实际走2→203→1000，保存各段成本与梯度证据。恢复加载同一source/config/optimizer/scheduler/RNG/sampler边界；不是从头初始化。两组已完成1000，**目前无需重新执行**。正常完成的冻结旧执行器`state.json`仍可能写RUNNING；以原子RESULT的training_complete、checkpoint update和进程退出共同判断，未篡改原始state。

当前训练器仅接受tiny，full-population入口明确拒绝。不存在本轮已完成的正式P2/P3恢复命令。

真实两rank恢复另用冻结`execution/ddp_resume_c0c8dfcb`，四更新有限执行探针：

```bash
cd "$RUNROOT/execution/ddp_resume_c0c8dfcb"
"$PYTHON" -m torch.distributed.run --nproc_per_node 2 --master_port 29582 \
  -m tools.decodable_st.verify_distributed --config "$RUNROOT/audits/tiny_job.json" \
  --output "$RUNROOT/runs/distributed_resume_v1/resumed.json" --updates 4 --stop-after 2
# 保存退出后，从同一源码、同一两卡、同一状态启动独立进程：
"$PYTHON" -m torch.distributed.run --nproc_per_node 2 --master_port 29582 \
  -m tools.decodable_st.verify_distributed --config "$RUNROOT/audits/tiny_job.json" \
  --output "$RUNROOT/runs/distributed_resume_v1/resumed.json" --updates 4 --resume
```

连续与重复对照同CLI、`--updates 4`，输出continuous.json/repeated.json，省略stop/resume。实际保存了第2步receipt及rank检查点hash后再续第4步；结果是离散状态精确、浮点严格失败保留，并非长程bitwise恢复认证。

## 固定终点解码与独立开发诊断

```bash
cd "$RUNROOT/execution/eval_endpoint"
"$PYTHON" -m tools.decodable_st.evaluate_decoding \
  --run-root "$RUNROOT/runs/RS_gradient_v1" \
  --output "$RUNROOT/evaluations/RS_gradient_v1" --required-update 1000
"$PYTHON" -m tools.decodable_st.evaluate_decoding \
  --run-root "$RUNROOT/runs/RS_gradient_v1" \
  --evaluation-config "$RUNROOT/audits/eval_dev16_job.json" \
  --output "$RUNROOT/evaluations/RS_dev16_v1" --required-update 1000
```

Rfrozen对应run和输出名替换成登记的Rfrozen_tiny64_v1/Rfrozen_dev16_v1；其实际执行SHA分别cff7ce29/df213cf3。每个结果存每场景数值和固定解码切片。dev override只准改已登记开发人口与目标来源，均值仍从train64计算。

## 去教师导出、部署和canonical链路

```bash
"$PYTHON" -m tools.decodable_st.export_student \
  --config "$RUNROOT/audits/tiny_job.json" \
  --run-root "$RUNROOT/runs/RS_gradient_v1" \
  --overlay-base "$RUNROOT/engineering_export_v2" \
  --output "$RUNROOT/representation_export1000_v1"
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PYTHON" -m tools.decodable_st.verify_export \
  --root "$RUNROOT/representation_export1000_v1"
"$PYTHON" -m tools.decodable_st.evaluate_planning \
  --config "$RUNROOT/audits/planning_eval_job.json" \
  --output "$RUNROOT/planning_engineering_dev8_v1"
```

overlay导出拒绝覆盖已存在目录；保留基础权重hardlink、strict delta及processor。planning_job指向2步reader工程export，limit8、原canonical CPU工具；不是1000步空间学生的正式规划实验。Navtest需要独立锁定方法/人口/源码，当前没有这种接受记录。

## 报表与受控可视化

```bash
cd "$REPO"
"$PYTHON" -m tools.decodable_st.render_student_comparison \
  --artifact-root "$RUNROOT" --output "$RUNROOT/private_student_visualizations_v1"
"$PYTHON" -m tools.decodable_st.plot_results \
  --artifact-root "$RUNROOT" --output reports/decodable_st/20261018
"$PYTHON" -m tools.decodable_st.summarize_round1 \
  --artifact-root "$RUNROOT" --output reports/decodable_st/20261018
"$PYTHON" -m pytest -q tests/decodable_st
```

实际教师/学生图不上传。公开结果只保留指标、训练曲线、来源hash、私有面板文件名与hash。旧实验、原始数据和基础大权重没有复制到git。

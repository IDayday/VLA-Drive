# 最新 IL／RL 完整 Navtest：2026-10-24 11:31 UTC 锁定

三份权重均已完成 **12146 场景／136 logs** 的全量评价。主结果为 IL63k **87.419393**、RL_CANONICAL46.2k **90.380828**、RL_SOURCE39.6k **90.014095**；两条 RL 相对共同 IL53k 父模型分别提高 **3.364803／2.998069 个 PDMS 点**，均未达到92。

固定训练 seed42、评价 seed2101；单候选、10步 source shift5 ODE、canonical NAVSIM v1.1、原后处理，无搜索或scorer选轨。这里是2026-10-24 11:31:27 UTC可用最新 COMPLETE checkpoint的用户指定观察；不追逐评价期间继续产生的新权重，不改变原IL100k、两条RL50k终点。新开发的历史两帧＋当前DINO版本另行训练，本表仍是原当前三视图Qwen-Flow。

## 主结果与历史位置

|模型|optimizer update|PDMS|相对共同IL53k|配对95%区间（PDMS点）|
|---|---:|---:|---:|---|
|RL共同IL父模型|53000|87.016025|0|—|
|IL|63000|87.419393|+0.403367|[-0.240921, +0.981407]|
|RL_CANONICAL|46200|90.380828|+3.364803|[+2.641597, +4.001370]|
|RL_SOURCE|39600|90.014095|+2.998069|[+2.407772, +3.508566]|

IL63k的+0.403367只表示IL继续学习，其区间跨0；不是RL收益。两条RL的父模型永久为53k，最新63k IL没有替换RL的Qwen/provider/reference。53k Navtest曾在用户指定父模型前被观察，`Navtest_used=true`；历史Navtest多次观察事实保留。区间按log聚类、保留scene权重bootstrap2000次，不含训练seed或推理seed不确定性。

|历史观察|CANONICAL PDMS|SOURCE PDMS|
|---|---:|---:|
|匹配20k|90.284539|90.233689|
|此前最新观察（35.8k／30.8k）|90.077079|90.264806|
|本次（46.2k／39.6k）|90.380828|90.014095|

CANONICAL比自身20k高0.096289点，SOURCE比自身20k低0.219594点；不能宣称继续训练单调提升。当前两条RL步数不同，不能将0.366733点绝对差归因于reward版本；正式配方对比仍等待共同50k及登记DEV选择。此次结果不用于Navtest调参或重选父模型。

## 安全分项与退化

以下均为0—100分项均值；完全通过要求metric=1，NC=0.5不算通过。

|模型|NC|DAC|TTC|EP|Comfort|DDC|
|---|---:|---:|---:|---:|---:|---:|
|IL53k|98.0899|95.2742|94.0557|81.7227|99.9506|98.4110|
|IL63k|98.1722|95.8587|94.4261|81.5803|99.9671|98.5139|
|CANONICAL46.2k|98.5880|98.1722|96.1798|83.7844|99.9835|97.7812|
|SOURCE39.6k|98.0981|97.5795|94.7966|85.2520|99.8683|98.0323|

CANONICAL相对IL53k的NC/DAC/TTC均值分别增加0.4981／2.8981／2.1242点，EP增加2.0618点，DDC下降0.6298点。SOURCE的NC仅增加0.0082点，DAC增加2.3053、TTC增加0.7410、EP增加3.5294点，Comfort下降0.0823、DDC下降0.3787点。总体提高仍伴随原安全场景新增违规。

|相对固定IL53k|CANONICAL46.2k|SOURCE39.6k|
|---|---:|---:|
|非零→零分|114|143|
|零分→非零|516|417|
|固定IL-safe10905场景的新增联合违规|174|217|
|NC新增／恢复|52／116|65／70|
|DAC新增／恢复|67／419|83／363|
|TTC新增／恢复|109／367|144／234|

|收益分解（PDMS点）|CANONICAL46.2k|SOURCE39.6k|
|---|---:|---:|
|正向G+|5.095643|4.597163|
|负向G−|1.730840|1.599093|

G+／G−分别为所有scene正差／负差绝对值之和，再除场景数转PDMS点；净值等于主表增益。任意小负差与事故次数不同，详见paired JSON中的实用阈值0.005原始score及零分/分项转移。

请求轨迹平均终点前向位置：父IL19.2751m、CANONICAL19.1369m、SOURCE19.6081m；实际模拟终点为19.7822／19.4411／20.1712m。CANONICAL净收益不能概括为一律前移；SOURCE更强进度倾向与更弱安全恢复并存，当前观察只支持这种描述，不是单因素因果结论。请求分段jerk RMS父IL3.9484、CANONICAL6.0255、SOURCE5.9207 m/s³；它不是官方Comfort事件，需分开解读。

## 完整性、部署与资源收尾

所有场景的identity/protocol seal、scene反馈token/log、有限8×3请求、评分请求等于模型请求均通过既有完整分析器核查。30个评价worker全部退出0，无重试、无复用、无缺失分母；总计36438条真实新单候选推理。RL合并导出对原非零LoRA策略的真实图像10步ODE对照通过，归一化最大误差分别2.68e−7、2.38e−7；未将随机SDE成绩当部署结果。

实际使用15张独立物理GPU、30个模型进程，每GPU两进程：vlawm-zt0—7和tsinghua-zt6的1—7。zt6 GPU0上的其他任务保持原状。未向原IL/RL发送停止信号；只在该批评价期间挂起CPU campaign调度owner，结束后恢复。有限legacy admission lease已关闭，两个新授权host不加入普通训练资源池。只读PID/start-ticks复核确认30个本批worker均已退出，15张评价GPU已释放。

## 实测成本

|模型|逻辑场景／实际新推理|累计worker wall秒|加载秒|前向秒|评分等待秒|单模型invocation跨度秒|
|---|---:|---:|---:|---:|---:|---:|
|IL|12146／12146|14114.99|776.72|10882.96|1112.19|1536.32|
|canonical|12146／12146|14142.25|954.26|10894.85|1049.67|1538.14|
|source|12146／12146|14136.36|949.31|10863.86|1056.29|1536.21|

三个模型并行worker总体跨度25.76分钟（11:47:20—12:13:05 UTC）；含导出、GPU导出对照、汇总的owner耗时35.57分钟；从权重登记到分析完成46.24分钟。累计worker wall为11.8294小时，15个GPU各自首个到最后一个worker的占用跨度合计6.3766 GPU-hours；后者包含加载、空隙等占用，不是独占GPU内核计算时间，前者还会重复计共享副本。CPU评分已流式并行；以上加载/前向/等待均是worker累计，不能直接相加当端到端wall。

## 源码、资产与可复核命令

推理／合并源固定`434cba8e47aef3e5ab8319f6b017ab5b0f860db1`；完整性及配对分析固定`70a68db2fdbb17badb3fce31d486db30c42fdc18`。登记、模型hash与实际完整结果见同目录registration、COMPLETE、completion_audit和六份summary/paired JSON。原始本批资产位于：

`/mnt/project/qwen-sourceflow-native-20261018/latest_NAVTEST_20261024T113127Z/`

以下汇总入口已由本批owner实际执行；示例CANONICAL，其余两组替换对应evaluation。复核可写入新输出文件，无需重复推理：

```bash
cd /mnt/project/qwen-sourceflow-native-20261018/snapshots/RL_training_review_20261024
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -m scripts.analysis.qwen_sourceflow.summarize_results \
  --evaluation /mnt/project/qwen-sourceflow-native-20261018/evaluations/canonical_update046200_NAVTEST_user_20261024T113127Z \
  --output /tmp/qwen_canonical46200_summary.json
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -m scripts.analysis.qwen_sourceflow.analyze_paired \
  --evaluation /mnt/project/qwen-sourceflow-native-20261018/evaluations/canonical_update046200_NAVTEST_user_20261024T113127Z \
  --reference /mnt/project/qwen-sourceflow-native-20261018/evaluations/selected_IL_NAVTEST \
  --output /tmp/qwen_canonical46200_paired.json
```

本次支持的结论是：原单候选ODE下存在约3点的真实RL净收益，继续训练未形成稳定单调增益，安全恢复多于新增失败但后者仍存在；尚未达到92。共同50k终点和原定DEV选择继续，不追加新算法、seed或Navtest调参。本报告只上传聚合统计与小型资格证据，不上传权重、optimizer、原始场景或缓存。

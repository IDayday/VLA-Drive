# 完整训练后的固定评估与导出边界

2026-10-14，本轮只补齐正在执行的单组6000步试验的收尾。
模型、几何计算、训练器、损失、sampler、特征读取和相机sidecar等九个已绑定
源码hash全部与运行中的 `EXPERIMENT_BINDING.json` 一致。没有热改训练。

几何读取器与先前全局读取器有相同参数形状；因此仅凭strict state_dict加载
不足以证明调用了正确函数。研究包现绑定模型/几何/公共读取组件源码hash，
读取定义或权重来源不符时拒绝。导出要求对应已选stage_val端点，已有正确包
幂等复用，未写完receipt的窗口使用实际payload内容比较恢复。

`finish_geometric_scorer.py` 仅等待本次实际PID/start-ticks句柄。只有6000步
完整checksum checkpoint、终态result和已释放RUN锁同时成立，才允许评估。
控制器逐阶段重查永久主机授权和GPU实际占用，跳过不可解释的物理显存占用。
训练停止但未完成时报告失败，不启动替代训练。全部收尾只读取stage_val、
selector_cal、固定dev和已拟合D_S，不访问navtest或修改ACTIVE。

最终模型使用四个固定stage_val真实观测，保留原图像batch布局：先从被选端点
生成参考，再由独立进程加载研究包，比较候选、预测分数、选轨及一次VLA计数。
这个已实现入口尚待完整训练结束后实跑；不得把17项单元回归当作该真实验收。

实测命令：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
/root/miniconda3/envs/ddp/bin/python -m pytest \
 tests/iqe/test_geometric_posttrain.py \
 tests/iqe/test_geometric_full.py \
 tests/iqe/test_geometric_readout.py -q
```

实际结果：17 passed，1.75s；仅一个环境pynvml弃用警告。覆盖真实几何投影
函数、候选独立性、梯度隔离、相机内容身份、同形状错实现、外部权重拒绝、
导出中断窗口、终态锁/损坏checkpoint、测试集访问拒绝及不可见GPU占用。
使用fixture的测试明确是工程单测，不替代真实图像和留出选轨结果。

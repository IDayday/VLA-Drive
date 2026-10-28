# Qwen-World：坐标损失权重匹配续训

两组从相同IL63k原生模型、Adam、scheduler、数据游标、全部RNG开始；分别完整续训2000步/global32，共64000次scene曝光。训练seed42，完整Navtest12146，评价seed2101，单候选clean-x十步shift5 ODE，canonical v1.1。无best-of-N，无DEV/Test拟合权重或候选回流。Test历史多次被观察，不能声称完全盲测。

CONTROL保留原归一化L1；PHYSICAL只改变三个坐标的固定权重。工程容差为0.5m/0.25m/2度，提前登记，非论文最优值；输出层梯度量级仅在32个TRAIN场景校准一次。两组均保持原8点时间等权、三维动作编码、噪声日程、全模型训练、四个辅助任务及其系数；原父候选库对两组共同固定，避免异步刷新改变数据。未加入Huber、角度环绕损失、运动学或RL修改。

| 组 | Navtest PDMS | 纵向MAE cm | 横向MAE cm | 航向MAE ° | ADE cm |
|---|---:|---:|---:|---:|---:|
| CONTROL | 88.381806 | 51.913 | 19.287 | 1.3358 | 60.382 |
| PHYSICAL | 88.336938 | 52.104 | 19.225 | 1.3482 | 60.509 |

PHYSICAL−CONTROL：-0.044867 PDMS点，44物理log配对95%区间[-0.273579, +0.196037]；G+=+0.805399，G−=-0.850266点。
本窗口判断：**证据不足：未证明稳定净提升**。只有一个训练/推理seed；2000步是有限续训窗口，不据一次阴性结果否定整类加权损失。

新失败、恢复、原高分→零分及NC/DAC/TTC转移见PHYSICAL_minus_CONTROL.json；相对原IL63k见每组JSON。几何误差为最终请求与原始GT的差，非控制器模拟结果。TRAIN物理误差逐时刻/通道及费用另表，不用加权训练loss自身下降充当成功。

资格验证包含原L1值/梯度对照、物理尺度与mask、多micro归约、资源撤回合同，以及真实八卡连续3步对中断1步再恢复2步的模型/Adam/scheduler/RNG/builder逐值检查。资格权重未进入正式训练。源码、登记、命令和失败证据保留；旧IL/RL没有被热修改或停止。

恢复：原运行目录使用相同REGISTRATION和arm，追加`--resume <本组最近合法checkpoint>`；不可用qualification目录初始化正式实验。启动入口`python -m tools.qwen_world_postcot.coordinate_study_owner --root <登记根>`只有一个owner，已启动后不可重复执行。

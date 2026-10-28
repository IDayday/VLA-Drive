# Qwen-World 坐标损失匹配实验

状态：32个TRAIN真实图像、完整63k模型的物理误差和梯度校准已完成；即将八卡恢复资格验证，尚无新训练终点或Navtest结果。

两组从IL63000的同一模型/Adam/scheduler/cursor/RNG分叉，每组2000步/global32，终点65000；原100k IL及既有RL不受影响。CONTROL为原归一化clean-x L1，PHYSICAL仅改变三个坐标的固定权重。8点时间权重、L1形式、原3维编码、噪声/shift5、全参数训练及四辅助损失保持。两组共同冻结父模型已有辅助候选库，避免异步刷新引入不同人口；不将其称为原100k生产任务的exact resume。

工程容差预登记为纵向0.5米、横向0.25米、航向2度；不是论文最优值、不按Test选择。物理容差换成归一化坐标权重后取均值1；用首32个TRAIN场景的动作输出层/末层norm梯度量级一次性校准总系数，固定为0.86074298620224。这是局部梯度量级匹配，不是全模型梯度等价。没有改变角度环绕、Huber、时间权重、动力学损失、生成时间权重或RL。

主判断：固定终点两组全Navtest12146、canonical v1.1、单候选K10 shift5、seed2101，报告PHYSICAL−CONTROL及相对IL63k的配对log bootstrap、安全/零分转移和G+/G−，不用TRAIN/DEV损失宣布成功。一个seed，2000步阴性不足以否定整类方法。

已有9项CPU资格PASS；下一步连续3更新与中断1+恢复2更新的真实八卡模型、全部Adam/scheduler/RNG/builder对照。资格权重不用于正式初始化。资格和失败成本保留。首个临时主机尝试缺少解释器，未启动模型；随后探针缺少CuBLAS确定性环境变量，在更新前停止，修复显式环境后校准PASS。未影响旧训练或已有Test结果。

实际CLI：`python -m tools.qwen_world_postcot.coordinate_study_owner --root /mnt/project/qwen-world-postcot-20261025/coordinate_loss_20261028`。单owner启动后禁止重复派发。正式组使用`tools.qwen_world_postcot.train_il --coordinate-registration <REGISTRATION.json> --coordinate-arm CONTROL/PHYSICAL`；恢复追加本组原生`--resume <checkpoint>`。所有命令均须从本分支运行，设置`WORLD_POSTCOT_TASK=qwen_world_coordinate_loss_20261028`，原生产源码不热改。

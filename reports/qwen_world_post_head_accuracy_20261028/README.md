# 动作后PDMS六分项预测检验登记

当前状态：真实IL60k辅助head已从匹配原生checkpoint严格提取；CPU数值检查通过，尚未运行完GPU准确性检验。

输入为当前合法观测和两条固定请求轨迹（旧IL40k、IL53k的完整Navtest保存输出）；真实分项仅用来计算预测误差，不进入actor输入或更新。使用2048个独立hash自然样本，加250个NC/DAC/TTC边界差异定向场景，分开汇总，不能把定向均值当总体表现。

训练辅助目标为六分项soft-label BCE＋0.25×NC/DAC成对排序，总IL中乘0.2。5k→60k的500更新窗口post均值0.627039→0.134424；期间候选库刷新，不能把这个下降当固定测试集准确性。EP等软目标BCE的理论最小值并非0，需看目标熵下界、excess BCE、分项MAE和违规识别率。

验证将报告每项MAE/RMSE/均值偏差、硬0/1子集Brier/AUROC/AUPRC、违规precision/recall与always-pass基线；NC=0.5等单列；还报告同场景两候选排序与组合PDMS误差。无双类时AUROC留空，不用0代表“没有风险”。

无新policy采样、无新仿真评分、无训练更新；这是预测头诊断，不是部署PDMS或已实现修复能力。主评测已用的GPU先排除。辅助检验可在其它许可卡上与已核实的World训练共享：保留32GiB最低实测显存门槛，绑定全部现有训练PID/start/argv哈希，300秒准入有效期，并持有专用诊断互斥lease；不与未知进程或评测worker挤占同卡。原主评测使用的lease与原训练任务保持。原训练和评测代码路径不改。

实际入口：`python -m tools.qwen_world_postcot.diagnose_post_head --help`；有限调度入口`python -m tools.qwen_world_postcot.post_head_audit_owner --help`。新增job_worker白名单仅允许此只读诊断模块；GPU UUID、head/model/data/score身份均记录在本地receipt。全部worker退出后唯一owner自动汇总并推送。

控制器首次只等待主评测入场，尚无辅助GPU任务时已按精确PID/start停止；记录CONTROLLER_HANDOFF_001，旧owner/日志保留。第二次仅变更执行准入和controller目录，预测模型/数据/数学实现与663d379逐字相同。

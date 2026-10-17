# Channel-control 独立窄审查

Reviewer：`/root/st_road_proxy_hypothesis`（不是该补丁作者）。日期：2026-10-17 UTC。

审查对象：`/mnt/project/VLA-Drive-st-teacher-channel-control-20261017`，固定提交 `bd2d1dd9a11ed3b236a95b680884f1495eddbe6f`；对比起点 `951711fbd2eae1730262a79575b08024bcd2001d`。对象工作区 clean。读取完整 14 个 owned 文件及必要的固定 runtime 调用实现。

**结论：未发现阻断配置准备合入的缺陷；可合入作为未运行的对照配置与诊断工具。此结论不是 formal 启动审批、真实数值集成验收或科学效果 PASS。**

## 当前核验

| 项目 | 静态/解析核验结果 |
|---|---|
| 384 严格复用 | `PCA384.json` 与已完成 SP2 的原 `CONFIG.json` 字节相同，SHA256 `16f8716ef14a2a9a57bca234be16357decc346e8919cc35f781fcc6f69e7abdd`。run ID、checkpoint receipt、源版本和原输出目录均保持原身份；没有建议重新运行 384。 |
| 256 唯一变化 | 与 384 的 JSON 结构比较，仅 `teacher.channels`、由维度决定的 `transform_identity`、隔离 `output_root` 不同；相机、384×288 教师图像、18×24 原生 grid、W48/view、K96/view、seed/更新预算/optimizer/精度均相同。 |
| 同一 PCA 基底 | PCA identity 为 `268723c8da3151492fd20a81b0f6f541f348a1187222132b055f4e4a528bfd46`，train-only，dimensions `[256,384]`，标准化且不白化。用 Python 标准库独立重算原契约和两份 projected transform hashes，均匹配。未读取 PCA tensor 内容。 |
| 元数据来源 | 10 个 `INPUT_PINS` 文件共 294,546 字节，尺寸/hash 全匹配。独立核对 `STUDY_BINDING` 的 config/target hashes、共同 Base/data/teacher 与 SP2 registration 的引用、source SHA/fingerprint。 |
| 旧端点 | receipt 的 `completed=5000`、`exposure=159992`、state identity 与 study binding 一致；不是假设恰好 160000 次曝光。只核验 receipt，不重新 hash 约 37GB 的 checkpoint 数据。 |
| 初始化身份 | 原 initialization 记录 Base identity 匹配，master 状态 `RESTORED_EXACT_FP32_SOURCE`。共享 readout 参数的同种子构造有作者 CPU 测试；文档明确最终输出 bias 不是公共前缀相等，不宣称整头逐元素一致。独立审查未重新实例化模型。 |
| 实际接线 | 固定 runtime `fdf25f99b702754787904215bf1c2a2b7515ea75` clean；`STWorldTrainingDataset` 使用 `expected_contract.channels` 加载同 basis 的前缀，`SpaceTimeStudent` 将该维度交给 `SparseFeatureReadout`，trainer 将相同维度交给全局有效元素分母。`current_weight=1`、原 FM repeat8 与十步推理不变。 |
| 误差口径 | 新诊断分别报告公共前256与额外128；`(256*MSE_common + 128*MSE_tail)/384` 正确。256 未预测的 tail 不被当成零误差。`relative_sse_to_zero` 明确是零预测能量基线，不是 train-variance 白化。 |
| 结果诚实性 | 256 为 `NOT_RUN`，学生效果 `UNTESTED`；9 CPU tests 是作者现有日志，未伪装真实 GPU/模型效果。本 reviewer 未重跑这些 Torch 测试。作者的测试 rchar 超出128MiB已在 RUNBOOK 主动披露。 |

## 非阻断的校验覆盖缺口

`prepare_pair.py::main / verify_pair` 检查四个配置/目标文件、固定 runtime 和 `INPUT_PINS`，但不自动读取 `STUDY_BINDING.json`，因此不能发现仅该文件的 arm hash 或 common-artifact crosslink 被误改。这次独立标准库检查已确认这些链接全部一致，所以不是当前产物错误。

若未来将 `prepare_pair.py` 的 `CONFIGURATION_PASS` 解释成“完整 study 身份已校验”，应增加对应交叉检查及负向测试；当前输出明确 `formal_launch_allowed=false`，正式新 registration 仍需实际核验，不应把此准备检查当审批替代。

## 真实执行前仍需要的证据

仅 256 需要新 registration、实际环境/资源核验及独立启动审查。继续保留已有 384 端点，使用相同初始化、数据顺序和原20k学习率时间轴下的5000步对照。真实 paired predictions、learned-W probe、完整固定 dev 规划/安全结果仍未产生；这些缺失不妨碍准备代码合入，但不能据此宣称256更好或正式实验已通过。

审查执行只使用 shell、git、JSON/hash/AST 标准库解析：未 import Torch，未打开模型权重/教师 tensor/大缓存，未使用 GPU、SSH，未改对象工作区和任何运行任务。结果是窄独立代码/证据审查，不是重新运行原作者测试。

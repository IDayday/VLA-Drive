# 最终Navtest准入：实现与已执行验证

新增`scripts.analysis.fm_transfer.final_lock`，只做既有资产、固定终点、预算、导出和评价证据核查，不启动模型或修改训练。它等待A/B各自完整SFT、各自固定20k RL以及全部登记DEV/训练诊断完成后，锁定四份权重和native/shift5两个独立读出；不按分数或seed择优。

本次CPU执行了16项针对性测试，全部通过。覆盖不完整/错误阶段恢复点、8rank预算与事务不一致、hash改变、自己的SFT祖先与导出不匹配、未完成准入拒绝，以及完整小型夹具的锁定与禁止覆盖。真实资产核查也完成：既有12,146个Navtest场景/136logs的输入与v1缓存成员均存在；没有新推理或读取Navtest得分来选方案。资产存在性不是对全部图片/缓存内容重新做数值验收。

实际已有12个里程碑及19份复用/分析证据通过核查，A完整SFT的八rank终点字段也通过。真实`check`仍输出`WAITING_FOR_BOTH_ENDPOINTS_AND_DEV`；没有发布真实终点LOCK，没有启动最终Navtest，也不把synthetic正例通过说成实验完成。

入口保留当前已资格验证的不可变评价source `45d2318d2dec265583860a26f891d8b79bb6068a`，检查与既有里程碑的worker hash一致。训练当前仍运行各自原不可变源码；本补丁不热更新、不重启训练。

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests/rl/test_fm_transfer_final_lock.py -q
CUDA_VISIBLE_DEVICES='' python -m scripts.analysis.fm_transfer.final_lock check
CUDA_VISIBLE_DEVICES='' python -m scripts.analysis.fm_transfer.final_lock prepare --output NEW_ASSET_RECEIPT.json
# 只在两个完整终点及全部固定DEV证据满足后才可能成功：
CUDA_VISIBLE_DEVICES='' python -m scripts.analysis.fm_transfer.final_lock lock
```

前两项和实际`prepare`已执行；真实终点`lock`尚未执行成功。该入口不是最终Navtest自动派发器，后续派发仍需现有worker入口、单一owner、即时主机授权及GPU身份检查。仅实现准入不代表完成最终评价。

私有证据：`/mnt/project/fm-transfer-20261016/final_navtest/assets.json`与`resources/final_lock_validation_v2.json`。后者记录实际验证的19份证据hash、实现hash和仍缺失的终点文件。没有上传原始Navtest清单、数据或缓存。

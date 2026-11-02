# IL44 parent audit

原生父权重：`/mnt/project/qwen-world-four-frameworks-20261029/runs/ARCH_D/IL_seed42/checkpoints/update044000`。

本轮不可变资产引用：`/mnt/project/d44-native-rl-20261103/IL44_PARENT`。按完整inventory硬链接保留原模型/训练head与原scored-export元数据，不额外复制8.3GiB模型，不加载旧IL optimizer进入RL。

- architecture D；formal IL updates44,000；cursor1,408,000；joint阶段；原生COMPLETE有效。
- native model SHA256：`9715b9c0cfbef7a7d3d94fb83e7b0e2f0838d21cf06d3277deef6733cccbc50e`。
- checkpoint files inventory SHA256：`49d3c1220f8720fca0a66a4b7be4da6293c4f226c0753b1f4897c699fb0d91ff`。
- identity SHA256：`c48f61ae2efd3292d4e719dd03f8ddc24f27f65e7ca40590f27c9aeec94d4d34`。
- 官方Qwen revision：`89644892e4d85e24eaac8bacfd4f463576704203`，processor/chat-template/权重各文件hash保存在原IL初始化收据。
- normalizer保持IL的源3维ActionCodec，不切换DDP z-score/sin-cos；身份由原source文件hash与部署配置绑定。

完整Navtest来自已有封存的D44 export：`periodic_NAVTEST_ARCH_D_044000_20261102T142147Z/D_IL_044000/SUMMARY.json`，SUMMARY SHA256与全表见[parent_audit.json](parent_audit.json)。

| 指标，百分制 | 值 |
|---|---:|
| PDMS | 89.0601875684 |
| NC | 98.5880125144 |
| DAC | 96.9372632966 |
| TTC | 95.4470607607 |
| EP | 82.9239676130 |
| Comfort | 99.9917668368 |
| DDC | 98.5386135353 |

该结果是12146场景/44物理logs/invalid0/seed2101/B1/K10shift5。新部署加载器在16场景资格复查中，与原D44 export的请求轨迹及所有原生分项逐值一致。这16场景不是新性能结论，也不取代完整基线。

44k训练轨迹loss必须标注是归一化clean-x L1；单rank日志样本约0.005355，不等于厘米误差，也不是整体IL loss。八rank与窗口汇总见训练loss审计产物。

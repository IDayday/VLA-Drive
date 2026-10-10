# Source provenance

原工作区 `/mnt/project/DriveVLA-M0`，起点 commit `6e96cf7321b134c42c2cf0fbbc315cd61c925b11`，原分支 `feature/navsim-candidate-relative-feasibility-audit-V2`。未 reset/stash，未覆盖、删除旧工作区或合并其他实验分支。原有两份 tracked 修改及104份无关 untracked 文件保留在隔离工作树；只提交本任务文件。

隔离工作区 `/mnt/project/DriveVLA-M0-iqe-20261009`；本任务分支 `feature/s0-incremental-query-experts-20261009`。完整原 tracked patch 为 `BASE_DIRTY.patch`，SHA256 `d54c0822b288f047903059c5910401b822c0fddc01f4f994896fd16b1742a896`。原始文件快照在 `/mnt/project/iqe-workspace-snapshot-20261009`，workspace manifest 在本目录。origin 为 IDayday/VLA-Drive；未 push/PR/合并。

实际框架绑定 `/mnt/project/VLA-Drive-action-video-source-1493ded`，源码 commit `1493deda247107efe076b9294863a6753391298a`，配置及 registration 为 `/mnt/project/action-video-foresight-artifacts/20261002/registrations/formal_S0_seed42_full100k_v2{_config,}.json`。从原调用链核实为 DDPActionVideoForesight → DDPFullForesight → DDPForesight → Qwenvl_OFT。现仓库64-query候选路径不作为 S0；只复用其 Q-Former/Decoder/MLP 实现并记录源码 hash。

用户先要求单 Query S0，随后明确纠正“原本 DiT + flow matching”“用原轨迹 S0”“基础框架，不是权重”。因此新建 Query 基础训练，保留原输入、通用 VLM 初始化及已有辅助监督，未加载历史100k驾驶 checkpoint，也未从64-query切出一条作为基础模型。

通用 VLM 实际路径 `/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-Instruct`，hidden2048。权重分片 SHA256、tokenizer/prompt、normalizer、官方 NAVSIM 源码和模块 hashes 由 S0_CONTRACT.json 给出。该路径包含 DriveDreamer 名称，仅为本机通用 Qwen 文件存放位置；没有引入其 DiT 策略或其他世界模型。

原正式 run registration 包含初始化、源 commit、world8/global32/micro4、targets identities和100k预算，未找到完整原 shell 命令；契约将 shell 字段记 null 并保留具体证据，不把推测命令写成运行事实。

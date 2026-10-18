# 原生教师审计

权重目录存在不作为加载成功证据。本轮严格加载了完整原模型与所用 decoder；没有随机补齐，没有 strict=False。

| 教师 | 源码 commit | HF revision | 权重 SHA256 |
|---|---|---|---|
| ViGT | a0bab2af48c6485774778fec33d68364c70d5618 | df0601a7c5fa71ce813b49143dd5a3692b6ebc8d | f19b10d90904e3535c8ef4c00cb832e3fd5c6ba904bbc7c2421578ab0b93d69d |
| SGDrive stage1 | fcda385371f8f40af5b8969ca5090be06f421358 | c6379bea8aaf635883b33fcc99499b59d42aad1e | 0a483311eff40bd6f058c0d6c8d84f467d0faec31f1d2581e03f08f891c5a00f |

ViGT 源码/权重 MIT；SGDrive 源码 MIT，HF 权重卡 Apache-2.0。第一份本地来源锁未区分 SGDrive 权重许可证，v2 单列修正，旧文件保留。逐文件锁及原样 vendor 在 `third_party/decodable_st/`；包装、坐标边界拒绝、fresh shapes 属于本项目适配。ViGT 的 MoGe 依赖固定旧 v1 API commit 07444410f1e33f402353b99d6ccd26bd31e469e8；权重单独校验。教师各用独立环境，未升级共享训练环境。

## ViGT

HF 权重没有 config.json，沿官方 notebook 的 ViGTHFConfig 默认构造，完整 559 keys 严格匹配。实际 query encoding 是 Identity，没有在缺 tinycudann 时替换非 Identity 编码。

- encode context：`[B,256,256,256]`，布局 CXY。
- decode query：`[B,N,3]`，native ROI x/y∈[-40,40)，z∈[-1,5.4)。
- native output：`[B,N,1]` occupancy logits；实际缓存为 `[1,4096,1]`，不省略最后一维。
- 原 decoder 保留 ROI 归一化、xy 转换、grid_sample、学习偏移 dq。ROI 外在调用前拒绝；未采用可能清零 ego 区域的渲染便利函数。
- 64 个预先选定训练 logs 的真实当前三图：完整 forward 与 split encode/decode 最大差均为 0。
- 固定 decoder 的输入梯度非零、有限，参数与 buffers 不更新。FP32 以及 BF16 存储/局部 autocast 的算子有限性通过；正式数值协议仍 FP32。

64 场景完整 context 占 4 GiB；全量 101592 场景预计约 6.2 TiB，因此没有构建全量 dense cache。另实际构建了 64 场景稀疏特征+精确原生 logits 缓存，约 602112 bytes/场景；全量约 57 GiB。稀疏 context 不被补零送回教师 decoder，logits 均先在完整教师 context 上离线计算。

能力仍 INCONCLUSIVE：原生一致性只能证明接口；独立 free/unknown 真值与高度原点尚未验收。ViGT 公开来源包含 nuPlan，NAVSIM 日志重叠未知，不能宣称严格无测试来源交集。[ViGT 代码](https://github.com/whesense/ViGT)、[论文](https://arxiv.org/abs/2602.05573)。

## SGDrive stage1

实际权重为 text hidden1536/28层、vision1024/24层；遵循实际配置，不按 HF 标题推断型号。完整 1119 keys 严格匹配；当前/未来两个 decoder 各 119 个参数/buffer key 全量匹配，没有 stage2/3 补权重。

- current/future hidden 均 `[B,625,1536]`。
- 各自 Decoder2D 包含原 project，raw `[B,32,100,100]`。
- 原转换为 `[B,2,100,100,16]`，单元测试定位 channel/height/xy。
- 每次调用新建 shapes list，避免原 forward 的 pop 副作用。
- 64 个真实当前场景，以明确的合法当前信息 prompt 运行原模型 inference，与捕获 hidden 后独立 decoder 的当前/未来输出最大差均 0。这个 prompt 保留原 world/dream token 接口，但不是含完整原训练 QA 的能力复现；其分布差异也限制教师能力解释。
- 教师没有 future QA/GT 轨迹前缀；原前缀包含当前 CAN 速度/加速度、合法 ego history、导航。其当前状态字段比 S0 多，已单列；不能当作与 S0 完全同信息的匹配教师对照。

原 get_occ 在 raw LiDAR 坐标中取点，以真实未来 ego 轨迹定义局部走廊，缺点与走廊外写零；没有可靠的 ego(t0) 时间补偿与 unknown 语义。future_idx=1 的路径列表时间构造未能从发布资产核验；不能将其猜成未来 1 秒，也不能把单未来二分类局部图称为 Occupancy Flow。

T-driving 代码只开放 future-head 适配，但在可靠标签/物理 horizon 前保持 BLOCKED。尚未运行适配、动态/变化区域 persistence 比较或时间蒸馏。不能由大面积静态 background IoU 绕过此条件。[SGDrive 原仓库](https://github.com/LogosRoboticsGroup/SGDrive)。

# 基线审计

用户明确选择 S0。采用原始 GT、共同 100k 的既有 S0，不按本轮结果重选 checkpoint。

| 项 | 实际证据 |
|---|---|
| 基础训练源码 | `1493deda247107efe076b9294863a6753391298a` |
| 基础 checkpoint 聚合 SHA256 | `78f870647015194798af067eefc8d8fedb6e59e6db2f3ae719c63b783871446f` |
| 本轮只读 baseline manifest identity | `428d31e219e5cb6c7cfa8c8486a561747eb1284114f0cc3b7eb4b4f63e127f91` |
| Qwen | Qwen3-VL-2B-Instruct；实际 hidden=2048，SDPA |
| W | 144，保留原可训练 query/位置及输入接口 |
| 当前 DINO | timm/vit_large_patch16_dinov3.lvd1689m |
| DINO revision | `30c1109559f65dea34316b0d4842d35c5771fe11` |
| DINO 权重 SHA256 | `45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941` |
| DINO 标签身份 | `7663c45b77dd711e8304e8d202644dcaa8467f7e265ff8b4c1816d59d2c131cb` |
| 共享历史训练成本 | 483.4813303 GPU-hours |

DINO 使用最后 post-norm dense patch，去掉 5 个 CLS/register，未做 L2。完整 16:9 源图经 bicubic 到 256×192；原 12×16 网格 pool2，得到每视图 6×8。学生原 TokenProjectionHead 与当前目标 weight=1 保留。不能把新的 ViGT head 叫作原 DINO 实现。

当前学生输入三前视 F0/L0/R0，当前时刻，16:9 ROI，1024×576 Lanczos。合法 ego 为原历史姿态计算的 `[1,4]`；没有由未来位移估当前速度，没有新增历史相机或标定输入。动作 8×4（归一化 xy、sin/cos heading），4 秒、0.5 秒；原 repeat8、Beta(1.5,1)、noise_s=.999、10 步 Euler、同场景 SHA256 绑定噪声。原 DiT 与后处理未重写。

S0 的历史训练还包含未来 DINO sequence 与交互监督：权重分别约 1.15495435、0.8328946。它不是纯当前 DINO 从头训练的模型。本轮保留其成熟权重作为共同起点；新阶段只使用登记的当前 DINO replay，不把历史额外监督抹掉。原始 tokenizer 文件、扩展 token rows、模型/optimizer 8 分片哈希在私有只读 manifest；新空间/时间 query 直接追加 embedding，不改词表。

新模块关闭时，同噪声原轨迹逐位一致。最初把尾部直接加入完整序列的实现改变 SDPA 张量形状，严格 hidden 容差失败；原失败保留。改为当前 forward 内原 prefix + 可微 KV 的尾部调用后，同批 action hidden 和轨迹均最大差 0。真实 batch=1/2/4、左右 padding、混合图像 grid 的同批 prefix 均逐位一致。batch 与逐样本之间仍有原生 kernel 形状舍入差异，不能宣称跨 batch bitwise 一致。

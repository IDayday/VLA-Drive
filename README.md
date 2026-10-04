# NAVSIM 优化轨迹 v3：便携训练数据

这是只包含数据和读取工具的分支 `data/navtrain-optimized-20261004-v3`。
上传的是上一轮已完整发布的 v3；正在运行的 v4 不包含在这里。

核心文件 `data/optimized_trajectories/navtrain_20261004_v3/trajectories.npz`
约 **18.16 MiB**。只带轨迹、token、验收标记和声明的扰动尺度；
优化候选、184 MiB 的逐场景评分 CSV、逐场景证明文件和机器路径均不需要下载。
采用标准 NPZ 的无损 LZMA 压缩，`numpy.load` 可以直接读取，不需要 Git LFS。
float64 轨迹、全部 token、顺序和验收标记与原发布版逐字节一致，没有减点或量化。

## 在另一台服务器下载

```bash
git clone --depth 1 --single-branch \
  --branch data/navtrain-optimized-20261004-v3 \
  git@github.com:IDayday/VLA-Drive.git navtrain-trajectories-v3
cd navtrain-trajectories-v3
python tools/optimized_trajectories.py verify
```

浅克隆只下载这个数据分支，不拉训练代码的大历史。读取和校验仅需 NumPy；
LZMA 支持是 Python 标准库的一部分，读取不使用 GPU。
同目录提供 `SHA256SUMS`；也可在数据子目录执行 `sha256sum -c SHA256SUMS`。

## 数据与训练语义

| 数组 | 形状/类型 | 含义 |
|---|---|---|
| `tokens` | `[103288]`，Unicode | 场景 token；每个 token 恰好一条轨迹 |
| `trajectories` | `[103288,8,3]`，float64 | 当前自车后轴局部坐标系的绝对未来姿态，x/y 米、yaw 弧度 |
| `accepted` | `[103288]`，bool | 92,015 条已验收；11,273 条为未验收的原 GT 回退 |
| `acceptance_noise_scale` | `[103288]`，float32 | 1/.5/.25 为已验收的声明尺度；0 表示未验收回退 |

8 个点对应 **0.5、1.0、…、4.0 秒**，不含 t=0，不是逐点位移增量。
62,546 条按标准尺度验收，4,559 条按半尺度，24,910 条按四分之一尺度。
`accepted` 不等同于“标准尺度下所有扰动均成功”。

来源版的全量平均 PDMS 为 **98.0042/100**，存储的全尺度扰动零分率为 **0.6208%**。
这些是保留历史验证种子的描述性统计，不是全量统一新种子评测；
已验收轨迹的密集几何检查失败数为 0。详细来源、数组 SHA256 和统计在同目录 `release.json`。

训练时按 **token** 对齐自己的训练集，保留原训练划分和样本数；不要按行号假设顺序一致，
也不要把开发集或 Navtest 场景加入训练。
`accepted=false` 保留现有原始规划标签。用于视频动作条件的实际自车轨迹、真实视频和
周车辅助标签继续使用原日志；优化轨迹用于规划监督。

```python
from tools.optimized_trajectories import load_release

metadata, bank = load_release()  # 校验文件/数组哈希、唯一 token、坐标与验收标记
position = {token: i for i, token in enumerate(bank["tokens"].tolist())}

i = position[scene_token]       # 缺失 token 会报错，不会静默错配
if bank["accepted"][i]:
    planning_pose_xy_yaw = bank["trajectories"][i].copy()  # [8,3]
else:
    planning_pose_xy_yaw = original_gt
# 模型所需的归一化、sin/cos 或 float32 转换继续交给现有训练编码器。
```

也可以直接 `np.load(path, allow_pickle=False)` 读取。
如需批量对齐，`select_tokens(bank, training_tokens)` 按给定 token 顺序返回数组，
缺失或重复 token 会报错。

## 兼容现有 S3 标签准备入口

在数据目录将便携包展开成原标签准备器认识的 `campaign/final/` 结构：

```bash
python tools/optimized_trajectories.py export-campaign \
  --output /your/data/navtrain_v3_campaign
```

然后在已有 S3 优化轨迹训练代码目录中执行：

```bash
python -m tools.s3_optimized_training.run prepare \
  --campaign /your/data/navtrain_v3_campaign \
  --data /your/data/student_train_v1 \
  --labels /your/data/labels_navtrain_v3
```

这会按该训练集 token 对齐；源包 103,288 场景不会强行覆盖已有训练人口。
导出只重新压缩/整理来源信息，姿态值不变；导出使用自己的内容身份并保留原发布身份。
此数据分支不包含训练代码，不会启动或改变已有训练。

发布前已逐字节比较源版的 token、float64 轨迹和验收标记，并通过三个读取/对齐/导出测试。
另已用现有 S3 准备器对齐真实的 101,592 个训练 token：90,513 条验收轨迹、
11,079 条原 GT 回退；源包中其余 1,696 个 token 未加入该训练划分。

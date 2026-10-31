原生 Safety10k 学习率 × 数据对照：启动与源码归档（2026-10-31）

四个8-rank任务已实际启动；启动有限梯度与完整新10k checkpoint只能证明执行，尚无新Navtest收益结论。
当前最高已验证父模型仍为 SafetyEP10k，PDMS 91.34989173702405。

四臂均继承 Safety10k current、Adam及步数、全部rank RNG与stream cursor20000，global10k→20k；不是新phase0，也不是仅weights初始化。
保留W3[7,8,9]、canonical reward、SafetyEP优势、beta50、原IL70 reference与原scope。
LRLOW：action1e-5 / Memory1e-6，full uniform。
LRCOS：action5e-5→5e-6 / Memory5e-6→5e-7，global10k→20k相对cosine，full uniform。
DATA：action5e-5 / Memory5e-6不变，静态80full/10low/10high；从nativecursor20000的stage2开始。
LRDATA：低固定LR加相同静态数据。DATA/LRDATA没有CFM teacher loss。
原Safety full/constant是四格control；cosine是额外调度形状对照。不同arm不重置optimizer以制造额外变量。

独立owner PID2385918/start1486502789，已精确收养attempt002；目标12/14/16/18/20k，训练不等评测。
12–18k Navtest由现有cadence负责；本owner只20k Navtest与TRAIN1024。cap20k不是50k承诺。
独立有限analyzer PID2385987/start1486502809，50组实际配对+5组四格交互；44物理log、2000bootstrap、seed42。
每个globalu与同u原Safety及固定Safety10k分别比较。缺同u control就等待，不拿10k代替；固定父score≥.9 cohort用于高分保持。
拆PDMS/NC/DAC/TTC/EP、zero修复与新增；全部实际结果齐全才COMPLETE。24h超时或真实失败独立标记，不补点、不自动停止/派发。

首次attempt001因Invocation要求新root本地1..10k有效历史而在首次optimizer更新前失败，未生成新child10k CP。
Root以显式native-fork journal origin10000修复，保父checkpoint/hash provenance，只核child10001..，没有伪造父journals。
失败source/root/receipts仍保留，不能当作四条路线无潜力证据。attempt002四任务已真实更新且新child10k seals READY。

本目录是一次精确快照，不是实时状态。原运行路径见SOURCE_ARCHIVE.json，HTML与原owner STATE继续更新。
owner CPU数学receipt对应d439版本，569仅加入与cadenceV2相同的pidfd兼容；保留真实测试版本，不改写收据。
运行owner569/analyzerf9d源码未改；仅外部launchplan补started及实际OWNER/LAUNCH链接。
所有NPZ、训练日志、权重、完整103288人口manifest和逐场景评测文件均不入git；本归档不改变科学runtime/主controller/运行source。

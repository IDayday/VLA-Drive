# 本次结果打包检查

执行时间：2026-10-14 23:33 UTC。此次工作只读取已有科学产物、整理报告并推送，不重新训练、重跑 Navtest 或重测模型。

- 36 个原始证据文件已按原字节归档，逐个验证大小及 SHA256，与 `SOURCE_MANIFEST.json` 一致。
- `RESULTS.csv` 的 23 行分开标注 dev_report、stage_val 和已有 Navtest；分别核对 1696/16、1024/53、12146/136 的场景/log 分母。
- 所有有完整分项的行核验 `selected - Base = Oracle gain - regret`；百分制和百分点转换与原 [0,1] 标签一致。
- 原 S 与 R 矩阵固定 dev 场景清单 hash 一致；专家训练证据、S 矩阵和已有 Navtest 的候选池 hash 一致。
- 已有 Navtest 的完成 receipt 与结果文件 SHA256 一致，且明确没有执行 Scorer。
- 几何训练实际终态 receipt 为 `COMPLETE / MAX_STEPS / 6000`；未将正在进行的最终开发评估标记完成。
- 汇总 README 的所有相对文件链接存在；新生成 CSV 使用 LF，原始证据 CSV 保留 CRLF 字节以维持 checksum。`git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --cached --check` 通过。
- 对研究分支起点以来的变更及本次新增报告做发布前检查：无大于 5 MB 的新增文件、模型权重/数组文件或匹配到的常见私钥、凭据模式。此检查不声称覆盖所有可能的秘密格式。

上述是本次实际报告检查。已有单元、真实图像、冻结和恢复测试均引用各自的原始运行证据，不把历史测试数量写成本次新执行结果，也不累计重复测试。

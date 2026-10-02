# 实验协议

## 1. 模态范围

**主线**：公式（LaTeX）、表格（HTML/Markdown）、公式-表格混合区域。  
**对照（可选）**：纯正文，仅作消融，不追求主指标。

## 2. 基准

| 基准 | 用途 | 指标 |
|------|------|------|
| OmniDocBench (v1.5/1.6) | 多系统普适性 | 元素匹配、公式、表格官方指标 |
| OCRErrBench | 诊断 / 修复闭环 | Acc、BAcc、Type-F1、Loc、ExactFix、VisFix、Preserve |
| UniMER-Test 公式子集 | 公式精修 | CDM、Case-F1 |
| 表格 TEDS 集（如 FinTabNet / OmniDocBench 表格） | 表格修复 | TEDS |

## 3. 基线系统（建议至少 4 个）

PaddleOCR-VL(-1.5)、MinerU(2.x)、DeepSeek-OCR、MonkeyOCR；可选 Qwen-VL 系。

每个系统：

1. 用官方推理得到 OCR 结果
2. 运行本方法闭环修复（仅公式 / 表格区域）
3. 在同一评测脚本下对比 before / after

**必须报告 Bad 子集**（初始预测错误的样本）提升，以及全集是否因过修而下降（Preserve）。

## 4. 闭环指标

| 指标 | 含义 |
|------|------|
| Preserve | 原本 Good 的输入未被改坏的比例 |
| ExactFix | 归一化后完全恢复参考 |
| VisFix | 渲染后与源图视觉一致 |
| Cost | 平均轮次、渲染调用、token / 时延 |

当前对比脚本读取官方 CDM/TEDS，不自行重实现。单条 metric=1 的 Good/Bad 划分仅作为标明 `proxy` 的分支统计；VisFix 保持空值，需独立盲审或视觉验证。页面平均和样本平均分开报告。

前后配对固定页面、GT 索引、GT 位置与归一化 GT；匹配变化时停止配对统计。策略和 judge 的 Observation 不携带参考标注或官方分数。训练/开发/测试必须隔离来源页面和图片，派生裁剪/增强还需近重复审核。区域格式见 [DATA_PROTOCOL.md](DATA_PROTOCOL.md)。

## 5. 判定层消融

- A0：冻结 DocEDR verifier（原文）
- A1：纯 Jev
- A2：Jev 级联（高置信接受，低置信升级教师/DocEDR）
- A3：Jev + 确定性渲染/结构 diff（推荐，防 judge 分数不可靠）

基础对照包括保持原始输出、一次修复、不更新渲染的多轮和更新渲染的闭环；保持学生、数据和推理预算一致，再增加 SFT/GRPO 对照。

## 6. 训练配方记录

每条 run 必须记录：

- 基座模型与版本
- 数据构成（源、数量、文/公式/表格比例）
- 课程阶段与步数
- reward 各项系数（α, λ, μ, ρ）
- GRPO 超参（G、ε、β、batch）
- 硬件、吞吐、墙钟时间

## 7. 复现

- 配置进 `configs/`，脚本进 `scripts/`
- 结果表用 `experiments/tables/` 的 CSV/Markdown 模板
- 随机种子固定；多 seed 时报告 mean±std

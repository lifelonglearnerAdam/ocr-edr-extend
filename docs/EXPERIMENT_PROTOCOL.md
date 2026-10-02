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

当前 `compare_official_results.py` 只报告已有官方 CDM/TEDS、归一化精确恢复和标明为 `proxy` 的 Good/Bad 分支指标。默认 proxy-Good 为初始单条 CDM/TEDS=1，不等于人工视觉一致性标签。`vis_fix` 保持空值，必须另做盲审或独立视觉验证；不能将渲染文件存在或代码哈希相同算作 VisFix。

前后配对固定初始参考匹配（页面、GT 索引、GT 位置、归一化 GT），不使用预测索引作为配对身份。匹配数量或身份变化时脚本报错，先建立固定匹配或明确另外报告全集官方指标，再计算配对分支指标。官方页面平均和单条样本平均分别报告。

Note 的 118 页及其所有裁剪仅作测试。独立训练数据需要页面/裁剪哈希阻断和近重复审核；`assert_trainable` 只是元数据校验，不能替代这些检查。模型 policy / judge 的输入禁止携带参考标注或官方分数。

## 5. 判定层消融

- A0：冻结 DocEDR verifier（原文）
- A1：纯 Jev
- A2：Jev 级联（高置信接受，低置信升级教师/DocEDR）
- A3：Jev + 确定性渲染/结构 diff（推荐，防 judge 分数不可靠）

同模型、同数据、同预算比较：保持原始输出、一次修复、不更新渲染的多轮修复、更新渲染的闭环。先建立推理基线，再引入 SFT/GRPO，分别报告新增训练的贡献。阈值、奖励权重和模型选择只用独立开发集决定，冻结后评估 Note 测试集。

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

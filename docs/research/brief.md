> Research notes from 2026-09-28. Literature status and cited results below do not imply implemented methods or independently reproduced results.

# 研究简报：OCR-EDR 方法普适化与 Agentic RL 优化

**日期**：2026-09-28
**深度**：standard
**状态**：Phase 1–2 完成，Phase 3 部分完成（文献与生态调研）

## 研究问题

如何把 OCR-EDR（Rendering-Aware Diagnosis and Repair）的闭环纠错范式，改造成：

1. **聚焦公式与表格**（正文纠错收益不明显、耗时高，仅作可选）
2. **Agentic RL 优化**多轮诊断-修改-重渲染-再判定
3. **小参数学生模型**（如 Qwen2B 级）可训练、可部署
4. **大模型蒸馏**（GPT / Claude 等教师）提供高质量轨迹与判定标签
5. **多主流 OCR 模型 × OmniDocBench** 上均有可复现提升（方法普适性）
6. **Jev 判定模型**作为诊断/奖励/级联首筛环节

## 范围边界

**In**
- 公式 / 表格 / 公式-表格混合区域的 OCR 后处理纠错
- OmniDocBench（及 OCRErrBench / UniMER-Test 公式子集）验证
- GRPO / Agentic RL 训练配方
- Jev 作为 verifier / reward / cascade judge
- 小模型蒸馏与多教师轨迹合成
- 共享代码仓库与实验配置管理

**Out（本轮不做）**
- 纯正文长文本纠错主线（会议结论：提升不明显且耗时）
- Monkey OCR Note 类别检测主线（并行项目，林之博）
- 整页 Layout 视觉一致性校验（讨论过但难度高，后续）

## 假设（待实验验证）

| 假设 | 验证方式 | 风险 |
|------|----------|------|
| H1 闭环 render-repair 对公式/表格增益大于正文 | 分模态 ablation | 正文也可少量保留作对照 |
| H2 Agentic RL（GRPO）优于纯 SFT 多轮 | 同数据同模型对比 | 轨迹奖励稀疏、方差大 |
| H3 Jev 可替代/级联降低 LLM verifier 成本 | 判定一致性 + 成本 | Jev 对多步推导/精美错误答案偏弱（见 Jev 文献） |
| H4 蒸馏教师轨迹可提升小模型修复率 | 蒸馏 vs 从头 SFT | 教师输出格式不一致 |
| H5 方法对多个 OCR 系统都有增益（普适） | OmniDocBench 多系统 Bad 子集 | 系统错误分布不同，可能过拟合单一 OCR |

## 关键事实（已核实）

### OCR-EDR / DocEDR（arXiv:2609.03445v1，WeChat Vision / Tencent）

- 任务：输入源图 `I`、可编辑 OCR 结果 `p`、渲染图 `R(p)`，闭环诊断并修复
- 动作空间：`inspect, diagnose_scope, localize, patch, global_patch, request_render, stop`
- 错误类：`invalid_output, global_mismatch, completeness, content, structure`
- 训练三段：Verifier SFT → Curriculum Repair SFT → GRPO（label-conditioned reward）
- 基座：Qwen3.5-9B；OCRErrBench 900 例（文/公式各半）
- 结果：诊断 Acc 94.78；错误输入修到视觉一致 86.23；DOCRcaseBench 公式 Case-F1 +30.99 vs DOCR-Inspector-7B；UniMER-Test Bad 子集公式 CDM 最高 +4.62
- 消融：无更新渲染的迭代反而变差；“迭代 + 更新渲染”才提升（ExactFix 82.17 / VisFix 86.23）

### 会议纪要（项目图片，2026-09 会）

- Agent RL 优先；公式与表格专项；缩小模型参数
- 构建 OmniDocBench 验证体系支撑发刊
- 引入 JEV 作为判断和奖励模型
- 10 月 19 日前完成初步实验与实施方案（@曾子墨 @刘英旭）
- Monkey OCR Note 类为并行线

### Jev（TypeSafe 决策模型，2026-09 爆发）

- 不生成文本，对候选标签/选项出概率；快、便宜、自带置信度
- 相对 LLM 裁判：便宜约 29–325 倍，延迟 30–220 倍更低；错误模式高度相关
- 适合：属性判断、打分、路由、级联首筛（confident accept / unsure escalate）
- 短板：多步推导验证、抗“包装精美的错误答案”弱于最强 LLM 裁判
- Visual Jev：共享视觉上下文的批量强制选择，适合 OCR 多问题判定
- 生态：GitHub 约 2000+ 项目；JEV-as-a-Judge、Jev-Mem 等

### OmniDocBench

- CVPR 2025，9 类文档源，19 layout 类 / 15 属性，多层级评测
- 主流 OCR/VLM 均在该榜报告；存在 v1.5 / v1.6 协议与 Hard 子集讨论（MinerU2.5-Pro）
- 相关：PaddleOCR-VL-1.5、MinerU2.5-Pro、DeepSeek-OCR、MonkeyOCR 等

### Agentic RL

- 从单步 MDP（RLHF/DPO）→ 时序 POMDP（工具、多步、稠密+稀疏奖励）
- GRPO 为当前主流（组相对优势，无 Critic，省显存）
- 与 OCR-EDR 闭环天然同构：状态 = (I, p_t, R_t, τ)，动作 = 诊断/编辑/渲染/停止

## 研究角度（Angles）

1. **A1 论文机制与可复现缺口** — OCR-EDR 方法、数据、训练配方、开源与否
2. **A2 Jev 作为 OCR 闭环判定/奖励** — 接口形态、置信度级联、视觉变体
3. **A3 OmniDocBench 多系统普适实验设计** — 基线、指标、Bad 子集、消融
4. **A4 Agentic RL / GRPO 训练配方** — 奖励设计、课程、小模型稳定训练
5. **A5 教师蒸馏与小模型** — 多教师轨迹、格式对齐、数据合成
6. **A6 反例与风险** — LLM-as-Judge 不可靠、过度修复、渲染等价误判

## 待补缺口

- [ ] OCR-EDR 是否开源 / 数据可得性（需查 GitHub / 作者页）
- [ ] Jev 公开 API / 权重 / Visual Jev 代码可复现路径
- [ ] OmniDocBench 官方评测脚本与多模型跑分入口
- [ ] “GPT Astra6 / Claude Opus5.5” 等教师的具体可用接口与成本（团队环境）
- [ ] 团队 GPU / 预算约束（会议提到鼓励用现有显卡）

## 成功标准

1. 公式 + 表格在 OmniDocBench 上，至少 2 个主流 OCR 基线在 Bad 子集有统计显著提升
2. 小模型（≤3B 级）可完成诊断-修复闭环，延迟与成本可接受
3. Jev 级联相对纯 LLM verifier 显著降本且准确率损失可控
4. 实验可在共享仓库一键复现（配置 + 脚本 + issue 跟踪）

> Research notes from 2026-09-28. Literature status and cited results below do not imply implemented methods or independently reproduced results.

# F2 — Jev 判定模型与 OmniDocBench 实验台

**检索日**：2026-09-28

## Jev（TypeSafe 决策模型）

一手/近一手来源：
1. arXiv:2609.29769 — Jev vs. LLMs as Rubric Judges（2026-09-24）
2. arXiv:2609.26550 — JEV-as-a-Judge: Accept When Confident, Escalate When Unsure（2026-09-22）
3. arXiv:2609.25845 — Visual Jev（2026-09-22）
4. arXiv:2609.30216 — Jev in the Wild（生态，2170 项目）
5. 微信公众号 PaperTody《彻底疯狂，21篇Jev扎堆上线》（二手综述，用于交叉印证）

### 机制

- **不做文本生成**；对强制选项 / 二分类 / 打分输出候选概率
- 置信度 `q` = 最高标签概率；可作级联闸门
- Visual Jev：同一图像共享视觉前缀，批量读问题后缀 → 适合 OCR 多准则判定

### 量化（来自上述论文 / 综述，需在本组复现时再核）

| 指标 | Jev | LLM 裁判（flash/GPT 级） |
|------|-----|-------------------------|
| 成本 | 约 0.36%–3.5% | 基准 |
| 延迟 | ~0.15s 中位 | ~1.9s 中位（综述数字） |
| 二值/有证据事实 | 与最强裁判差 <3pt | — |
| 多步推导 / 精美错误 | 落后可达 14–20pt | 更强 |
| 级联 | q≥0.9 仅升级 34%，准确率 ≈ 最强单裁判 | 成本约 47% |

### 接到 OCR-EDR 的可能形态

1. **判定层**：`Good/Bad`、错误类（5 类）、是否 rendering-equivalent
2. **奖励层**：GRPO 的 `s_V` / 过程事件（recover / regress）
3. **级联**：Jev 高置信直接接受；低置信升级到冻结 DocEDR 或教师 VLM
4. **风险**：不能单独信任 Jev 做需要多步符号推导的公式结构校验 → 必须与渲染一致性/符号 diff 结合（呼应 2607.13347：LLM-as-Judge 分数作为闭环优化信号不可靠）

## OmniDocBench

- 论文：arXiv:2412.07626（CVPR 2025），https://github.com/opendatalab/OmniDocBench
- 9 类文档源；19 layout 类；15 属性；端到端 / 任务级 / 属性级评测
- v1.5 / v1.6 协议演进（MinerU2.5-Pro 引入 Hard 子集并校正 element-matching）
- Real5-OmniDocBench：物理成像鲁棒性扩展（本组可暂不纳入主线）

### 建议实验矩阵（普适性）

| OCR 系统 | 公式 | 表格 | Bad 子集提升 |
|----------|------|------|--------------|
| PaddleOCR-VL / VL-1.5 | ✓ | ✓ | 目标 |
| MinerU / MinerU2.5 | ✓ | ✓ | 目标 |
| DeepSeek-OCR | ✓ | ✓ | 目标 |
| MonkeyOCR | ✓ | ✓ | 目标 |
| Qwen-VL 系 | ✓ | ✓ | 对照 |

指标建议：
- 公式：CDM / Case-F1 / ExactFix / VisFix
- 表格：TEDS / cell-F1
- 闭环：Preserve（防过修）、诊断 Acc / BAcc、渲染次数、轮次成本

## 风险文献

- arXiv:2607.13347 — LLM-as-a-Judge Scores Are Unreliable Optimization Signals in Closed-Loop Table Recognition
  → 闭环中必须有**确定性结构变化检测**，不能只靠 judge 分数

## 待核实

- [ ] Jev 官方 API / 本地权重获取方式与许可证
- [ ] Visual Jev 代码：https://github.com/guanxuyu-sv/Visual-Jev
- [ ] OmniDocBench 评测脚本对表格/公式的官方指标定义

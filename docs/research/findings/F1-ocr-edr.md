> Research notes from 2026-09-28. Literature status and cited results below do not imply implemented methods or independently reproduced results.

# F1 — OCR-EDR / DocEDR 核心机制

**来源**：https://arxiv.org/html/2609.03445v1 （arXiv:2609.03445v1, 2026-09-03）
**置信度**：高（原文 HTML）
**检索日**：2026-09-28

## 任务定义

输入：文档区域图 `I`、OCR 预测 `p`、渲染图 `R(p)`
输出：修正后的 `p̂`，与源图内容与视觉结构一致（允许 rendering-equivalent 等价编码）

闭环（Algorithm 1）：
1. 诊断是否一致（Good 则保留，含等价编码）
2. 分类 + 定位错误
3. 局部 `patch` 或全局 `global_patch`
4. 可选 `request_render` 更新渲染
5. 再评估直至 `stop` 或预算耗尽

## 错误分类（表 1）

| 类 | 范围 | 修复操作 |
|----|------|----------|
| invalid_output | Global | global_rewrite |
| global_mismatch | Global | global_rewrite |
| completeness | Local | insert/delete |
| content | Local | insert/delete/replace |
| structure | Local | insert/delete/replace |

策略动作：`inspect, diagnose_scope, localize, patch, global_patch, request_render, stop`

## 训练

1. **Verifier SFT**：学习 (I,p,R) → 有效性 + 错误集合（类/位置/建议修正）
2. **Curriculum Repair SFT** 三阶段：
   - 单轮直接修复（无渲染）
   - 学会选择 request_render vs stop
   - 真实渲染反馈 + 多轮残差修复 + 失败恢复
3. **GRPO**：label-conditioned reward
   - Good：保留 + 惩罚多余编辑/渲染/轮次
   - Bad：终局质量（参考一致 + 冻结 Verifier）+ 过程进度 − 交互成本

关键消融（表 E2）：仅迭代不更新渲染 → ExactFix 掉到 66.82；迭代+更新渲染 → 82.17 / VisFix 86.23

## 对本组改造的含义

| 原文 | 本组方向 |
|------|----------|
| 文/公式混合 | **只做公式+表格**（正文收益小、耗时大） |
| Verifier 为冻结 DocEDR | **可替换为 Jev / Jev 级联** |
| GRPO 已用 | 强化为 **Agentic RL 主线**（工具、多步、课程） |
| Qwen3.5-9B | **Qwen2B 级小模型** + 大教师蒸馏 |
| 单篇论文内多系统 Bad 子集 | **OmniDocBench 上多主流系统系统性验证普适性** |

## 引用

- Zhao et al., "OCR-EDR: Rendering-Aware Diagnosis and Repair for Closed-Loop OCR Improvement", arXiv:2609.03445v1, 2026.

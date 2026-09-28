# 研究简报摘要

完整版见 `research/ocr-edr-extend/brief.md`（本仓库 docs 同步副本）。

## 一句话

把 OCR-EDR 的「诊断-修复-重渲染」闭环，改造成 **公式/表格专项 + Agentic RL + 小模型 + Jev 判定** 的普适后处理，并在 OmniDocBench 多系统上验证。

## 论文机制（OCR-EDR）

- 输入 `(I, p, R(p))`，动作 `inspect/diagnose/localize/patch/global_patch/request_render/stop`
- 训练：Verifier SFT → Curriculum Repair SFT → GRPO
- 关键发现：**必须「迭代 + 更新渲染」**，否则过修

## 本组改造点

1. 模态收窄到公式 + 表格
2. Jev 替代/级联 verifier 与 reward
3. Agentic RL 强化多轮工具式交互
4. Qwen2B 级学生 + 大模型蒸馏
5. OmniDocBench 多主流 OCR 普适验证

## 风险

- Jev 对多步推导与抗干扰弱 → 必须叠加渲染/结构确定性信号
- 小模型易过修 → Preserve 奖励与 Good/Bad 分支
- 多系统错误分布不同 → 按系统分层报告 Bad 子集

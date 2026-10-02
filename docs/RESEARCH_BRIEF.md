# 研究简报摘要

本文件记录研究方向；实验状态与可运行命令见 [README](../README.md)。

## 一句话

把 OCR-EDR 的「诊断-修复-重渲染」闭环，改造成 **公式/表格专项 + Agentic RL + 小模型 + Jev 判定** 的普适后处理，并在 OmniDocBench 多系统上验证。

## 论文机制（OCR-EDR）

- 输入 `(I, p, R(p))`，动作 `inspect/diagnose/localize/patch/global_patch/request_render/stop`
- 训练：Verifier SFT → Curriculum Repair SFT → GRPO
- 关键发现：**必须「迭代 + 更新渲染」**，否则过修

## 本组计划验证的改造点

1. 模态收窄到公式 + 表格
2. Jev 替代/级联 verifier 与 reward
3. Agentic RL 强化多轮工具式交互
4. Qwen2B 级学生 + 大模型蒸馏
5. OmniDocBench 多主流 OCR 普适验证

## 可检验的研究问题

1. 在相同学生模型与推理预算下，更新渲染的闭环是否优于一次修复与不更新渲染的多轮修复？
2. 公式与表格专项训练能否提高 Bad 子集修复率，同时维持 Good 子集 Preserve？
3. Jev 级联能否降低判断成本；误改和升级率是否可接受？
4. 在独立训练数据上训练的策略能否迁移到未参与训练的 OCR 系统？

这些是待实验回答的问题。已有 Note 基线和合成状态机检查不能证明方法有效或论文创新性。

## 风险

- Jev 对多步推导与抗干扰弱 → 必须叠加渲染/结构确定性信号
- 小模型易过修 → Preserve 奖励与 Good/Bad 分支
- 多系统错误分布不同 → 按系统分层报告 Bad 子集

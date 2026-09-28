# OCR-EDR-Extend

OCR-EDR 闭环纠错方法的 **公式 / 表格专项** 普适化与 **Agentic RL** 优化协作仓库。

> 基础论文：[OCR-EDR: Rendering-Aware Diagnosis and Repair for Closed-Loop OCR Improvement](https://arxiv.org/abs/2609.03445) (arXiv:2609.03445)

## 目标（本组研究方向）

1. **聚焦公式与表格**：正文纠错收益不明显且耗时，不作为主线。
2. **Agentic RL**：用 GRPO 等方法优化多轮「诊断 → 定位 → 修改 → 重渲染 → 再判定 → 停止」。
3. **小参数学生模型**：以 Qwen2B 级模型为主进行训练与部署。
4. **大模型蒸馏**：从 GPT / Claude 等教师模型合成高质量诊断与修复轨迹。
5. **多系统普适性**：在 **OmniDocBench** 上验证对多个主流 OCR / 文档解析模型均有提升。
6. **Jev 判定层**：引入 Jev（决策模型）作为判断 / 奖励 / 级联首筛，降低 verifier 成本。

## 快速开始

```bash
# 环境（示例）
conda create -n ocredrenv python=3.11 -y
conda activate ocredrenv
pip install -r requirements.txt

# 数据与评测
python scripts/download_data.py --bench omnidocbench
python scripts/eval_omnidocbench.py --config configs/eval/omnidocbench_formula_table.yaml

# 训练（示例）
python scripts/train_grpo.py --config configs/train/grpo_qwen2b_formula_table.yaml
```

## 目录结构

```
ocr-edr-extend/
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── experiment.md
│   │   ├── bug.md
│   │   └── data-issue.md
│   └── PULL_REQUEST_TEMPLATE.md
├── configs/
│   ├── train/          # SFT / GRPO / 蒸馏
│   ├── eval/           # OmniDocBench / OCRErrBench / UniMER
│   └── judge/          # Jev / 级联 verifier
├── docs/
│   ├── ROADMAP.md
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── RESEARCH_BRIEF.md
│   └── MEETING_NOTES.md
├── scripts/            # 一键复现入口
├── src/
│   ├── diagnosis/      # 诊断与定位
│   ├── repair/         # patch / global_patch
│   ├── render/         # 渲染与等价判定
│   ├── judge/          # Jev / verifier / reward
│   └── train/          # SFT / GRPO / distill
├── data/               # 不进 Git，放 LFS 或网盘索引
└── experiments/        # 运行日志、结果表（可选 LFS）
```

## 实验规范

见 [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)。提交结果必须包含：

- 配置文件哈希 / 路径
- 随机种子
- 指标定义（公式 CDM / Case-F1；表格 TEDS；闭环 Preserve / ExactFix / VisFix）
- 基线与 Bad 子集划分说明

## 贡献

见 [CONTRIBUTING.md](CONTRIBUTING.md)。用 Issue 跟踪：实验任务、数据问题、bug、复现失败。

## 时间节点（来自组会）

| 日期 | 事项 |
|------|------|
| 国庆假期 | 组会暂停，推进实验 |
| **10 月 19 日前** | OCR 纠错初步实验 + 实施方案（@曾子墨 @刘英旭） |
| 假期后 | Doc Cloud 论文分享（@黄城楷 @林之博） |
| 并行 | Monkey OCR Note 类修正（@林之博） |

## 引用

```bibtex
@article{zhao2026ocredr,
  title   = {OCR-EDR: Rendering-Aware Diagnosis and Repair for Closed-Loop OCR Improvement},
  author  = {Zhao, Linnan and Liu, Kang and Yu, Hao and Zhan, Jiabo and Sun, Chong and Li, Chen},
  journal = {arXiv preprint arXiv:2609.03445},
  year    = {2026}
}
```

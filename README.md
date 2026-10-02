# OCR-EDR-Extend

OCR-EDR 闭环纠错方法的 **公式 / 表格专项** 普适化与 **Agentic RL** 优化协作仓库。

> 基础论文：[OCR-EDR: Rendering-Aware Diagnosis and Repair for Closed-Loop OCR Improvement](https://arxiv.org/abs/2609.03445) (arXiv:2609.03445)

本项目的本地工作区是 `~/d/recovered_final/optimization/ocr-edr-team`。研究方向依据该工作区的独立研究简报推进；MonkeyOCR Note 表格/公式复核属于另一个项目。MonkeyOCR 可作为多系统实验中的一个基线。

完整资料：[研究简报](docs/research/brief.md)、[OCR-EDR 机制](docs/research/findings/F1-ocr-edr.md)、[Jev 与评测](docs/research/findings/F2-jev-omnidocbench.md)。引用数字来自研究记录，尚未在本项目独立复现。

## 目标（本组研究方向）

1. **聚焦公式与表格**：正文纠错收益不明显且耗时，不作为主线。
2. **Agentic RL**：用 GRPO 等方法优化多轮「诊断 → 定位 → 修改 → 重渲染 → 再判定 → 停止」。
3. **小参数学生模型**：以 Qwen2B 级模型为主进行训练与部署。
4. **大模型蒸馏**：从 GPT / Claude 等教师模型合成高质量诊断与修复轨迹。
5. **多系统普适性**：在 **OmniDocBench** 上验证对多个主流 OCR / 文档解析模型均有提升。
6. **Jev 判定层**：引入 Jev（决策模型）作为判断 / 奖励 / 级联首筛，降低 verifier 成本。

## 当前可运行部分

- 通用状态机：`inspect / diagnose_scope / localize / patch / global_patch / request_render / stop`，包括渲染过时、候选回退和预算约束。
- 独立区域清单验证：公式/表格输入、测试标记、来源图片和训练/留出集的页面与精确图片重复检查。
- PNG 渲染程序适配器，以及调用官方 OmniDocBench 的隔离评测入口。
- 固定参考匹配的前后比较，分别报告页面平均、样本平均及明确标记的 Preserve/Good/Bad 代理指标。
- CPU 检查、合成样例与服务器运行说明。
- 真实 Qwen2-VL-2B 推理提案接口、受限 MathText 公式渲染、参考隔离的受控实验与成本记录。实验条件包括不修改、源图单轮、带渲染单轮、更新/过时渲染双轮。

当前已接入未训练 2B 模型的独立公式提案实验；完整闭环的策略/视觉 judge、完整 TeX/表格渲染器、Jev、蒸馏和 SFT/GRPO 尚待实现。受控实验用于检查模型行为，不能视作真实 OCR 基准提升。先看 [假设与决策](docs/research/HYPOTHESES.md)、[公式实验协议](docs/research/FORMULA_PILOT.md)、[首轮真实模型结果](docs/research/RESULTS_20261002.md) 和 [实施计划](docs/EXPERIMENT_PLAN.md)。

## 快速开始（Python ≥3.10，CPU）

```bash
python -m pip install -r requirements-core.txt
python -m unittest discover -s tests -v
python scripts/demo_loop.py
python scripts/validate_manifest.py --manifest examples/synthetic/regions.jsonl

# 官方数据和本项目预测准备好后，先生成配置与命令
python scripts/eval_omnidocbench.py \
  --config configs/eval/omnidocbench_formula_table.yaml \
  --baseline paddleocr_vl \
  --official-repo /path/to/OmniDocBench \
  --gt /path/to/OmniDocBench.json \
  --pred-dir /path/to/paddleocr_predictions \
  --dry-run
```

训练依赖保留在 `requirements.txt`，训练 YAML 是待验证配方，当前没有 `train_grpo.py` 入口。格式见 [数据协议](docs/DATA_PROTOCOL.md)，远端路径见 [服务器说明](docs/SERVER_RUNBOOK.md)。

## 目录结构

```
ocr-edr-extend/
├── README.md
├── CONTRIBUTING.md
├── pyproject.toml
├── requirements-core.txt
├── requirements-dev.txt
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
│   └── ocr_edr/        # 闭环、渲染接口、清单与官方结果对比
├── examples/synthetic/ # 合成输入，仅用于软件验证
├── tests/              # CPU 与 CLI 检查
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

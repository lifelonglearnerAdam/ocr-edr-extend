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

10月7日完成[第二具名公式解析器开发诊断](docs/research/SECOND_PARSER_RESULTS_20261007.md)：固定SFT在LaTeX-OCR原生候选上的core-CDM由0.8994到0.9899，6个完整/4个部分修复；同源此前Nougat结果仍保留负向结论，不能称独立迁移。另完成[PubTabNet文章隔离四角色数据](docs/research/TABLE_DATA_RESULTS_20261007.md)与训练/model-dev局部JSON目标，发现并排除了发布split文章重叠；标注缺行风险已标记，尚未运行表格训练或锁定评测。

10 月 6 日已将五个方向写成[统一研究设计](docs/research/INTEGRATED_RESEARCH_20261006.md)，含分层消融、独立验收/校准与parser留出要求；当前仍是待验证机制。首轮两个192-step公式LoRA和三模型96-dev输入推断已完成，见[SFT结果](docs/research/SFT_SCREEN_RESULTS_20261006.md)：受控错误全修复到core-CDM=1，原生metric不匹配没有修复且发生一次可见结构退化。当前官方Jev是文本-only，视觉证据必须另生成并计费，见[接口核查](docs/research/JEV_INTERFACE_CHECK_20261006.md)。尚无完整五方向系统或论文效果结论。

- 通用状态机：`inspect / diagnose_scope / localize / patch / global_patch / request_render / stop`，包括渲染过时、候选回退和预算约束。
- 独立区域清单验证：公式/表格输入、测试标记、来源图片和训练/留出集的页面与精确图片重复检查。
- PNG 渲染程序适配器，以及调用官方 OmniDocBench 的隔离评测入口。
- 固定参考匹配的前后比较，分别报告页面平均、样本平均及明确标记的 Preserve/Good/Bad 代理指标。
- CPU 检查、合成样例与服务器运行说明。
- 真实 Qwen2-VL-2B 推理提案接口、MathText 与完整 Tectonic 公式渲染、参考隔离实验、图像顺序/角色标签消融及实际 token/CPU 生成成本记录。
- 独立 UniMER 开发样本、Nougat-LaTeX 原生输出、图像单独识别对照；全部样本保留，原有精确光栅代理保持独立标注；已新增固定配对的官方 core CDM 离线重评、参考自检与哈希核验。
- WeasyPrint 中文/合并单元格表格渲染与单步 JSON 编辑；固定配对的官方 TEDS/TEDS-S 离线评测，分别报告修复、回退、退化、页面平均及成本。官方源码先与固定 Git blob 核验。

当前已接入完整 TeX/表格渲染和独立评测，包括官方 core CDM 重评（尚非端到端基准评测）。10 月 6 日从已核验的 UniMER-1M 训练包冻结了 128 个 train 源和 32 个 dev 源，生成保持/受控错误及全部原生 Nougat 输出，共 384/96 条监督记录；数据准备阶段没有训练更新。随后在 RTX 4090 上完成了两个训练样例的一步 LoRA 可行性检查，未保存 checkpoint，该smoke之后已完成第一轮直接目标SFT开发筛查，结果与限制见上文。完整闭环的学习策略/视觉 judge、Jev、教师轨迹蒸馏和 Agentic GRPO 尚待实现；表格 demo 的原始解析器身份未公开，不能据此声称具名解析器迁移。未训练 2B 的开发结果包含失败与退化，不能视作基准提升。先看 [假设与决策](docs/research/HYPOTHESES.md)、[10 月 3 日公式结果](docs/research/RESULTS_20261003.md)、[10 月 4 日对照与表格结果](docs/research/RESULTS_20261004.md)、[10 月 6 日监督数据准备](docs/research/RESULTS_20261006.md)、[4090 单卡训练检查](docs/research/GPU_READINESS_20261006.md)、[相关工作与新颖性检查](docs/research/PRIOR_ART_20261003.md) 和 [实施计划](docs/EXPERIMENT_PLAN.md)。

## 环境（Windows / Ubuntu）

| 平台 | 用途 | 解释器 |
|------|------|--------|
| Windows 工作机 | 文档、轻量脚本、代码审阅 | 系统 Python 或 **Windows** venv：`.\.venv\Scripts\python.exe` |
| Ubuntu / GPU 机 | 模型推理、完整渲染器、官方评测、训练 | POSIX venv：`bin/python`（见 [SERVER_RUNBOOK](docs/SERVER_RUNBOOK.md)） |

Linux 下创建的 `.venv`（`bin/`+`lib/`）**不能**在 Windows 直接用；Windows 请本地重建 venv。CI（3.10 / 3.12）是跨平台质量闸门。

## 快速开始（Python ≥3.10，CPU）

**Windows**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\demo_loop.py
.\.venv\Scripts\python.exe scripts\validate_manifest.py --manifest examples\synthetic\regions.jsonl
```

**Ubuntu / macOS**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-core.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/demo_loop.py
.venv/bin/python scripts/validate_manifest.py --manifest examples/synthetic/regions.jsonl
```

官方数据和本项目预测准备好后，先生成配置与命令：

```bash
.venv/bin/python scripts/eval_omnidocbench.py \
  --config configs/eval/omnidocbench_formula_table.yaml \
  --baseline paddleocr_vl \
  --official-repo /path/to/OmniDocBench \
  --gt /path/to/OmniDocBench.json \
  --pred-dir /path/to/paddleocr_predictions \
  --dry-run
```

公式开发筛查入口为 `scripts/train_formula_sft.py`，冻结配置 `configs/train/sft_formula_screen.yaml`；先核验本项目train/dev及模型哈希，再运行两个固定终态对照。`scripts/run_sft_screen.py`只读取参考隔离输入，`adapt_sft_screen.py`固定syntax-only输出后才交官方离线评估。完整Agentic GRPO配置仍是待验证配方，当前没有 `train_grpo.py` 入口。格式见 [数据协议](docs/DATA_PROTOCOL.md)，远端路径见 [服务器说明](docs/SERVER_RUNBOOK.md)。

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

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

## 当前可运行部分

- `src/ocr_edr/loop.py`：独立的编辑 / 重渲染 / 再判断状态机；编辑后清空旧渲染与判断，未验证的修改回退到初始预测。
- `src/ocr_edr/render.py`：调用团队渲染程序的 PNG 适配器；每次渲染使用独立目录。
- `scripts/prepare_note_eval.py`：导入已有 Note 诊断数据与官方匹配结果，保留 `test_only` 标记与图片哈希。
- `scripts/eval_omnidocbench.py`：生成官方配置，在独立运行目录调用上游评测程序。
- `scripts/compare_official_results.py`：固定参考匹配的前后对比；区分官方页面平均与样本平均。
- `tests/`：CPU 单元与命令行集成检查；GitHub Actions 检查 Python 3.10 / 3.12。

**研究状态**：当前是可运行的实验基建。模型 policy / judge、真实公式/表格渲染程序、Jev、教师蒸馏、SFT 和 GRPO 训练尚待接入。`demo_loop.py` 使用合成输入和脚本化判断，只验证状态机。现有基线报告不是修复效果，也不是新模型结果。

## 快速开始（Python ≥3.10，CPU）

```bash
python -m pip install -r requirements-core.txt
python -m unittest discover -s tests -v
python scripts/demo_loop.py

# 本地只读服务器镜像；从仓库根目录运行
python scripts/prepare_note_eval.py \
  --source-root ../../server_mirror \
  --output data/raw/note_eval

# 官方评测：先生成配置与命令；确认数据路径后去掉 --dry-run
python scripts/eval_omnidocbench.py \
  --config configs/eval/omnidocbench_formula_table.yaml \
  --official-repo ../omnidocbench-eval \
  --gt ../../server_mirror/OmniDocBench_note/OmniDocBench_note.json \
  --pred-dir ../../server_mirror/baselines/monkeyocrv2_b_note/markdowns \
  --dry-run
```

官方 CDM/TEDS 的运行环境由上游仓库提供；CPU 基建不需要安装 PyTorch 或下载权重。训练配置是待实验验证的配方，当前没有 `train_grpo.py` 训练入口。数据获取说明见 [data/README.md](data/README.md)，服务器操作见 [docs/SERVER_RUNBOOK.md](docs/SERVER_RUNBOOK.md)。

## 已有 Note 基线核对

| 项目 | 数量 / 分数 |
|------|-------------|
| Note 测试页面 | 118 |
| 诊断 GT 元素 | 25 公式 + 37 表格 |
| 官方匹配评测记录 | 28 公式 + 37 表格 |
| 可明确一对一对齐的诊断元素 | 23 公式 + 37 表格；2 公式保持未对齐 |
| Formula CDM（官方页面平均） | 65.05273810% |
| Table TEDS（官方页面平均） | 73.12665319% |

原始记录与图片留在服务器或本地 `data/raw/`；Git 只保存 [核对摘要](experiments/tables/note_baseline_audit.json)。公式样本平均是 68.69642857%，表格样本平均是 72.09616353%，不能替换上表的官方页面平均。详见 [docs/BASELINE_AUDIT.md](docs/BASELINE_AUDIT.md)。

## 目录结构

```
ocr-edr-extend/
├── README.md
├── CONTRIBUTING.md
├── pyproject.toml
├── requirements-core.txt
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
│   └── ocr_edr/        # 状态机、渲染适配、数据导入与指标对比
├── tests/              # 不使用真实测试集的合成检查
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

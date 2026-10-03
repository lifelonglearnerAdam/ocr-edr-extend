# 数据目录（不进 Git）

请在此说明：

| 数据集 | 获取方式 | 存放路径 | 许可 |
|--------|----------|----------|------|
| OmniDocBench | https://github.com/opendatalab/OmniDocBench | `data/omnidocbench/` | 见官方 |
| OCRErrBench | 待确认作者是否开源 | `data/ocerrbench/` | 待确认 |
| UniMER-Test | 官方仓库 | `data/unimer/` | 见官方 |
| 训练轨迹（蒸馏/GRPO） | 团队生成 | `data/processed/` | 内部 |
| 教师模型轨迹（GPT/Claude 等） | **API 额度未申请**，暂缓 | `data/teacher_trajs/` | 视厂商条款 |

**禁止**将评测集原文、教师模型密钥、未授权权重提交进 Git。

## 独立项目的数据约定

本项目使用自己的区域 JSONL、OCR 预测与运行记录。格式见 `docs/DATA_PROTOCOL.md`，CPU 样例在 `examples/synthetic/`。原始训练数据、冻结评测集和教师轨迹尚待准备；不同项目的复核记录与既有分数不能代替本项目的实验。

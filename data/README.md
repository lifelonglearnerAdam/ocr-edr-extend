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

## 当前可用输入

已有 Note 诊断与基线的本地只读镜像位于本仓库外的 `../../server_mirror`。为避免服务器共享文件的账号权限问题，验证过的评测镜像另存于 `/data/yxliu/datasets/ocr-edr-note-eval-20261002`。使用 `scripts/prepare_note_eval.py` 导入，生成 `data/raw/note_eval/` 下的元素记录、官方匹配记录、图片哈希阻断清单和摘要。所有导入记录均标记 `split=test`、`test_only=true`。

`scripts/download_data.py` 是数据获取说明入口，不提供自动下载器。先按官方许可获取数据，再将路径交给评测入口。独立训练数据和教师轨迹尚未准备好。

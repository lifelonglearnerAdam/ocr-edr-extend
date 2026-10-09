# 数据目录（不进 Git）

请在此说明：

| 数据集 | 获取方式 | 存放路径 | 许可 |
|--------|----------|----------|------|
| OmniDocBench | https://github.com/opendatalab/OmniDocBench | `data/omnidocbench/` | 见官方 |
| OCRErrBench | 待确认作者是否开源 | `data/ocerrbench/` | 待确认 |
| UniMER-Test | 官方仓库 | `data/unimer/` | 见官方 |
| UniMER-1M 小规模 train/dev | 已校验的官方训练包，固定版本与图片/标注索引 | `data/processed/unimer-supervision-20261006/` | 数据卡 Apache-2.0；另有 HME100K 手动获取说明 |
| PubTabNet 四角色受控表格 | IBM原仓库所链镜像，按PMC文章重新隔离 | `data/processed/pubtabnet-four-roles-20261007/` | 原标注CDLA-Permissive-1.0；图片遵守PMC文章许可；镜像卡Sharing标签冲突保留 |
| 训练轨迹（蒸馏/GRPO） | 团队生成 | `data/processed/` | 内部 |
| 教师模型轨迹（GPT/Claude 等） | **API 额度未申请**，暂缓 | `data/teacher_trajs/` | 视厂商条款 |

**禁止**将评测集原文、教师模型密钥、未授权权重提交进 Git。

## 独立项目的数据约定

本项目使用自己的区域 JSONL、OCR 预测与运行记录。格式见 `docs/DATA_PROTOCOL.md`，CPU 样例在 `examples/synthetic/`。已准备小规模公式 train/dev 与直接标注监督记录，正式冻结评测集、表格训练数据和教师轨迹尚待准备；不同项目的复核记录与既有分数不能代替本项目的实验。

10 月 6 日的数据含 128 个 train 源、32 个 dev 源，最终 384/96 条监督记录。全量 UniMER-Test 的 23,757 张图片及索引标注进入排除检查。原生输入文件不含参考；SFT 文件包含目标，禁止交给推理接口。该压缩包不提供原始文档 ID，图片/公式去重不能等同于原始文档级独立。详见 `docs/research/SUPERVISION_PROTOCOL_20261006.md` 和对应结果报告。

10月7日的表格四角色按文章组为train128、model-dev32、gate-calibration32、locked64，受控案例410/103/102/213。全发布val/test文章从train/model-dev排除，test文章亦不能进入calibration/locked；旧demo页未进入训练。输入只含当前HTML和图像identity，标签/地址target/原bbox另存。p0002发布参考缺原图底部行已标记，table训练准入尚未完成；锁定/校准图像未人工查看或用于效果选择。详见 `docs/research/PUBTABNET_PREPARATION_20261007.md`。

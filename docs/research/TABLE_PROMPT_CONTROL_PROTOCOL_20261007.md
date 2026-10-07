# 表格动作提示对照追加协议

2026-10-07，完整literal-example开发推断前冻结。本对照在看到描述式提示的103例协议失败后制定，属于后续探索，不是事前确认性检验。原始调用、适配器、指标和源码归档保持不变。

## 动机与训练源检查

描述式提示得到0/103可执行动作，60次token cap。为了区分动作格式学习与视觉修复，对固定首个准入训练源p0001的全部3个既有变体，按同一顺序运行两个提示：原字段描述，以及在原提示末尾追加四个完整JSON响应形状。该检查只读train五字段输入，不读参考、target action或正确位置；1个独立源不能支持质量/泛化结论。

实际结果：描述式提示0/3合法动作、3次cap、576输出token、100.245860秒生成；显式示例3/3合法动作、18输出token、24.069340秒生成，**全部为stop，没有编辑**。两个条件输入token为3,526与3,796，增加90 token/调用。这只表明在该训练源上，提示足以改善协议遵循；不证明修复错误。

## 固定完整开发对照

- 原model-dev的全部103案例/32源，原source/HTML/hash不变；不按旧输出质量选择案例。
- 相同Qwen基座revision、CPU bfloat16/SDPA、4线程、seed20261007、greedy、192新token和图像bounds。
- 唯一有意变化为附加`LITERAL_ACTION_EXAMPLES`；在run receipt中记录`prompt_format=literal_examples`，并存完整prompt、源码和实际token成本。
- 四个JSON对象仅为响应形状示例，索引0与占位文本明确标为placeholder；不含当前正确动作/位置/答案。没有从参考填入任何值。
- 适配器仍要求一个准确schema、无重复key的对象；cap、格式、动作或渲染失败回退，不将`stop:true`改写为合法stop，不作重试或模型输出猜测性修复。
- 推断完成后再执行官方TEDS/TEDS-S及完整成本统计。报告合法动作、stop、实际修改、Bad修复和Good退化，不能只报告格式通过率。

此次提示对照单独评分，不通过放宽SFT配对检查把不同提示的base/adapter混在一起。后续base/all/no-explicit-preservation公平比较必须使用同一`--prompt-format literal_examples`和相同生成源码/runtime；训练数据和原两臂381步配方不因此改写。原描述式base保留为独立负向对照。

```bash
.venv-pilot/bin/python scripts/run_table_sft_screen.py \
  --dataset data/processed/pubtabnet-four-roles-20261007 \
  --config configs/train/sft_table_screen.yaml \
  --model-path data/raw/model-cache/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c \
  --model-receipt experiments/runs/server-selection-20261006/model-receipt.json \
  --condition base --device cpu --prompt-format literal_examples \
  --output experiments/runs/table-sft-screen-20261007/inference/base-literal-examples
```

该命令不会优化模型。校准/锁定图像继续未用，表格SFT待实际可用GPU。原生parser接口的PubTabNet训练来源限制仍适用，不能将本次受控开发集称为原生或独立未见来源评测。

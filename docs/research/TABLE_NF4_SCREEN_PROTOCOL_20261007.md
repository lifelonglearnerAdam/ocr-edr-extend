# 独立NF4表格提议能力与保持消融

2026-10-07，在完整训练和该条件开发推断前固定。现有校园GPU占用时，使用本地RTX5070 Laptop GPU启动独立精度条件；不把它改名为原BF16实验，不将两个条件的基座/adapter混配。

## 准入证据与范围

[本地可行性协议](TABLE_NF4_READINESS_PROTOCOL_20261007.md)的原全logits NF4检查在首样本OOM、0步；投影loss版本完成两个预定内存边界样本、112个adapter张量改变、冻结梯度为0、三组冻结权重样本不变，峰值allocated3.7803 GiB。其同进程重载前显存门槛失败，原failed receipt保留。随后独立进程重载112个adapter张量逐值一致，peak allocated2.2946 GiB。只有两份相互绑定的训练与独立重载证据共同支持此可行性判断，不把failed receipt覆写成completed。

临时两步adapter不用于完整训练初始化。首次OOM的协议、config与源码保持原样。主张仅限硬件可执行性，无质量收益。

## 完整对照

使用`configs/train/sft_table_nf4_screen_20261007.yaml`，Qwen2-VL-2B固定revision、127篇PMC文档/406条准入训练监督、32文档/103条model-dev。p0002整族已排除，calibration/locked不参与loss、checkpoint选择或本轮推断。

`all`与`no_explicit_preservation`分别从原始NF4 base初始化；语言q/v LoRA r4/alpha8、dropout0，lr1e-4、AdamW weight_decay0、clip1、seed20261007、381步×accumulation4。每臂1524条暴露、每文档12次；原文档均衡schedule不变。分别为24181/30406监督token，实际修复变体暴露量不同，不能称同token预算的纯保持目标因果效应。

仅词表projection省略ignored标签对应位置；完整图像、prompt、prefix和transformer上下文均保留。训练image bounds100352–200704、max4096，不静默截断。NF4 double quant、BF16量化matmul、非四位参数经PEFT转FP32；视觉模块若被量化则明确记录，优化器仍只含语言LoRA。最少7 GiB初始空闲、无其他compute作业、PyTorch allocator总量80%，不降低既有BF16门槛。

## 推断和评估

三个条件`base/all/no_explicit_preservation`均在相同本机、torch2.8.0+cu128、Transformers4.57.1、PEFT0.17.1、bitsandbytes0.48.1、Pillow11.3.0及相同NF4/FP32混合层配置执行；不能与此前CPU BF16 base直接配对。

全部103条开发输入只读取源图、当前HTML、由当前HTML产生的DOM地址；三个条件均用固定literal JSON examples、greedy、max_new_tokens192。生成只保留末位置lm_head logits，完整transformer输入与cache未删减；先以微型Qwen生成序列精确一致测试核验。该优化改变临时张量量，不改变greedy使用的末位置分数，实际质量与耗时仍以运行记录为准。

先冻结全部raw调用，之后离线严格JSON/cap/render回退与官方对称TEDS/TEDS-S。全样本、逐原始文档、按preservation/controlled类型报告修复、退化、动作地址与完整动作匹配以及token/时延/显存。最多只作单seed已检查开发源的可行性/能力结论，不选best checkpoint，不因指标差而重训或挑样本。

此阶段为H1/提议能力与保持对照提供证据。现有12条交互教师轨迹不进入本次optimizer；H4蒸馏需要另行匹配表示/预算。独立judge/Jev、闭环证据控制、RL和未知parser/锁定评测仍待完成。训练完成或loss下降均不等于研究目标完成。

统计附录已在本条件开发推断开始前固定：[文档级配对bootstrap、风险/成本及全修改源审计](TABLE_NF4_ANALYSIS_PROTOCOL_20261007.md)。完整四臂离线evaluation冻结后，执行：

```bash
.venv-pilot/bin/python scripts/summarize_table_sft_screen.py \
  --evaluation-dir experiments/runs/table-nf4-screen-20261007/evaluation \
  --output experiments/runs/table-nf4-screen-20261007/statistics
```

该统计工具不替换原评估器，不改raw模型输出；拒绝少于103例/32文档或缺臂、hash漂移的比较。区间是单seed已检查开发集的描述，不做确认性或低风险保证。

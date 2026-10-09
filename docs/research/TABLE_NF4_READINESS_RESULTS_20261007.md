# 本地NF4训练：显存失败、投影loss与独立重载

2026-10-07。本地RTX5070 Laptop GPU可在当前边界样本上完成两步NF4语言LoRA优化，并在独立新进程逐值重载112个adapter张量。原始失败全部保留。**这证明有限的训练可执行性，不证明纠错质量、完整训练稳定性或量化优于BF16。**

## 现场条件和依赖

校园8×3090机器在检查时全部有作业；另一台凭据不被接受，第三台无路由。本机名义8 GiB GPU没有其他compute作业。量化分支要求初始化前至少7 GiB空闲，PyTorch allocator上限80%；CUDA报告可用总量7.527 GiB，故本次allocator上限约6.022 GiB，保留桌面余量但不能保证系统级排他预留。

保持原PyTorch2.8.0+cu128、Transformers4.57.1、Pillow11.3.0，PEFT0.17.1/bitsandbytes0.48.1从独立包目录加载，wheel与PyPI发布SHA256一致，`pip check`通过。bitsandbytes固定版本的官方构建矩阵包含CUDA12.8/sm120；读取范围/URL/访问日期见[协议](TABLE_NF4_READINESS_PROTOCOL_20261007.md)。没有把现有BF16默认16 GiB门槛降低为7 GiB。

NF4 double quant、BF16矩阵计算、PEFT将403个非四位参数tensor转FP32；326个Linear4bit模块中130个位于视觉部分。视觉权重仍被冻结，只有语言q/v LoRA训练：544768参数。这里的“量化学生”与此前CPU BF16推断条件有明确差别。

## 预定两个边界样本

只用已冻结406条训练预检查选择最大处理后序列`p0055-extra_row`和最大视觉grid `p0079-cell_perturbation`，未按loss或正确性挑样本。输入共3676处理后token、34监督token；图像范围100352–200704、token cap4096，不缩图、不截断。两步可行性adapter不进入完整训练初始化。

| 运行 | optimizer步 | 观察 | 最终回执 |
|---|---:|---|---|
| 原NF4全位置logits | 0/2 | 首个2708-token样本在交叉熵处分配1.53 GiB失败 | failed，保留原源码与配置 |
| 仅监督位置logits | 2/2 | 有限loss/梯度，112个adapter tensor改变；冻结梯度0、三组冻结权重样本不变 | failed：优化完成后同进程重载前不足7 GiB空闲 |
| 独立进程重载上述checkpoint | 无额外优化 | 112个保存/加载tensor逐值相等，原训练回执hash未变 | completed，独立证明重载 |

投影loss运行峰值allocated **3.7803 GiB**、reserved **4.65625 GiB**，训练循环计时 **4.9045 s**，不含加载/预处理；独立重载峰值allocated **2.2946 GiB**。原两条failed receipt未改成completed。新的可行性CLI结束于`optimizer_completed_reload_pending`，必须再运行独立验证脚本。

## loss等价性与推断保护

本地Transformers4.57.1源码显示原Qwen2-VL先对全部位置生成词表logits，再将logits转FP32交叉熵；`2708 × 151936 × 4 bytes ≈ 1.533 GiB`，与OOM分配量一致。显式`supervised_only`分支仍计算完整多模态transformer，只将`labels[t+1] != -100`的`hidden[t]`送入lm_head，目标与均值分母不变。它在数学上省略零loss项，而非删除上下文/监督或近似词表。

CPU测试覆盖非连续masked标签、empty/错维度拒绝、微型Qwen+普通LoRA的文本及真实视觉patch；loss容差`rtol=atol=1e-6`，adapter梯度`rtol=1e-5, atol=1e-7`。独立只读审查又检查非重入梯度检查点、非零LoRA和tied embeddings。只支持普通单LoRA、Qwen2-VL及列明输入字段，其他adapter模式不泛化声明。

NF4生成只对末位置生成词表logits，完整transformer/KV cache保留；真实视觉微型Qwen的4-token greedy生成与全logits逐token一致。该路径和依赖/量化模块被记录并强制配对，不能将NF4 adapter与旧BF16 base直接比较。测试先复现旧评估器允许精度/投影混配，再验证拒绝精度、投影、bitsandbytes版本与量化结构漂移，同时允许完整一致的NF4 cohort。

## 完整训练和研究边界

[完整NF4实验协议](TABLE_NF4_SCREEN_PROTOCOL_20261007.md)随后冻结两臂各381步、每文档12次暴露、同精度base及三个条件各103条开发推断。当前先执行`all`，之后`no_explicit_preservation`，两次checkpoint重载、base/all/no-preservation推断和官方离线评估；流程每个阶段重新核验固定源码/config，失败即停，拒绝已有输出，观察现存进程时绑定boot ID/PID启动时刻/命令hash。

本记录不提前填入完整训练或质量结果。原BF16两臂、教师轨迹蒸馏、独立judge/Jev、RL、三seed及锁定迁移仍未完成。后续只有完整103例结果可支持本次单seed开发提议能力的结论。

140项全环境测试通过、无optional skips，包含真实TeX/表格与官方指标；Ruff/Black通过。完整测试曾暴露外部config路径在源码快照阶段使失败回执停在initializing，已修复并保留失败/通过日志。代码审查未发现训练与推断路径阻断问题。

[公开可行性证据包](../../experiments/artifacts/table-nf4-readiness-20261007/)保存原失败、独立重载、数值日志、源码和依赖hash。原始训练图/HTML、adapter权重、凭据和宿主私有路径不在该证据包内。

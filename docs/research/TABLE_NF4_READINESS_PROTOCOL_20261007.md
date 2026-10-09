# 8 GiB 本地 GPU 的独立 NF4 训练可行性检查

2026-10-07，在首次量化模型加载/优化之前固定。问题是现有Qwen2-VL-2B表格监督能否在本地RTX5070 Laptop 8 GiB显卡上完成受限训练与检查点重载。它不替代已冻结的BF16两臂，不产生蒸馏、修复或泛化结论。

## 固定条件

- 使用同一Qwen revision `895c3a49bc3fa70a340399125c650a463535e71c`、127文档/406条已准入train记录及原始图像/prompt/target；不读calibration/locked标签。
- 现有PyTorch `2.8.0+cu128`、Transformers `4.57.1`、Pillow `11.3.0`保持原样。新增依赖通过独立包目录加载：PEFT `0.17.1`、bitsandbytes `0.48.1`；wheel与PyPI公开SHA256核验。
- 新增显式`nf4_lora_8gb`精度条件：NF4四位权重、double quant、BF16量化矩阵计算、SDPA；按PEFT准备函数将非四位参数转FP32。逐项记录被量化模块，视觉骨干可被冻结量化，但仍只优化语言q/v LoRA。
- 不启用CPU/disk offload。只有GPU初始至少7 GiB空闲且没有其他compute作业时开始；PyTorch allocator限制为总显存的80%，给桌面保留余量。这不是排他式GPU资源预留，非PyTorch分配也需报告。
- 默认`bf16_lora`仍要求16 GiB，未显式选择新条件不能降低门槛。
- r=4、alpha=8、dropout=0、AdamW lr=1e-4、weight_decay=0、clip=1、seed=20261007；image bounds100352–200704，cap4096，不截断或缩图以通过检查。

## 两个预定内存边界样本

选择只用既有406条token预检查，SHA256 `047f6862b51073a389ff8bd6dc83c3e6630cf8efe418e70a2dd57167f7d0ebb1`。按最大processed sequence与最大视觉grid元素数分别取一个样本，同分按sample_id排序，去重；不按loss、质量或正确性筛选。

- `p0055-extra_row`：2708处理后token、13监督token、grid `[1,16,36]`。
- `p0079-cell_perturbation`：968处理后token、21监督token、grid `[1,10,64]`。

在同一次初始化中依次执行两个batch1、accumulation1的optimizer步。该临时adapter只作可行性检查，不作为后续381-step训练的初始化或best checkpoint。保留失败，不换更短样本重报成功；若调整配方，另立协议和run。

## 验收及后续触发

逐样本重新检查prefix/assistant掩码、token数和图像grid；确认有限loss/梯度、仅语言LoRA进入优化器、frozen参数无梯度且样本不变、adapter参数确实更新；保存峰值显存、耗时、训练/模型/源码hash。释放模型后从原始量化base重载adapter，并逐tensor检查保存/加载一致。OOM、依赖或重载失败必须保留failed receipt，不能报告为训练就绪。

若通过，才能冻结独立NF4两臂与同精度base推断对照；两臂仍用全部127文档、每文档12次暴露、381步。推断也必须匹配量化、非量化层dtype、硬件和依赖，不能与此前CPU BF16基座直接配对归因。实际token预算、浮点近似与单seed限制继续披露。教师轨迹表示、独立judge、Jev、RL及锁定迁移保持各自未完成状态。

## 本次读取的一手兼容性来源

访问日期均为2026-10-07，完整下载内容及hash存于本地运行依赖目录：

- [bitsandbytes 0.48.1安装说明](https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/0.48.1/docs/source/installation.mdx)：读取最低Python/PyTorch要求、NF4能力要求及Linux CUDA12.8–12.9包含sm120的构建矩阵。
- [PEFT v0.17.1准备函数](https://raw.githubusercontent.com/huggingface/peft/v0.17.1/src/peft/utils/other.py)：读取`prepare_model_for_kbit_training`冻结base、非Params4bit转FP32及梯度检查点分支。
- [bitsandbytes PyPI元数据](https://pypi.org/pypi/bitsandbytes/0.48.1/json)与[PEFT PyPI元数据](https://pypi.org/pypi/peft/0.17.1/json)：读取版本、依赖条件与wheel hash。文档支持不等于本机训练已通过。

## 首次失败后、第二次运行前的显式补充

首次按上文全位置logits执行，在首个2708-token样本的官方交叉熵中OOM，完成optimizer步为0；保留原配置、源码快照和failed receipt。源码及堆栈表明词表维度为151936，单个`2708 × 151936 × float32`矩阵约1.533 GiB，与失败分配量一致。更改allocator比例、缩图、截断或换样本均不采用。

第二次仍使用同一NF4精度、样本、完整上下文与显存上界，新增显式`loss_projection: supervised_only`：完整多模态transformer仍处理全部token；仅把`labels[t+1] != -100`对应的`hidden[t]`送入lm_head，再对相同目标计算FP32交叉熵均值。因ignored标签对loss及梯度贡献为0，这在数学上与原masked causal CE相同；不省略被监督token或它们对上下文的反传。实际浮点结果允许测量到的容差，必须先用含LoRA的微型Qwen模型比较loss和所有参数梯度、边界标签与拒绝路径。

它只供明确的单图/普通LoRA训练内核，不能隐式作用于生成、prompt-tuning、混合adapter或其他loss定义。默认BF16和原配置仍使用原官方全位置loss。新运行使用独立配置、目录和源码hash，原失败不会改为成功。第二次通过也不证明完整训练或最终质量。

第二次优化完成后，同进程独立重载前的7 GiB门槛未满足，该run仍为failed。随后仅对已保存checkpoint在全新进程重载，保持相同门槛和80%allocator，不重训、不改原receipt。两份hash相互绑定的证据共同支持可行性判断。当前CLI因此拆为`check_table_nf4_readiness.py`（optimizer_completed_reload_pending）与`verify_table_nf4_checkpoint.py`（独立completed），具体观测见[结果](TABLE_NF4_READINESS_RESULTS_20261007.md)。

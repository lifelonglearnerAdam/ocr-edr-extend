# 表格JSON局部编辑SFT开发筛查

2026-10-07，新的表格optimizer调用前冻结。H1/H4的受控监督阶段；不等于原生parser改进、teacher trajectory或Agentic RL。

## 训练准入与标签含义

原始128 train文档、410记录保持冻结。基于已记录的源图审查，排除p0002整族4个变体，另生成127文档/406记录的准入版本，不替换样本、不改发布HTML。其他标签是带已披露噪声的发布弱监督；8个训练源检查不能证明全体准确。准入策略/审查ledger和输入文件哈希必须绑定；校准/锁定角色不得进入加载器。32文档/103例model-dev完整保留，不按难度或正确性筛选。

表格地址仅从当前HTML计算；target/action/reference和原bbox不上送推理提示。每个target是已验证可执行的stop、replace_cell、set_span或delete_row，受控错误只支持声明的动作空间。文档ID、图像bytes/pixels及角色隔离在准备阶段完成，加载时再次核验源图inventory和hash，跨机器只重定位相同图像identity。

## 固定两臂

基座Qwen2-VL-2B-Instruct revision `895c3a49bc3fa70a340399125c650a463535e71c`，两臂均从base开始，不加载公式SFT或临时smoke权重。语言q/v LoRA r4、alpha8、dropout0，AdamW lr1e-4、weight_decay0、clip1、seed20261007，bfloat16/SDPA与梯度检查点。assistant target和终止token参加loss，source/user/header/padding全部屏蔽。

- `all`：每文档保留保持+数字+extra-row，以及存在时的span变体。
- `no_explicit_preservation`：去掉显式stop保持样例，其余完整保留。

各文档均12次样例暴露，确保3/4变体（完整臂）和2/3变体（去keep臂）都完成整数次循环；共127×12=1,524次暴露，gradient accumulation4，共381 optimizer steps。变体分配和序列长度不同，监督/输入token成本实际记录，不能假称同token。固定终态checkpoint，无dev checkpoint搜索；单seed只作开发筛查。

图像bounds100352–200704，训练processed sequence上限4096且不截断；全部候选在加载模型前检查token长度、prefix边界和hash。若超限，保留失败receipt并重新制定资源协议，不静默删除例子。推断192新token、greedy、单JSON动作，截断/多动作/格式错误回退并计入成本。

## 验证和进入执行的条件

准入排除、角色拒绝、文档暴露平衡及跨机路径/内容验证的CPU测试先失败再通过。使用已有formula SFT优化内核，并保留其固定已执行版本；新内核不读取dev/calibration/locked文件。训练GPU必须实际有足够空闲显存及未被他人任务占用；不能因空闲显存片段就忽略接近100%的GPU工作负载。

按完整model-dev逐例比较unchanged、base、all、no-explicit-keep：TEDS/TEDS-S、局部动作位置、执行/渲染失败、保持/数字/span/extra-row分支、cost和source-grounded误改分开报告。官方metric满分或目标能回放不代表真实源图完全正确；当前受控数据不替代native表格parser。校准/locked图像仍不用于模型/阈值选择。

本协议准备时GPU主连接未恢复、备用GPU忙；训练代码和本地预检完成不等于optimizer已经运行。执行后以实际job进程、run/checkpoint hash和完整结果为准。五方向的独立视觉验收、Jev实际接入、教师轨迹、GRPO及锁定迁移仍需后续证据。

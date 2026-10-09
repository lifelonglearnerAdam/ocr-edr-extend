# NF4开发筛查的固定统计与源审计规则

2026-10-07。完整`all`训练尚在运行，三个条件的本轮model-dev模型调用尚未开始。该附录固定结果分析，不修改20个已冻结训练/推断/评估执行文件，也不改变优化器或教师标签。

## 配对与统计单位

输入必须来自完整且成功的官方离线评估：每个`sample_id`同时有`unchanged_0/base/all/no_explicit_preservation`四臂、103个case/32个PMC文档，源家族、文档、variant和两个初始参考分数完全相同。重采样单位是原始PMC文档，不能将同源变体视为103个独立观察；每个文档先对它的变体取算术平均，再对文档等权取均值。

预定四个对比：`base−unchanged_0`、`all−base`、`no_explicit_preservation−base`、`all−no_explicit_preservation`。逐个报告TEDS/TEDS-S终态分数差。固定seed20261007、10000次paired document bootstrap，百分位区间取排序样本的index `floor(0.025B)`和`min(B−1,floor(0.975B))`。case、document按ID排序，因此行顺序不影响同seed结果。

全体文档的比较是主要描述；preservation、cell_perturbation、extra_row、span_perturbation按实际存在variant分别列出，缺少某变体的文档不人为补样本；每项给出自己的case/document分母。该分层是描述性的，不做多个区间择优、p值或确认性显著性声明。一个seed和已被检查的32源不证明泛化/低风险保证。

## 风险与成本

初始指标满分定义沿用`>=1−1e-12`，与源图正确性分开。对每臂保留初始满分数、满分后退化数、初始非满分数、全修复数、部分改善、下降、保持/回退/输出上限计数。既有原始摘要作为逐case和逐document数值来源，统计脚本不改变动作或重新选择终答。

按原统一设计的权重`λ ∈ {1,2,4}`报告度量代理效用：`U_i = (TEDS_final_i−TEDS_initial_i) − λ I[initial_metric_match and final_nonmatch]`，先文档内再文档间等权平均。它没有把生成成本转换成质量单位，不是已验证RL奖励，也不是部署judge。TEDS-S另作结构诊断，不把两个指标权重相加后择优。

完整成本保留原model_calls、input/output tokens、generation_seconds、render_attempts/render_seconds；同时报告case与document等权平均。生成时间不含加载/预处理，重渲染具有内容缓存，不称端到端时延；精度/硬件/依赖与源hash不同的run拒绝配对。

## 模型输出之后的源审计

对所有实际修改的model-dev源、所有度量退化、所有非满分→满分的源建立合并审计队列；不只挑提升图。查看源图、初态、冻结终态和执行地址，记录数值/结构改动是否有可见依据、未改区域是否保持、是否存在参考表示分歧。若全体零修改，保留零修改事实而非制造修复案例。

这属于本助手的事后有标签开发审查，不是独立盲标；结论按源限定。补充图可展示真实案例与反例，但不能用选图推断整体视觉准确率。教师pilot已见过的train例不混入model-dev队列；校准/锁定图像不打开。

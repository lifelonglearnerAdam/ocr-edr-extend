# 固定NF4 checkpoint的提示示例消融

2026-10-09，在本对照的模型调用开始前冻结。此前literal-example NF4完整实验为负向：base平均TEDS0.957542557，all0.938525881，no-explicit-preservation0.907276678；71个初始非满分均未全修复，两臂各12/32个满分输入退化。all有24个、no-preservation有7个动作直接写入提示示例的占位文本。这些观察触发本次**事后探索性**对照，原始实验结果与输出均保留。

研究问题：固定终态checkpoint下，字面JSON示例是否解释部分模板复制/退化？描述式提示与训练时prompt相同；literal版本额外添加既有四个示例。因此改变提示同时改变token长度和训练/推断prompt匹配，不能归为唯一的“文字示例语义”因果效应。

三个模型条件base/all/no_explicit_preservation在两种提示下各跑完整103例/32文档，共618次新调用。两个已完成381-step adapter逐文件hash固定，不重新训练或用开发结果挑checkpoint。此前literal运行不能替代此次新同源码literal条件；重放一致性单独检查，不挑更优输出。

训练config、NF4/FP32准备、依赖、图像范围、greedy/max192、动作/渲染回退不变。新增独立inference protocol明确允许descriptive_schema/literal_examples，绑定训练config和两组adapter权重hash；原NF4默认路径仍拒绝未声明prompt偏离。两种提示分开做同cohort四臂评估，再按同sample/document固定配对比较。

两个格式均保留全部调用、合法动作、stop/no-op/实际修改、目标地址/完整动作匹配、占位词复制、cap、参考修复/退化及成本；不得把全stop恢复的指标当成纠错收益。两种输入相差的token量实际报告。按原PMC文档等权、seed20261007/10000次配对bootstrap计算描述式减literal的TEDS/TEDS-S差；非确认性结果。

不读取calibration/locked图像/标签，不改变原训练目标或参考。无论对照是否改善，当前负向结果均需公开；后续优先解决动作定位和有源依据的提议能力，独立verifier/教师监督不能假称已经集成完成。

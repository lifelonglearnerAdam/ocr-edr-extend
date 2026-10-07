# 第二具名解析器开发对照：LaTeX-OCR

日期：2026-10-07。完成[固定协议](SECOND_PARSER_PROTOCOL_20261007.md)的32-source开发parser-shift诊断。SFT终态checkpoint只用Nougat/标注扰动/保持样例训练，本次LaTeX-OCR输出未进入训练或模型选择。源图已经参与此前开发/检查，所以不是独立锁定测试，也不证明无预训练重叠。

## 实际运行

LaTeX-OCR作者源码commit `5c1ac929bd19a7ecf86d5fb8d94771c8969fcb80`，权重来自作者下载程序指向的`v0.0.1` prerelease；资产身份、大小与本地SHA256固定，未把本地hash冒充发布方独立签名。使用作者resizer和temperature=0.2的采样，每源固定seed，不称greedy。32/32源输出，无空结果、无512token上限命中；约9.47秒CPU FP32测量窗口。剪贴板副作用关闭，不修改模型/预处理/decoder。

学生base和既有all-SFT终态各跑32次source-only调用，bfloat16、CPU、相同图像和文字候选、greedy/256token、无参考访问。base30次格式回退，SFT31次可用提议、1次cap回退；全部输入保留。先固定syntax/render-only final，再加载参考作官方core-CDM。该闸门不判断视觉正确性。

## 全量配对结果

| 条件 | Mean core CDM / 32 | 不匹配全修复 / 11 | 部分改善 | metric-Good退化 / 21 |
|---|---:|---:|---:|---:|
| unchanged LaTeX-OCR | 0.899375 | 0 | 0 | 0 |
| base + 固定输出契约 | 0.899375 | 0 | 0 | 0 |
| 既有直接目标SFT + 同契约 | 0.989906 | 6 | 4 | 0 |

同32源的逐源mean改善为0.09053125，即9.0531个百分点。2000次source-group配对bootstrap描述性95%区间为[0.03416,0.16459]；一次seed、已检查开发图及选样限制，不能升级成确认性显著性/独立普适性结论。原始文档ID缺失，组单位仅到图像。

这与Nougat输入的负向开发结果一起说明：固定学生在这两种原生错误分布上的行为不同。LaTeX-OCR初始不匹配11/32、Nougat3/32，不能把差异全部归因模型“善于跨parser”；初始化难度、错误类型和接近满分的上限效应也是解释。主表分开报parser，不合成一个看起来普遍上涨的总分。

## 源图审查避免把分数当真值

全32个source/initial/student/reference面板已查看，参考未改写。六个metric满分修复中，大小写、乘号、sigma、R_e/数值、Pi上下标和分母/指数分组有可见源依据。例如`s0054`恢复R_e=12.9 a_0，`s0103`恢复Pi-prime等式；在Nougat输入同一source上的旧SFT额外上标伤害并未被改写成成功。

部分改善仍有错误：`s0037`恢复大部分积分结构但保留额外zeta上点/字体差异；`s0082`恢复整体指数积分却仍把q(x)写成U(x)。`s0135`原图含缺字形方框，学生写bar-Gamma、参考写sharp，不能因CDM由0提高到0.852就认定视觉修好。空集glyph等发布参考约定亦保留。该审查是非盲的开发观察，不是独立人工VisFix标签。

SFT在`s0011`触及256token上限，回退到原始解析器；不增加cap重跑或挑更好答案。31条最终字节变化包含排版/空白和良性重编码，不等于31个错误修复。base只变1条，CDM没改变；格式拒绝不能证明raw候选内容都不好。

## 成本和实现核验

base/SFT平均input tokens相同280，output约33.47/59.47，CPU生成时间约153.85/239.61秒；仅生成，不含文件核验、加载、渲染或评分。CPU环境与10月6日CUDA12.4训练/推断环境不同，这不是其原始calls逐位重放，不比较速度优劣。

本地HF快照的合法blob符号链接曾被仅支持普通文件的模型校验拒绝；失败在模型调用前，旧日志保留。修正后仅允许相同模型缓存的hash命名blob目标或普通snapshot文件；外部链接、路径穿越及hash漂移的负测试通过，未关闭完整性检查。code/weights/data/64个calls hash及96条CDM记录均保留。

## 决策

这是支持继续H1/H4并扩大**独立原生错误验证**的信号，尚非完整H5。下一批要有未用于调试的新文章/源组、固定parser和阈值，以及至少三seed确认。继续做提议/验收分离，不拿metric-full-match代替局部结构/源证据；Jev仍只能读取独立生成的结构化证据，教师/GRPO效果未完成。

[记录包](../../experiments/artifacts/second-parser-development-20261007/README.md)只含聚合、exact源码与运行/审查receipt，原图、参考、raw输出和checkpoint留在忽略目录。表格四角色与标注准入进展见[TABLE_DATA_RESULTS_20261007.md](TABLE_DATA_RESULTS_20261007.md)。五方向整体目标保持进行中。

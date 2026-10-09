# 第二具名公式解析器的固定开发诊断

协议日期：2026-10-07，LaTeX-OCR整批调用前固定。对应H5的有限parser-shift筛查，不能代替新来源上的锁定迁移评估。

## 固定解析器与样本

使用作者 `lukas-blecher/LaTeX-OCR`，源码commit `5c1ac929bd19a7ecf86d5fb8d94771c8969fcb80`；作者下载程序指向的 `v0.0.1` prerelease包含 `weights.pth` 和 `image_resizer.pth`，记录GitHub asset身份、字节数、本地SHA256。发布方未提供独立SHA256时明确区分下载身份固定与独立完整性证明。代码许可MIT，来源URL和实际读取日期另记。版本号不能取代checkpoint哈希。

输入保持原来的32个开发源，使用只含family ID、source path/hash的清单，不读取发布参考或候选。每个源只调用一次，保留空输出/长度上限/失败；不能重试挑更好答案。作者设置为temperature=0.2的采样并带图像resizer，不伪称greedy。对每个源以 `20261007:family_id`哈希确定随机seed，记录配置和模型文件，以便行为复核；没有调参或选温度。

当前机器缺GPU登录认证时可在独立的本地依赖目录执行CPU FP32模型，不改变既有Qwen/评测环境。兼容性补丁仅针对Python/第三方导入，不变更权重、decoder、预处理或输出后处理；若必须修改算法，则另列变体且保留原始失败。禁止隐式checkpoint下载和剪贴板更新。

## 比较

原来的Qwen2-VL-2B base与all终态SFT均固定不变，对这32个LaTeX-OCR原生候选生成source-only提议；复用相同严格标签/TeX合法性/渲染回退，再离线评分，全部输入覆盖。另保留解析器unchanged基线。没有将LaTeX-OCR错误加入此次SFT训练，也没有按这次错误选择checkpoint/阈值。

官方core-CDM、保守字符串匹配、源图审查和Good/Bad分支分别报告；不能将字符串或metric-full-match等同视觉正确。该32源已用于模型开发，原始文档ID不可用；训练集和本parser预训练数据可能重叠。因此结果只能称第二解析器上的固定开发诊断，不能称独立锁定测试或已证明普适迁移。部署验收和正式校准仍待后续。

原native-Nougat对照的候选与checkpoint不改写。第二parser的成本、FP32环境/temperature、异常和实际序列长度另记，速度不能与Qwen CPU/GPU或Nougat beam5直接当同条件对照。发布结果时同时报告完整parser初始分布，不能只挑Bad。

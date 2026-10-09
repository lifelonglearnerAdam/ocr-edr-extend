# 冻结学生在原生表格错误上的开发诊断

2026-10-09，在新模型调用前冻结。目标是检验描述式提示下受控重复行修复，是否在同32个已检查model-dev文章的真实PaddleOCR det/rec/SLANet候选上产生收益。它属于分布诊断，不是未见来源或最终H5锁定实验。

输入使用原先32源、0parser失败的显式original-frame-compatible原生结果，SHA256 `6bfe68cd225e3cac1f645849174209725936367b31fe8a91bf7a248c93ccd2f3`。每源只保留一条native候选；不按TEDS好坏选择源。基础原生TEDS0.982848074，仅在冻结三臂输出后用相同官方normalizer评分。

模型条件为同精度base/all/no_explicit_preservation，两组原381-stepcheckpoint固定、无新增训练。使用相同NF4/BF16计算/非量化FP32、greedy192token、source image+current HTML/addresses、descriptive_schema。每条件32次调用，共96次，全部保留caps/非法动作/回退成本。局部动作仍stop/replace_cell/set_span/delete_row，不新增能直接抄参考的动作。

生成阶段仅读取model-dev三字段源清单与冻结native输入，不打开参考。完整calls/receipt先冻结，之后独立离线加载已发布model-dev reference，评分unchanged/base/all/no-preservation共128条固定配对，报告全源修复、退化、动作/成本及结构变化。不会使用calibration/locked源。

不能将公开PubTabNet预训练来源描述为parser未见，也不能用当前已审查model-dev证明来源泛化。即使零退化，32源仍不足以提供1%风险保证。若学生只擅长合成重复行、在native上无净收益，下一阶段需改进源证据/定位和真实错误监督，再谈独立verifier、Jev和RL的增量。

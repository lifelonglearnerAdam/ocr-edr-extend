# 具名原生表格输出开发协议

2026-10-07，32源正式parser推断前冻结。目标是在同一批model-dev源上取得未经人工扰动的、具名且可复现的HTML候选，为后续学生修复及动作空间消融建立输入。不是独立未见来源的泛化检验。

## 数据与模型

仅使用PubTabNet四角色数据的`model_dev-source-inputs.jsonl`，32个原始PMC文档各一张源表图。每条只有family_id、source_image、source_sha256；不读取发布HTML、原cell bbox、target action或正确错误位置。全部32条保留，不按parser成功、分数或错误数量筛选。calibration/locked不进入本阶段。

作者PaddleOCR源码commit `e218c0a35fbcb5085ab3527931b2d8d08f8f0292`；使用官方英文表格检测、识别与标准SLANet权重及`TableSystem`/`TableMatch`，算法源码不修改。模型与字典逐文件hash核验，三组权重由[已核查的官方文档组合](NATIVE_TABLE_INTERFACE_CHECK_20261007.md)下载。官方URL本身不是不可变content ID，记录此次下载字节hash，不冒称作者公布了SHA256。

完整配置见`configs/eval/native_ppstructure_20261007.yaml`：CPU float32，MKLDNN开启、4线程，det_limit_side_len736/min，rec_image_shape3,32,320，table_max_len488，英文table字典，seed20261007；其他选项来自该提交的默认值，并保存完整解析参数。`rec_algorithm=SVTR_LCNet`沿用作者评估命令的参数默认值，真正的识别权重身份由英文table_rec文件hash确定，不据此重命名其训练架构。

在正式dev运行前，固定首个已准入训练源p0001做了一次运行时检查：独立Paddle3.0.0b2 CPU环境成功返回非空HTML、16个cell框和16条OCR结果，约5.60秒单次pipeline时间。没有查阅参考、比较精度、训练更新或按dev结果选择环境。模型导入/forward可用不代表识别正确。

## 记录、评分与动作约束

逐源保存原始HTML、预测cell框、OCR框/文本/置信度、作者分阶段计时与外层pipeline计时；不对内容修正或重试。OCR非有限置信度记null；非法非有限几何等异常记为parser失败，保留源和调用时间。初始化失败另有failed receipt，不能当完整32源运行。

原始输出全部冻结后才读取model-dev参考。官方OmniDocBench TEDS/TEDS-S源码先核验固定Git blob，参考先做自一致性检查；显式parser失败按预定零分保留在分母。平均单位是32个源/PMC文档，而不是原103个相关受控变体。

编辑器分析独立于质量评分：检查原始预测HTML是否符合当前adapter；对合法HTML比较DOM行/单元格数和相同DOM形状下的纯文本/span差异。当前stop/replace_cell/set_span/delete_row不能增加DOM行或单元格；若预测数量少于发布参考，精确恢复该参考DOM在现有动作空间中不可达。该结论针对发布参考及原始DOM，不是完整TEDS最优修复oracle，也不能跳过源图审查直接认定参考视觉正确。

这一区分用于指导“扩大操作空间/允许全表重写/多步编辑”的后续消融，不能用受控错误目标可回放率代替原生可达性。parser训练涉及PubTabNet，而当前开发源来自公开train；模型具体逐图训练清单未知，不作parser-unseen、pretraining-clean或锁定迁移声明。

# 原生表格基线接口与来源核查

读取日期：2026-10-07。此记录核查官方源码/文档，尚未下载或运行原生表格parser权重，不含新的parser性能结果。当前环境无`web-access`技能，使用GitHub官方API读取原文并校验Git blob；没有以搜索摘要代替原文。

## 候选与固定版本

候选为PaddleOCR的英文PP-Structure表格流水线：英文表格文本检测、文本识别、SLANet结构/单元格坐标预测及官方匹配器。官方`v2.10.0`标签解析到commit `e218c0a35fbcb5085ab3527931b2d8d08f8f0292`；版本名称不能替代模型文件hash和运行环境核验。

已读取的主要来源：

| 原始来源 | 本次实际读取范围 | 支持的事实 |
|---|---|---|
| [table/README.md](https://github.com/PaddlePaddle/PaddleOCR/blob/e218c0a35fbcb5085ab3527931b2d8d08f8f0292/ppstructure/table/README.md) | 全文 | 三模型加匹配器构成完整HTML流水线，官方提供CPU运行示例和PubTabNet英文模型组合 |
| [predict_table.py](https://github.com/PaddlePaddle/PaddleOCR/blob/e218c0a35fbcb5085ab3527931b2d8d08f8f0292/ppstructure/table/predict_table.py) | 全文 | `TableSystem.__call__(img)`只需图像；返回HTML、预测cell框与分模块计时，不需要参考或GT框 |
| [matcher.py](https://github.com/PaddlePaddle/PaddleOCR/blob/e218c0a35fbcb5085ab3527931b2d8d08f8f0292/ppstructure/table/matcher.py) | 文件已完整取回并校验；当前接口判断依据调用处 | 官方匹配器文件固定；尚未逐分支审查或执行其行为 |
| [models_list.en.md](https://github.com/PaddlePaddle/PaddleOCR/blob/e218c0a35fbcb5085ab3527931b2d8d08f8f0292/docs/ppstructure/models_list.en.md) | 全文，重点2.1/2.2 | 英文检测、识别、SLANet模型均声明在PubTabNet上训练；模型下载地址和结构模型区分明确 |
| [SLANet.yml](https://github.com/PaddlePaddle/PaddleOCR/blob/e218c0a35fbcb5085ab3527931b2d8d08f8f0292/configs/table/SLANet.yml) | 全文 | 标准SLANet为PPLCNet/CSPPAN/SLAHead，训练路径指向PubTabNet train，评估路径指向val；结构输入边长488 |
| [SLANet-LCNetV2说明](https://github.com/PaddlePaddle/PaddleOCR/blob/e218c0a35fbcb5085ab3527931b2d8d08f8f0292/docs/algorithm/table_recognition/algorithm_table_slanet.md) | 全文 | 这是另一骨干/训练流程的模型，不能与上述标准SLANet权重混称 |
| [LICENSE](https://github.com/PaddlePaddle/PaddleOCR/blob/e218c0a35fbcb5085ab3527931b2d8d08f8f0292/LICENSE) | 文件已取回并校验；源码文件许可头已读 | 作者源码标记Apache-2.0；这不自动确定源文章图片权利 |

README中的旧`ppstructure/docs/models_list*`、`doc/doc_en/table_recognition_en.md`链接在该提交未成功读取；通过固定Git树定位了上表实际路径。保留失败记录，不将未读链接列为证据。官方性能数字未经本项目重跑，不放进本项目结果表。

## 数据和解释限制

当前32个model-dev源均从PubTabNet公开train划分抽取。官方SLANet配置也使用该训练划分，英文检测/识别模型同样声明使用PubTabNet；具体发布checkpoint没有逐图训练清单。因此这批源不能称作parser未见过的来源，也不能把学生的文档角色隔离外推为parser预训练隔离。

选此模型的理由是可识别来源、官方完整流水线和CPU可运行接口，不依据尚未生成的预测质量。可以在完整32源上先做原生错误开发诊断；后续独立迁移需要额外未参与选择的来源和明确的pretraining/benchmark边界。单换结构模型、共用OCR检测识别器的变体也不能宣称是完全独立的多个OCR系统。

## 下一步执行边界

1. 分别核验官方检测/识别/结构权重、字符字典及依赖；下载后记录文件字节hash。SLANet链接包含`paddle3.0b2`，实际导出格式/运行时兼容性需实测，不能猜测为已验证的Paddle2或Paddle3配方。
2. 只输入32条model-dev的源图清单，不输入发布HTML、原cell bbox、target action或受控错误位置。保存所有原始输出和解析失败，完整HTML由官方流水线产生。
3. parser原生质量与学生修复分别离线评估。表格地址从原生预测HTML推导；校准/锁定图像保持未用。
4. 先审计现有一步动作的可达性：缺失行/单元格和多处错误通常超出当前`stop/replace_cell/set_span/delete_row`的一步范围。不能以受控扰动的100%目标可回放推断原生错误都可修复。
5. 对所有候选保留调用成本、框/HTML来源及失败记录，再冻结修复配置。此文没有提前决定任何模型有效、迁移成功或视觉验收可靠。

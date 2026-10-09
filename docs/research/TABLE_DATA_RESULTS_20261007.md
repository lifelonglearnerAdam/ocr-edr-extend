# 表格四角色数据准备及文档交集审计

日期：2026-10-07。对应五方向框架的表格监督/独立校准前置条件；本轮未训练表格模型，未执行锁定评测，不产生TEDS性能改善结论。

## 为什么重新做split

全11.24GB发布包重新校验后，实际清点得到500,777 train、9,115 val、9,138 test图片。509,892条发布标注仅覆盖train/val。PubTabNet图片名带PMC文章ID；直接官方split存在原始文章交集：

| 发布split对 | 共享PMC文章 |
|---|---:|
| train / val | 6,105 |
| train / test | 6,124 |
| val / test | 216 |

这证明本包发布的图片分区不满足本文要求的文章独立性，并不说明所有先前基于图片split的研究都无效。禁止在未检查的情况下把图片独立外推成文档独立。

新的train/model-dev候选排除所有17,794个出现在val/test的文章；calibration/locked候选来自val，但另排除test文章。固定文章hash决定角色，固定文件hash顺序选择满足既定大小/动作条件的表，一文章只取一表。

## 最终冻结规模

| 角色 | 独立文章/源表 | 保持 | cell数字扰动 | span扰动 | 重复末行 | 总案例 |
|---|---:|---:|---:|---:|---:|---:|
| train | 128 | 128 | 128 | 26 | 128 | 410 |
| model-dev | 32 | 32 | 32 | 7 | 32 | 103 |
| gate-calibration | 32 | 32 | 32 | 6 | 32 | 102 |
| locked-evaluation | 64 | 64 | 64 | 21 | 64 | 213 |

最终六个角色对的PMC文章、图片字节hash及解码pixel hash交集均为0。271个冻结元数据/所选图片文件逐一复核哈希；828个目标动作均通过现有adapter精确恢复canonical参考。变体同行来自同一文章，不能当828个独立观察。

四角色分区已存在，但calibration32仍不能支持1%低风险证书，见[统一设计的样本量边界](INTEGRATED_RESEARCH_20261006.md)。锁定评测图片未人工查看、未送模型作性能选择；更没有拿它来优化阈值或训练。未经源码/版本/成本验证的Jev、教师和RL不会因此自动成立。

## 可以和不可以得出的结论

标注结构/内容的转换保留受支持inline格式、正span及字符转义，按cell闭合顺序拼接；不支持项记录排除。只给推理端源图和当前HTML；correct cell/address/action、原bbox及角色/变体信息留在监督文件。训练和model-dev已装配410/103条JSON局部编辑目标，calibration/locked目标没有进入装配器。

这批错误来自发布标注的受控扰动，不是具名OCR原生错误、教师生成轨迹或视觉审计真值。现有编辑空间可以恢复这些构造错误，但仍不支持任意缺失行插入；不能据此说真实错误都可修。后续必须补具名表格parser及完整原生样本。

独立官方TEDS/TEDS-S仅用于train/model-dev target readiness：所有正确target自分1.0，train的282个受控案例、dev的71个受控案例在初始TEDS上低于1。没有比较任何新模型最终输出；metric selfcheck不是源图正确性证书。

## 训练前发现的发布标注缺口

固定首8个train源/参考渲染检查中，`p0002`（PMC4279644_002_01.png）原图底部有“Overall number of samples having mixed infection”一行，发布HTML缺失。独立查看原尺寸图片确认。未改写参考或丢掉结果；另记录label-review标记，在table optimizer开始前需要决定排除或独立校正的训练准入。不能把错误annotation作为无条件视觉Good监督。

其他7个源在接触图检查中未见明显整行缺失，但未进行逐字符独立盲审，不代表全128条train标注正确。校准/锁定图像不参与这个训练检查。通用CSS改变字体、边框和行高，只是内容/结构render evidence，不保证与原图像素一致。

## 许可和复现记录

原IBM许可文件在commit `8ffde9024bd331f5a61c3c549fdf30dd3c3e43a5`固定：标注CDLA-Permissive-1.0；图片受PMC及原文章许可约束。镜像卡标注CDLA-Sharing-1.0，与原说明冲突，保留两边实际证据，不假定所有图片同一许可。

原包、标注、图片和target保存在忽略目录 `data/processed/pubtabnet-four-roles-20261007/`；[源码与聚合记录](../../experiments/artifacts/pubtabnet-preparation-20261007/README.md)不含原图或HTML。模块错误行为与动作回放测试先失败再通过，当前完整环境101项测试通过，真实渲染/官方指标无跳过。代码/检查说明不替代模型效果。

下一步：训练源标注准入审计，固定局部JSON学生训练/保持消融，再接入独立视觉验收和同证据上的Jev/规则路由；最终锁定评测在独立model-dev与校准阶段完成后运行。原先4个已检查demo页继续留在历史开发集，未进入这批训练。

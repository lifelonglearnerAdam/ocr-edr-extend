# 公式 direct SFT 与显式保持样例消融

2026-10-06，在新训练/推断前冻结。它是统一五方向框架中 proposer 能力开发筛查，非锁定测试和最终论文实验。

## 固定输入和预算

Qwen/Qwen2-VL-2B-Instruct revision `895c3a49bc3fa70a340399125c650a463535e71c`；384条train/96条dev及模型文件哈希使用现有监督记录。仅语言 q/v LoRA，r=4、alpha=8、dropout=0、lr=1e-4、AdamW weight_decay=0、grad_clip=1；seed=20261006，bfloat16、SDPA、梯度检查点；192 optimizer steps ×4 gradient accumulation，batch1，768样例暴露。固定image bounds100352–200704、输出生成上限256tokens，无训练sequence截断，超过2048个处理后token则训练预检失败。

两臂都从原始基础模型初始化，不加载临时GPU检查参数。`all`使用每源3变体，完整清单shuffle循环2次；`no_explicit_preservation`排除显式keep，使用controlled/native每源2变体，循环3次。每源暴露次数均为6；保持删除改变target/上下文分布及实际token，不是逐token完全匹配。原生匹配参考的59条train样例仍在第二臂，所以这是**去显式保持对**，不是排除所有Good。

训练前核验train/dev/checksum/image路径及源图/家族隔离，预处理全部训练行以检查prompt相等、assistant边界掩码、序列长度和实际视觉grid。loss只监督assistant+终止token，input image/user/header/padding均屏蔽。dev只能进入对照评估，不进入loss/梯度/checkpoint选择。保留全部训练行，无goodness筛选、标签改写或根据表现重试。固定terminal checkpoint，不做best-of。

## 实现验收

CPU单元测试先证明：hash mismatch/dev训练/path escape/重复ID/保持target漂移被拒；train/devfamily图重复被拒；采样计划次数和暴露相同；标签前缀漂移、空assistant监督被拒。真实GPU检查确认只有LoRA在优化器、frozen梯度为零、loss/梯度有限、参数改变、checkpoint可加载且哈希有记录。错误留下failed receipt，不覆盖先前run、不假称断点恢复。

## 推断和评估

对96dev输入（32源×3变体）分别跑`base`、`all`终态、`no_explicit_preservation`终态；相同源图、候选、prompt、greedy及256新token。推断输入严格五字段，不读取reference、variant、SFT target。保存raw、token/cap/grid、候选和源码/adapterhash；标注由独立离线阶段读取。

输出适配固定`<latex>...</latex>`完整响应，whole display wrapper按现有规范处理；拒绝非契约、截断、unsafe TeX和失败render，回退baseline，并单独保留被拒raw候选。渲染成功不是接受正确性；这不是learned gate。未改baseline、提议前raw/contract候选和适配后final分别可供离线评估，但不能按reference挑final。

官方core CDM用现有固定源码/环境，先reference selfcheck，失败不产虚假聚合；string完全匹配和CDM=1都不等于视觉Good。按variant及source报告delta、fix/regression、输出/渲染失败、拷贝率及成本；误差区间以32源为重采样单位，不以96行作为独立N。开发集已查看，不作held-out/generalization或无预训练重叠主张。

## 解释与后续

loss下降只是优化证据。这个筛查只比较base/直接目标SFT/保持样例选择；不是教师蒸馏、RL、Jev或parser transfer完成。两臂同一次seed仅探索，结果无提升也完整保存。依据提议能力而非训练loss决定是否值得下一阶段；任何后续配方调整重新冻结，并用新验证/校准/留出样本确认。

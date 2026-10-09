# 当前助手作为教师：训练集轨迹试验

2026-10-07。用户明确授权当前助手担任教师；不再以额外教师API访问作为本阶段前提。目标是实际生成可回放的source-grounded编辑轨迹，并建立同最终答案的final-only与trajectory监督视图；不在本阶段宣称学生性能提升。

## 教师与可复现范围

当前会话最新turn_context记录模型标签`gpt-6-astra`、effort `max`。这是交互模型标签，不是不可变权重snapshot。保存每例给教师的任务描述、源图/当前HTML hash、动作、简短可核验的视觉证据摘要、工具返回及终态；不导出私有会话、隐藏思维过程或凭据。工具回放可复现；交互教师的完整上下文和采样不能视为可精确复现的API实验。没有精确的逐例教师token/费用测量，不记作零成本或最优教师。

## 训练数据隔离与选例

先对全部127个已准入训练源生成同一坐标兼容配置的原生Paddle候选。只读取train三字段源清单、source inventory和准入metadata，不读训练target/reference文件；p0002排除，calibration/locked禁止，model-dev不进入教师监督。

首批选择p0009–p0020共12个已准入训练文档，避开记录中先前已进行源/参考审查的首8个train源。每个文档只出一个case，避免跨变体对照泄漏答案。按固定hash排序选择6个native候选，其余6个从对应controlled输入按固定hash选一个变体；不根据模型输出、分数或难度选例。

教师任务只显示匿名case ID、源图、当前HTML/当前渲染和从当前HTML推导的DOM地址。variant名、原生/受控来源标签、参考HTML、正确动作与正确位置留在映射/验证文件，生成阶段不查看。教师知道整体研究背景，因此不声称对所有研究假设盲测，也不声称预训练未见这些公开图像。

## 可观察轨迹与预算

每例最多3次编辑及对应重渲染，再stop；失败的操作也占预算。每个决策包括一个动作、简短的source-visible证据说明和不确定性标记。它是可公开的操作/证据记录，不是隐藏chain-of-thought。

可用局部动作沿用stop/replace_cell/set_span/delete_row。若所需修改超出现有局部语法，可提交完整global_patch HTML，须通过相同安全HTML parser与renderer；它单独标记为全局改写，不假装一次局部修改。教师不能读参考后偷偷修订旧输出，不能把模糊字形凭领域常识填成“确定”答案；不确定时保留abstention或review-required。

动作执行后对比源图与更新后的渲染，记录新的HTML和render hash，防止对旧渲染验收。教师的停止判断标记为self-check，不能充当独立judge。没有把参考分数反馈给生成中的轨迹。

## 冻结后核验与未来消融

12条轨迹（含失败/abstention）先冻结，再做执行、渲染、局部作用域及发布参考一致性检查。参考仅为弱监督；不一致进入审查，不自动认定教师或参考正确。机械可回放与参考一致不能改称独立视觉金标。保留所有生成/拒绝数量和理由。

在同一批通过准入的episodes上，派生两个视图：teacher final-only（只给最终已核验答案）与teacher trajectory（给相同最终答案对应的动作/工具/简短证据记录）。匹配学生模型、源文档、初态、最终答案与训练token预算；同一次轨迹的final-only不叫独立重新询问的direct-answer教师实验。若后续比较单独询问的直接答案，需新增隔离上下文条件。

当前训练内核只支持单图/单目标，不能把多步轨迹随意塞进原单JSON目标后声称完成蒸馏。轨迹数据先收集和核验，学生表示、工具状态训练及匹配token预算另冻结。原两组381-step直接SFT配方和原始结果保持独立；教师数据不进入已留出的model-dev/calibration/locked角色。

## 工具复现与封存后的准入规则

下列工具是本批交互过程的包装；它们不能重新采样出完全相同的交互教师。新生成使用全新目录；原封存包和生成源码快照禁止覆盖。原始这批提示词由单独工具步骤写入并绑定hash，原seal也由显式工具步骤生成；补充CLI消除了这两个必需的手工接口步骤，不追改原证据。

```bash
teacher_run=experiments/runs/interactive-teacher-new-run
.venv-pilot/bin/python scripts/prepare_teacher_table_pilot.py \
  --dataset data/processed/pubtabnet-four-roles-20261007 \
  --native-run experiments/runs/interactive-teacher-pilot-20261007/native-pool \
  --teacher-instructions configs/train/interactive_table_teacher_20261007.txt \
  --output "$teacher_run/packet"
# 教师逐例读取源图、当前HTML/渲染并写独立decision文件后，执行该文件：
.venv-pilot/bin/python scripts/execute_teacher_table_step.py \
  --packet "$teacher_run/packet" --decision "$teacher_run/packet/decisions/CASE-00.json"
# 全部12例stop后，才封存并打开训练参考：
.venv-pilot/bin/python scripts/seal_teacher_table_pilot.py \
  --packet "$teacher_run/packet" --output "$teacher_run/cohort-seal.json"
.venv-pilot/bin/python scripts/audit_teacher_table_pilot.py \
  --packet "$teacher_run/packet" --seal "$teacher_run/cohort-seal.json" \
  --dataset data/processed/pubtabnet-four-roles-20261007 \
  --native-run experiments/runs/interactive-teacher-pilot-20261007/native-pool \
  --official-root data/raw/omnidocbench-table-portable-20261004 \
  --output "$teacher_run/audit"
```

审计工具先验证所有文件、初始和更新渲染、完整动作/终态链和预算，之后才解析映射/参考；角色、文章和图像身份交集均核验。机械检查无法证明完整交互上下文的盲化或实际图片观看行为。原渲染失败保持回滚，即使离线重试成功也不覆写原事件。

本批临时准入规则在第一次参考评分前写入[审计计划](../superpowers/plans/2026-10-07-teacher-audit.md)：机械有效、合法终态HTML、low不确定性终止、两种官方参考分数均在`1e-12`内满分。其他样本进入复核队列，仍保留完整分母。该规则偏向参考一致的例子，不是独立视觉金标；两种配对视图尚未匹配训练token，也没有实际学生更新。详见[12例完整结果](INTERACTIVE_TEACHER_RESULTS_20261007.md)。

# 表格诊断—修复组件实验（事后开发）

2026-10-09，在新诊断训练与推断前冻结。原生诊断失败与源图探针提示定位/错误监督仍有缺口。本轮将“诊断哪里错”和“改成什么”分开训练与评价，不把新组件默认视作有效验收层。

## 实验问题与预算

1. 一个≤3B的独立适配器能否从源图+当前HTML识别valid/invalid、错误类别及候选DOM范围？
2. 在同一冻结修复器上，预测诊断是否增加修复、降低误改？位置被故意移到另一个合法位置后，收益是否保留？
3. 已知构造位置但不给正确文本/恢复span的理想诊断，能带来多大上限？若上限也失败，应先改善修复器而非夸大诊断能力。

诊断模型从固定Qwen2-VL-2B原基座开始，另训练rank4语言LoRA；不从现有refiner继续训练。使用既有准入train127篇/406例（p0002整族排除），每文档12次、1524暴露、381步×accum4，seed20261007、lr1e-4、同NF4/BF16计算/非量化FP32、监督位置loss及4096序列上限，不截断或过滤难例。目标来自已准入的受控动作：stop→valid；replace_cell→content/cell；set_span→structure/cell；delete_row→extra/row。诊断目标不含正确文本、恢复span或参考HTML。训练后固定终态，不以dev效果选checkpoint，独立进程核对112adapter张量重载。

## 模型接口与隔离

输出仅三个字段verdict/error/region。valid必须error=null/region=null；invalid类别content/extra/structure；region为当前候选内cell(row,cell)或row(row)。严格拒绝重复字段、bool索引、不存在位置、未知类型、无效JSON和输出触顶；错误也计入完整分母。candidate与source哈希绑定，候选更新后不能沿用旧诊断。该region是DOM范围，尚非图片坐标。

推断诊断模型读取全部model-dev103受控输入和原native32候选，共135次；只给图像/当前候选及自身地址，不读取dev目标或参考。原生没有可信诊断金标，不算原生诊断准确率。calibration/locked图像与参考均不进入本轮。

诊断是独立适配器，但与refiner共享基座、训练文章和构造器，错误可能相关；没有独立专家视觉金标，不能以它当独立验收正确性证书，也不直接声称已完成Jev。

## 固定修复器的对照

使用既有含保持all381-step终态、描述式提示、相同图像/当前候选、greedy192token和单原子动作。每个输入都调用，不靠真值门控省调用：

- 无诊断：复用冻结描述式103次和native32次；先核对checkpoint/软件/源码与数据身份。
- learned：附加严格解析后的模型诊断；诊断失败附加unknown，仍保留成本与回退记录。103受控＋32native。
- displaced_region：保留预测verdict/error，只将invalid范围按固定哈希移到不同的合法DOM位置；valid/unknown不变。103受控；如候选无替代位置则记录未改变，不筛除。
- oracle_controlled：从已发布受控目标投影理想verdict/error/region，**只用于明确的标签辅助上限**，不提供正确文本/span。103受控，不能描述为部署或学长verifier实绩。

新增refiner341次、diagnosis135次，共476次；真实prompt长度与完整双组件成本记录，不把更多token的收益归为纯架构因果。learned/displaced尤其用来检验位置提示，改变索引也可能改变少量token。无诊断复用数据须与同冻结条件一致。

严格hint schema按当前候选重算，附加说明诊断可能错误、仍依据源图，只改支持差异；不把诊断作为系统高优先指令。所有生成先冻结，再离线动作解析/渲染/参考评分；提示与动作绑定不可借兼容路径绕开源图或预算检查。

## 指标和停止决策

受控诊断：合法率、verdict准确率、含合法性的strict joint、invalid类型/位置、各构造类型、valid误报和bad漏报，全部103分母。Gold只表示构造目标，不是独立视觉正确标签。

下游：保留全部四条件103候选/32文档，原native无诊断/learned各32源；TEDS/TEDS-S、全修复、原满分退化、规范化修改、失败/cap、每组件调用/token/秒/渲染。按文章配对10000bootstrap/seed20261007描述区间；单seed事后比较，不宣称确认性显著或1%风险保证。学长公式verifier材料不用于新模型目标或冒充已接入。

若理想诊断不能解决数字/跨度，优先训练位置/证据更明确的修复或补真实错误教师监督；若理想有效但learned无效，先改善诊断。只有独立验收与可用提议成立后，再做同预算RL与continued SFT、多seed和locked迁移。五方向目标仍未完成。

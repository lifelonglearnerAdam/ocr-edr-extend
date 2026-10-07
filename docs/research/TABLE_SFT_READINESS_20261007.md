# 表格SFT准入、掩码与评估准备

2026-10-07。范围为[固定表格协议](TABLE_SFT_SCREEN_PROTOCOL_20261007.md)的执行准备。训练更新仍为0；软件验证、模型基座推理及SFT效果是不同证据层次。本记录不产生表格训练提升结论。

## 已验证的数据和预算

原始128文档/410例保持冻结。按已记录的源图审查排除p0002整族4例，得到127文档/406例，不替换源、不重写参考。其他未独立核实的标注仍是弱监督。model-dev的32文档/103例全部保留；calibration/locked不进入训练或本轮推理。

全部406训练例完成Qwen processor预检查：prefix逐token一致，loss仅覆盖assistant目标与结束token，无静默截断。最大processed sequence为2,708，协议上限4,096；单条监督token范围7–114。103条model-dev源身份已校验，未优化。

| 固定消融臂 | 文档 | 每文档暴露 | 总暴露 | 计划optimizer步 | processed token | assistant监督token |
|---|---:|---:|---:|---:|---:|---:|
| all | 127 | 12 | 1,524 | 381 | 2,087,582 | 24,181 |
| no_explicit_preservation | 127 | 12 | 1,524 | 381 | 2,133,604 | 30,406 |

完整臂保持/数字/extra-row各483次，span75次；去显式保持臂数字/extra-row各712次，span100次。此对照匹配文档暴露和优化器步数，**没有匹配监督token或修复变体暴露量**。后者不能被隐去，也不能将结果解释为在一切训练成本相同条件下保持目标的纯效应。

准入train SHA256为`89a81398c23e72dea77682c9dd9a48b55bc0b836d8bbf25008b9fd56c25a4b42`，model-dev监督SHA256为`e7c32d34765b7e540820d7b1f199a679643edb89f9910be6124eb22b2715ad03`。原始准入运行源码在扩展加载器前已归档，回放应使用该归档及其hash，不能将当前模块哈希冒充原始执行版本。

## 可执行路径与检验

`train_table_sft.py`使用表格专用准入/角色检查，再调用共享LoRA内核。共享函数体经AST比较与已执行的公式SFT内核相同；原公式执行源码保留。GPU或依赖初始化失败会保留failed receipt和0完成步，不能当作训练成功。

`run_table_sft_screen.py`只加载五字段model-dev输入、源身份inventory和源图。提示中的地址从当前HTML计算；不读取参考、target action、known error address或角色标签进入模型。完整103例原始调用先冻结。SFT模型必须来自相同配置、准入快照且完成固定381步；checkpoint逐文件核验，公式adapter不能混入。

`evaluate_table_sft_screen.py`随后离线执行严格单JSON动作。截断、重复JSON字段、错误地址、格式或渲染失败均回退原输入，并保留调用、token和时间成本。原始参考与正确动作仅在此阶段用于评分/诊断。官方TEDS、TEDS-S、逐案例均值、逐PMC文档均值、目标地址匹配、完整动作匹配、保持退化、回退原因分别报告。动作地址正确但文本错误不算完整修复。

配对比较要求同输入、模型、配置、生成源码、PyTorch/Transformers/Pillow、设备与精度；GPU条件还匹配型号/CUDA runtime，两个SFT臂另匹配准入hash及PEFT版本。独立代码审查发现并修复了后一组比较遗漏，真实CLI拒绝测试在修复前失败、修复后通过。

本轮全环境117项测试通过，无集成跳过；Ruff/Black通过。包含真实TeX/表格渲染、未改动官方TEDS/CDM回放、完整CLI回放和失败边界。生成时间不包括模型加载和图像/提示预处理；渲染重放时间使用内容缓存，单独报告，不能冒充端到端部署时延。

## 模型开发源图检查

在未查阅此次模型输出的情况下，对全部32个model-dev源/发布参考做单轮assistant视觉检查。未确认新的整行或数值遗漏；这不是独立人工逐字符审计，不提供32个视觉Good金标。

- p0139的两段DNA序列在源图分行，在发布参考HTML中直接拼接；记录表示边界差异，不宣称碱基转录错误。
- p0140源图本身写`1000 km³`，即使“area”使单位看起来反常，也不能用常识覆盖可见源内容。
- p0153的面板级疑似缺值经原始PNG、PDF文本、准备HTML和原包第204012行标注复核后排除；未认定标注缺陷。

本轮没有修改model-dev参考、过滤案例或观察校准/锁定图像。统计主表始终保留103例；风格、换行和未确认字符的不确定性与官方metric一致率分开解释。

## 复现命令

从仓库根运行；先具备已核验数据、模型receipt和兼容的独立依赖环境。`task_python`指经过版本检查的解释器，模型路径必须以固定revision命名。两臂输出使用新的目录。训练前需现场核对GPU空闲显存及进程占用。

```bash
task_python=.venv-pilot/bin/python
task_dataset=data/processed/pubtabnet-four-roles-20261007
task_admission="$task_dataset/admitted-supervision"
task_model=data/raw/model-cache/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c
task_model_receipt=experiments/runs/server-selection-20261006/model-receipt.json
task_run=experiments/runs/table-sft-screen-20261007

# On a freshly checked available GPU; execute both arms from the same base.
"$task_python" scripts/train_table_sft.py \
  --config configs/train/sft_table_screen.yaml --dataset "$task_dataset" \
  --admission "$task_admission" --model-path "$task_model" \
  --model-receipt "$task_model_receipt" --arm all --output "$task_run/training/all"

# Base inference can run on CPU. All paired adapter conditions use the same runtime.
"$task_python" scripts/run_table_sft_screen.py \
  --config configs/train/sft_table_screen.yaml --dataset "$task_dataset" \
  --model-path "$task_model" --model-receipt "$task_model_receipt" \
  --condition base --device cpu --output "$task_run/inference/base"

# For an SFT condition, additionally pass its completed --adapter-run and --admission.
# Only completed full-coverage run directories are accepted by the evaluator.
"$task_python" scripts/evaluate_table_sft_screen.py \
  --dataset "$task_dataset" --runs "$task_run/inference/base" \
  --official-root data/raw/omnidocbench-source --output "$task_run/base-evaluation"
```

同样以`--arm no_explicit_preservation`执行第二训练臂；所有SFT推理的生成源码应与base冻结版本相同。这里给出命令不表示两个训练作业已执行。已为一台校园3090服务器独立暂存159张train/model-dev图、准入文件和固定模型；逐文件哈希通过。服务器已有工作负载，本轮未启动表格训练。

[公开准备证据](../../experiments/artifacts/table-sft-readiness-20261007/README.md)仅包含协议、token计数、hash、源码及有边界的审查记录，不包含原图、原始HTML、凭据或模型权重。后续仍需完整基座/SFT结果、具名原生表格parser、独立视觉验收、教师轨迹、Jev/同证据规则比较、GRPO及锁定验证。

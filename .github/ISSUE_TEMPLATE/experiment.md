name: 实验任务 Experiment
description: 新实验 / 消融 / 复现任务
title: "[exp] "
labels: ["experiment"]
body:
  - type: textarea
    id: goal
    attributes:
      label: 目标
      description: 要验证的假设或要得到的数字
    validations:
      required: true
  - type: textarea
    id: setup
    attributes:
      label: 配置
      description: 模型、数据、脚本路径、超参
    validations:
      required: true
  - type: dropdown
    id: modality
    attributes:
      label: 模态
      options: ["公式", "表格", "公式+表格", "正文对照", "混合"]
    validations:
      required: true
  - type: input
    id: deadline
    attributes:
      label: 期望完成时间
  - type: textarea
    id: success
    attributes:
      label: 成功标准
      description: 例如 CDM +2pt on Bad subset

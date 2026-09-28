name: Bug / 复现问题
description: 代码错误、评测失败、复现不一致
title: "[bug] "
labels: ["bug"]
body:
  - type: textarea
    id: what
    attributes:
      label: 现象
    validations:
      required: true
  - type: textarea
    id: repro
    attributes:
      label: 复现步骤
      description: 命令、配置、commit
    validations:
      required: true
  - type: textarea
    id: expect
    attributes:
      label: 期望行为
  - type: input
    id: env
    attributes:
      label: 环境（GPU / CUDA / commit）

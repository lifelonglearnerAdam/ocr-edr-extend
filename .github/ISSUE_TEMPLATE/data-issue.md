name: 数据问题
description: 标注错误、缺失、泄漏风险
title: "[data] "
labels: ["data"]
body:
  - type: dropdown
    id: type
    attributes:
      label: 类型
      options: ["标注错误", "样本缺失", "泄漏 / 污染", "渲染等价争议", "其他"]
  - type: textarea
    id: detail
    attributes:
      label: 详情
      description: 样本 ID、路径、截图说明
    validations:
      required: true
  - type: checkboxes
    id: leak
    attributes:
      label: 是否涉及评测集泄漏
      options:
        - label: 与 OmniDocBench / OCRErrBench / UniMER 有重叠风险

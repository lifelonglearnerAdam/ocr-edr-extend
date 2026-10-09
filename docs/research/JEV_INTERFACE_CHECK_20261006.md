# Jev 接口核查：当前模型不能直接读取图片

检索日期：2026-10-06。实际读取了发布方[模型页](https://docs.typesafe.ai/models)和[置信度页](https://docs.typesafe.ai/confidence)，HTTP200；API参考页的一次请求超时，未把它算作已读取。页面快照及访问记录保存在本地 `experiments/runs/sft-screen-20261006/vendor-preflight/`。

## 官方文档可支持的事实

模型页列出 `jev-1.13.0`，输入为文本、JSON对象或文本数组，明确写明 **No image, audio, or video input**。文档推荐将非文本输入先转成文本/结构字段。`jev-latest`等别名会指向版本号，因此实验应固定具体model ID并保留响应/访问时间，不能假设别名不变。

Choice返回的`confidence`不是未经转换的最大标签概率。n个选项时文档公式为

\[
c=\frac{p_{\max}-1/n}{1-1/n}.
\]

例如3选项confidence=0.9对应pmax约0.9333；2选项对应0.95。这个分布统计量本身不等于 OCR 任务正确率，也不能直接作已校准的视觉风险证据。应同时保存完整probabilities、厂商confidence、pmax、选项数和独立校准结果。

## 对本项目的直接修正

当前不能把 `(source image, current rendering)` 直接交给这个Jev模型做视觉验收。H3 保留，但具体研究接口改为**结构化证据路由**：合法性、编辑范围、局部拓扑变化和可追溯的视觉证据由确定性程序/独立视觉模块生成；Jev读取这些状态输出accept/reject/escalate。证据生成会有错误和成本，Jev无法补回未提供的视觉信息，必须测试完整链路而非只测文本路由准确率。

比较：(a)证据上的固定规则/阈值；(b)Jev读取同一证据；(c)独立视觉judge；(d)Jev路由加升级。各臂把生成视觉证据、拒绝候选和升级的全部成本计入。不能给Jev额外参考答案或oracle特征；不能把其他VLM实验标为Jev效果。

早期文献笔记中的Visual Jev是文献方向，不能据此断言当前托管端点具备视觉接口。没有调用收费模型，账户/API许可、额度、返回格式与视觉模块尚未实测；H3的真实实验仍未完成。本文档核查的是接口/语义，不是OCR效果或全面版本可用性证明。

# 共享研究进展页

在线地址：<https://lifelonglearnerAdam.github.io/ocr-edr-extend/>。

`index.html`是可离线打开的单文件报告，包含内嵌数据和已核对图表，无外部字体/JS依赖。`research-data.json`保存公开数值快照；`senior-provenance.json`标记学长材料来源、转录数字和未独立复现的边界。

更新流程：

```bash
.venv-pilot/bin/python scripts/update_research_dashboard.py
.venv-pilot/bin/python scripts/build_research_dashboard.py
```

只有完整且hash相符的实验进入成果数字，运行进度来自最近读取的阶段回执。已发布页面每60秒检查同目录JSON快照是否更新；本地证据变化后生成/提交快照，GitHub Pages workflow自动部署。它不是浏览器直接连接训练GPU，离线下载版保留对应时间的快照。

当前用户已明确授权及时上传更新，可启用本地观察器：

```bash
.venv-pilot/bin/python scripts/watch_research_dashboard.py \
  --publish --interval 60 \
  --receipt experiments/runs/research-dashboard-20261009/watcher.json
```

观察器只提交生成的HTML/JSON，不携带其他已暂存文件；没有证据/模板变化时不制造时间戳提交。推送冲突或验证失败会停并保留回执，不force-push。新增实验/学长材料须先完成來源核对、协议与输入绑定，再接入更新器；原始失败、限制和归属均保留。

页面按“几分钟结论 / 模块与实验详细解释 / 证据复现”三个层次组织，包含指标术语、表格原负结果、固定checkpoint提示消融、公式/原生parser/教师记录及H1–H5路线。学长核验百分比与修复TEDS不直接比较，原PDF私人Windows路径不分发。

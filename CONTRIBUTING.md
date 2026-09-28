# 贡献指南

## 分支模型

| 分支 | 用途 |
|------|------|
| `main` | 可复现、评测通过的稳定版 |
| `dev` | 日常集成 |
| `feat/*` | 功能 / 实验分支 |
| `exp/*` | 一次性消融、可不合并 |

流程：`feat/xxx` → 开 PR 到 `dev` → 实验通过后 `dev` → `main`。

## 提交约定

```
<type>(<scope>): <short summary>

type: feat | fix | data | eval | train | docs | chore
scope: formula | table | judge | render | grpo | distill | omidocbench | ...
```

示例：

- `feat(judge): add Jev cascade verifier with confidence gate`
- `eval(omnidocbench): formula CDM on PaddleOCR-VL Bad subset`
- `train(grpo): reward terms for preserve vs repair branches`

## Issue 标签建议

- `experiment`：待跑实验
- `data`：数据构造 / 标注
- `bug`：代码或复现问题
- `judge`：Jev / reward 相关
- `blocked`：被资源 / 许可证 / 算力阻塞

## PR 要求

1. 关联 Issue
2. 写清：改了什么、为何、如何验证
3. 实验类 PR 附配置路径 + 关键指标表
4. 不提交权重、原始数据、密钥（用网盘 / HF 链接写在 `data/README.md`）

## 代码风格

- Python：`ruff` + `black`（或团队统一）
- 配置优先 YAML；命令行覆盖
- 评测脚本必须可单独运行，不依赖训练环境

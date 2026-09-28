#!/usr/bin/env python
"""OmniDocBench 公式/表格评测入口骨架。"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pred-dir", default=None, help="覆盖配置中的预测目录")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(f"[eval] benchmark={cfg['benchmark']['name']} scope={cfg['scope']}")
    print("[TODO] 接入官方评测脚本，输出公式 CDM/Case-F1、表格 TEDS、闭环 Preserve/ExactFix/VisFix")
    print("[TODO] 同时报告全集与 Bad 子集")


if __name__ == "__main__":
    main()

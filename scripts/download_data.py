#!/usr/bin/env python
"""下载评测数据（OmniDocBench 等）。骨架：请按官方仓库说明填写。"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default="omnidocbench", choices=["omnidocbench", "ocerrbench", "unimer"])
    parser.add_argument("--dest", default="data")
    args = parser.parse_args()
    dest = Path(args.dest) / args.bench
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[TODO] 从官方源下载 {args.bench} 到 {dest}")
    print("参考：")
    print("  OmniDocBench: https://github.com/opendatalab/OmniDocBench")
    print("  请将下载说明与镜像链接写在 data/README.md")


if __name__ == "__main__":
    main()

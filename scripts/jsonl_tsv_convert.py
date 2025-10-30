#!/usr/bin/env python3
"""
jsonl_tsv_convert.py

シンプルな JSONL ⇔ TSV 相互変換ツール。

機能:
- JSONL → TSV: 各行のJSONオブジェクトのキーの和集合を列にしてTSV出力。
- TSV → JSONL: 各行をJSONオブジェクトとして1行ずつ出力。
- UTF-8 デフォルト。

注意:
- ネストしたオブジェクト/配列は JSON 文字列にしてセルへ格納します。
- 列順は「最初に見つかった順」を維持しつつ、新規キーは末尾へ追加します。

使用方法 (Usage):
- 変換方向は拡張子で自動判定されます。

1) JSONL → TSV
    python3 scripts/jsonl_tsv_convert.py -i data/sample/faq_dataset.jsonl -o output.tsv

2) TSV → JSONL
    python3 scripts/jsonl_tsv_convert.py -i input.tsv -o output.jsonl

オプション:
- --encoding: 入出力の文字エンコーディング（既定: utf-8）
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Set


def detect_direction(input_path: str) -> str:
    _, ext = os.path.splitext(input_path)
    ext = ext.lower()
    if ext in {".jsonl", ".ndjson"}:
        return "jsonl-to-tsv"
    if ext == ".tsv":
        return "tsv-to-jsonl"
    # デフォルトは JSONL→TSV とする
    return "jsonl-to-tsv"


def open_input(path: str, encoding: str) -> io.TextIOBase:
    return open(path, "r", encoding=encoding, newline="")


def open_output(path: str, encoding: str) -> io.TextIOBase:
    # newline="" は csv.writer で改行が二重にならないようにする
    return open(path, "w", encoding=encoding, newline="")


def to_scalar_or_json(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, bool):
        return value
    # dict / list / その他は JSON 文字列に変換
    return json.dumps(value, ensure_ascii=False)


def jsonl_to_csv(
    input_stream: io.TextIOBase,
    output_stream: io.TextIOBase,
) -> None:
    field_order: List[str] = []
    seen_fields: Set[str] = set()
    rows: List[Dict[str, object]] = []

    for line_num, line in enumerate(input_stream, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSONL の {line_num} 行目でパースに失敗しました: {e}") from e
        if not isinstance(obj, dict):
            raise ValueError(f"JSONL の {line_num} 行目はオブジェクトではありません")

        # 新規キーを順序付きで収集
        for key in obj.keys():
            if key not in seen_fields:
                seen_fields.add(key)
                field_order.append(key)

        # 値はTSV向けに整形
        row = {k: to_scalar_or_json(obj.get(k)) for k in obj.keys()}
        rows.append(row)

    writer = csv.DictWriter(
        output_stream,
        fieldnames=field_order,
        delimiter='\t',
        quoting=csv.QUOTE_MINIMAL,
        extrasaction="ignore",
    )
    writer.writeheader()
    for row in rows:
        # 欠損キーは空文字にする（Noneは空セルとして出力される）
        safe_row = {k: row.get(k, None) for k in field_order}
        writer.writerow(safe_row)


def csv_to_jsonl(
    input_stream: io.TextIOBase,
    output_stream: io.TextIOBase,
) -> None:
    reader = csv.DictReader(input_stream, delimiter='\t')
    for line_num, row in enumerate(reader, start=2):  # 1行目はヘッダ
        # 空文字は None にするかどうかは今後の拡張に委ね、まずは空文字のまま
        # 数値や真偽値への自動変換は行わず文字列のまま保持
        # 必要に応じて後でオプション追加
        safe_obj: Dict[str, object] = {}
        for k, v in row.items():
            # None は空セル（ヘッダはあるがデータが空）
            if v is None:
                safe_obj[k] = None
            else:
                safe_obj[k] = v
        json_record = json.dumps(safe_obj, ensure_ascii=False)
        output_stream.write(json_record + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="JSONL と TSV の相互変換ツール",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="入力ファイルパス (拡張子で方向自動判定)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="出力ファイルパス",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="入出力の文字エンコーディング (デフォルト: utf-8)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    direction = detect_direction(args.input)
    
    try:
        with open_input(args.input, args.encoding) as in_f, open_output(args.output, args.encoding) as out_f:
            if direction == "jsonl-to-tsv":
                jsonl_to_csv(in_f, out_f)
            elif direction == "tsv-to-jsonl":
                csv_to_jsonl(in_f, out_f)
            else:
                raise ValueError(f"不明な direction: {direction}")
    except Exception as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

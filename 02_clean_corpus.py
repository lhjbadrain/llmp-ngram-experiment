"""清洗1998年上半年人民日报带音语料；原始文件只读。

运行：python .\02_clean_corpus.py
输出：data/cleaned/*.txt（正文）、*.jsonl（含文章编号的正文），
      data/results/cleaning_stats.json、cleaning_examples.json、cleaning_report.md。
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time

BASE = Path(__file__).resolve().parent
# 编号只在行首匹配；前三段标识文章，最后一段标识段落。
RECORD_ID = re.compile(r"^(\d{8}-\d{2}-\d{3})-(\d{3})\s+")
PRONUNCIATION = re.compile(r"\{[^{}]*\}")
POS_SUFFIX = re.compile(r"/([A-Za-z]+)(?:\][A-Za-z]+)?$")


def clean_line(text):
    """将一行原始记录转换为编号及词列表；编号不混入模型正文。"""
    match = RECORD_ID.match(text)
    if match is None:
        raise ValueError("非空行缺少预期记录编号，需要人工检查")
    body = text[match.end():]
    body, removed = PRONUNCIATION.subn("", body)
    if "{" in body or "}" in body:
        raise ValueError("存在未配对花括号，需要人工检查")
    tokens = body.split()
    # 本次实检没有词性或复合词标记，遇到此类格式先提示，不盲目删除。
    if any(POS_SUFFIX.search(w) or "[" in w or "]" in w for w in tokens):
        raise ValueError("遇到未经核实的词性/复合标记，需要人工检查")
    return match.group(0).strip(), match.group(1), tokens, removed


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=BASE / "1998-上半年-带音")
    parser.add_argument("--output", type=Path, default=BASE / "data" / "cleaned")
    args = parser.parse_args()
    corpus, output = args.corpus.resolve(), args.output.resolve()
    if not corpus.is_dir():
        raise FileNotFoundError(f"语料目录不存在：{corpus}")
    if output == corpus or corpus in output.parents or output in corpus.parents:
        raise ValueError("输出目录不能与原语料目录重合或互相包含")
    sources = sorted(corpus.glob("*.txt"))
    expected = [f"1998-{m:02d}-2003版-带音.txt" for m in range(1, 7)]
    if [p.name for p in sources] != expected:
        raise ValueError(f"预期六个月文件：{expected}；实际：{[p.name for p in sources]}")

    output.mkdir(parents=True, exist_ok=True)
    results = output.parent / "results"
    results.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    stats = {
        "source_directory": str(corpus), "output_directory": str(output),
        "encoding_in": "gb18030", "encoding_out": "utf-8",
        "python": sys.version, "platform": platform.platform(),
        "rules": ["删除行首记录编号（编号另存元数据）", "删除成对花括号注音", "按现有空白切词并规范为单空格", "保留标点及全角数字", "跳过解码失败或含替换字符的行并记录", "不去重，不划分训练/验证/测试集"],
        "files": [], "skipped_lines": [],
    }
    totals = Counter()
    examples = []
    source_hashes = {}
    for path in sources:
        raw = path.read_bytes()
        source_hashes[path] = hashlib.sha256(raw).hexdigest()
        month = path.name[:7]
        counts = Counter()
        article_ids = set()
        month_examples = 0
        txt_path, jsonl_path = output / f"{month}.txt", output / f"{month}.jsonl"
        with txt_path.open("w", encoding="utf-8", newline="\n") as plain, jsonl_path.open("w", encoding="utf-8", newline="\n") as structured:
            for number, raw_line in enumerate(raw.splitlines(), 1):
                counts["source_lines"] += 1
                try:
                    text = raw_line.decode("gb18030", errors="strict")
                    reason = "原文包含Unicode替换字符" if "\ufffd" in text else None
                except UnicodeDecodeError as exc:
                    text = raw_line.decode("gb18030", errors="replace")
                    reason = f"GB18030解码失败：{exc}"
                if reason:
                    counts["skipped_encoding_lines"] += 1
                    stats["skipped_lines"].append({"source": path.name, "line": number, "reason": reason, "preview": text, "raw_hex": raw_line.hex()})
                    continue
                if not text.strip():
                    counts["blank_lines"] += 1
                    continue
                try:
                    record_id, article_id, tokens, removed = clean_line(text)
                except ValueError as exc:
                    raise ValueError(f"{path.name} 第{number}行：{exc}") from exc
                counts["removed_record_ids"] += 1
                counts["removed_pronunciations"] += removed
                if not tokens:
                    counts["empty_after_cleaning"] += 1
                    continue
                cleaned = " ".join(tokens)
                plain.write(cleaned + "\n")
                structured.write(json.dumps({"id": record_id, "article_id": article_id, "source_line": number, "tokens": tokens}, ensure_ascii=False) + "\n")
                counts["output_paragraphs"] += 1
                counts["output_tokens"] += len(tokens)
                article_ids.add(article_id)
                if month_examples == 0 or (removed and month_examples < 3):
                    examples.append({"source": path.name, "source_line": number, "before": text, "after": cleaned, "removed_pronunciations": removed})
                    month_examples += 1
        counts["articles_with_output"] = len(article_ids)
        totals.update(counts)
        stats["files"].append({"source": path.name, "source_bytes": len(raw), "source_sha256": source_hashes[path], **dict(counts), "clean_txt": str(txt_path), "clean_txt_sha256": hashlib.sha256(txt_path.read_bytes()).hexdigest(), "clean_jsonl": str(jsonl_path), "clean_jsonl_sha256": hashlib.sha256(jsonl_path.read_bytes()).hexdigest()})
        print(f"{month}: {counts['output_paragraphs']:,}段，{counts['output_tokens']:,}词，去注音{counts['removed_pronunciations']:,}处，跳过编码异常{counts['skipped_encoding_lines']}行")

    # 核对行数守恒、输出中不残留标记、两种输出行数相同、原始文件哈希不变。
    verified = 0
    for entry in stats["files"]:
        paragraphs = 0
        with Path(entry["clean_txt"]).open(encoding="utf-8") as handle:
            for line in handle:
                if "{" in line or "}" in line or RECORD_ID.match(line) or "\ufffd" in line:
                    raise RuntimeError("输出残留标记或替换字符")
                paragraphs += 1
        with Path(entry["clean_jsonl"]).open(encoding="utf-8") as handle:
            structured_count = sum(1 for _ in handle)
        if paragraphs != entry["output_paragraphs"] or paragraphs != structured_count:
            raise RuntimeError("输出段落数核对失败")
        if entry["source_lines"] != sum(entry.get(k, 0) for k in ("output_paragraphs", "skipped_encoding_lines", "blank_lines", "empty_after_cleaning")):
            raise RuntimeError("输入/输出行数核对失败")
        verified += paragraphs
    for path, original_hash in source_hashes.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != original_hash:
            raise RuntimeError(f"原文件哈希发生变化：{path}")
    stats["totals"] = dict(totals)
    stats["verification"] = {"source_hashes_unchanged": True, "line_accounting_ok": True, "no_residual_markers": True, "verified_output_paragraphs": verified}
    stats["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    write_json(results / "cleaning_stats.json", stats)
    write_json(results / "cleaning_examples.json", examples)
    report = ["# 真实语料清洗记录", "", "这是数据准备记录，尚未训练模型或生成文本。", "", "## 处理规则", ""]
    report += [f"- {rule}" for rule in stats["rules"]]
    report += ["", "清理编号和注音是为了让词频、词对频次对应自然文本。正文中未检出常见POS/复合标签，因此没有对普通文字套用额外删除规则。", "", "每一输出行仍是原来的一个段落，不把段落误称为句子。JSONL单独保留记录/文章编号，后续可按文章划分数据。当前输出是全量清洗语料，不能直接当作独立测试集。", "", "## 实测统计", "", "| 月份 | 原始行 | 输出段落 | 词数 | 去注音 | 编码异常行 |", "|---|---:|---:|---:|---:|---:|"]
    for entry in stats["files"]:
        report.append(f"| {entry['source'][:7]} | {entry['source_lines']:,} | {entry['output_paragraphs']:,} | {entry['output_tokens']:,} | {entry['removed_pronunciations']:,} | {entry.get('skipped_encoding_lines', 0)} |")
    report += ["", f"合计：{totals['output_paragraphs']:,}段，{totals['output_tokens']:,}词，跳过编码异常{totals['skipped_encoding_lines']}行。", "", "编码异常行不猜测修复；完整行预览、原始字节十六进制和位置见cleaning_stats.json。", "", "原始文件处理前后SHA256一致；输入行数去向、输出段落数及标记残留检查通过。", "", "## 清洗前后示例", ""]
    for example in examples[:3]:
        report += [f"来源：{example['source']} 第{example['source_line']}行", "", "清洗前：", "", "```text", example["before"], "```", "", "清洗后：", "", "```text", example["after"], "```", ""]
    report += ["## 复现", "", "```powershell", "python .\\02_clean_corpus.py", "```", "", "重复运行会重新生成清洗输出和记录，不修改原始语料。运行环境、源文件及输出文件SHA256见cleaning_stats.json。"]
    (results / "cleaning_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"合计：{totals['output_paragraphs']:,}段，{totals['output_tokens']:,}词。")
    print("检查通过：原始文件哈希不变；行数守恒；输出未残留编号、注音或替换字符。")
    print(f"清洗结果：{output}\n清洗报告：{results / 'cleaning_report.md'}")


if __name__ == "__main__":
    main()



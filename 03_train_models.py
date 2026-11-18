"""真实数据抽样与训练：python .\03_train_models.py

默认取完整文章，目标20万训练词、各2万验证/测试词；最后一篇可超预算。
使用NLTK WittenBellInterpolated；不在本步骤评测困惑度或生成文本。
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys
import time

import nltk
from ngram_model import BOS, EOS, UNK, make_model, training_ngrams

BASE = Path(__file__).resolve().parent
SEED = 20261118


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_split(months, budget, seen, split, output):
    articles = defaultdict(list)
    sources = []
    for month in months:
        path = BASE / "data" / "cleaned" / f"1998-{month:02d}.jsonl"
        sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                articles[record["article_id"]].append(record)
    # 哈希排序避免总取月初新闻；固定种子使文章选择可复现。
    ranked = sorted(articles, key=lambda a: (hashlib.sha256(f"{SEED}:{a}".encode()).hexdigest(), a))
    selected, selected_ids = [], []
    total = removed = 0
    for article_id in ranked:
        retained = []
        for record in articles[article_id]:
            digest = hashlib.sha256(" ".join(record["tokens"]).encode("utf-8")).hexdigest()
            if digest in seen:
                removed += 1
                continue
            seen.add(digest)
            retained.append(record)
        if retained:
            selected_ids.append(article_id)
            selected.extend(retained)
            total += sum(len(r["tokens"]) for r in retained)
        if total >= budget:
            break
    if total < budget:
        raise ValueError(f"{split}可用语料不足预算")
    with (output / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output / f"{split}.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for record in selected:
            handle.write(" ".join(record["tokens"]) + "\n")
    metadata = {"months": months, "target_tokens": budget, "tokens": total,
                "paragraphs": len(selected), "articles": len(selected_ids),
                "article_ids": selected_ids, "exact_duplicates_skipped_while_selecting": removed,
                "sources": sources,
                "selected_sha256": hashlib.sha256((output / f"{split}.jsonl").read_bytes()).hexdigest()}
    print(f"{split}: {len(selected_ids)}篇文章，{len(selected):,}段，{total:,}词", flush=True)
    return selected, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-tokens", type=int, default=200000)
    parser.add_argument("--heldout-tokens", type=int, default=20000)
    args = parser.parse_args()
    if args.train_tokens < 1 or args.heldout_tokens < 1:
        parser.error("词数预算必须为正数")
    split_dir, model_dir, result_dir = BASE / "data/splits", BASE / "models", BASE / "results"
    for directory in (split_dir, model_dir, result_dir):
        directory.mkdir(parents=True, exist_ok=True)
    seen = set()
    data, manifest = {}, {"seed": SEED, "selection": "文章ID固定盐值SHA256排序，按完整文章选取；选取过程中删除完全重复段落", "splits": {}}
    for name, months, budget in (("train", [1, 2, 3, 4], args.train_tokens), ("valid", [5], args.heldout_tokens), ("test", [6], args.heldout_tokens)):
        data[name], manifest["splits"][name] = prepare_split(months, budget, seen, name, split_dir)
    article_sets = [set(manifest["splits"][s]["article_ids"]) for s in ("train", "valid", "test")]
    if any(article_sets[i] & article_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("文章编号交叉")
    save_json(result_dir / "split_manifest.json", manifest)

    frequencies = Counter(w for r in data["train"] for w in r["tokens"])
    if any(w in frequencies for w in (BOS, EOS, UNK)):
        raise ValueError("正文与保留的模型特殊符号冲突")
    # 频次至少2；按频次降序、同频按字典序，最多保留20000个普通词。
    kept_words = [w for w, c in sorted(frequencies.items(), key=lambda pair: (-pair[1], pair[0])) if c >= 2][:20000]
    kept = set(kept_words)
    train_paragraphs = [[w if w in kept else UNK for w in r["tokens"]] for r in data["train"]]
    unk_count = sum(p.count(UNK) for p in train_paragraphs)
    if unk_count == 0:
        raise ValueError("UNK没有训练观测，需要调整词表策略")
    vocabulary_record = {"min_count": 2, "max_ordinary_words": 20000, "ordinary_words": kept_words,
                         "special_tokens": [BOS, EOS, UNK], "training_unk_tokens": unk_count}
    save_json(result_dir / "vocabulary.json", vocabulary_record)
    vocab_hash = hashlib.sha256((result_dir / "vocabulary.json").read_bytes()).hexdigest()
    metrics = {"run_time_local": datetime.now(timezone(timedelta(hours=8))).isoformat(),
               "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
               "nltk_version": nltk.__version__, "tool": "NLTK WittenBellInterpolated",
               "training_timing_scope": "仅model.fit及其遍历n-gram的时间；不含分词、划分、词表、保存、检查和评测",
               "vocabulary_sha256": vocab_hash, "vocabulary_size_including_specials": len(kept_words) + 3,
               "shared_training_paragraphs": len(train_paragraphs), "training_unk_tokens": unk_count,
               "runs": [], "evaluation_status": "尚未评测困惑度或生成文本"}
    expected_predictions = sum(len(p) + 1 for p in train_paragraphs)
    for order in (2, 3, 4):
        model = make_model(order, kept_words)
        started = time.perf_counter()
        model.fit(training_ngrams(train_paragraphs, order))
        elapsed = time.perf_counter() - started
        types, observations = [], []
        for size in range(1, order + 1):
            counts = model.counts[size]
            types.append(len(counts) if size == 1 else sum(len(fd) for fd in counts.values()))
            observations.append(counts.N())
        if any(value != expected_predictions for value in observations):
            raise RuntimeError("各阶计数总数不一致")
        if model.counts[1][BOS] != 0 or model.counts[1][EOS] != len(train_paragraphs):
            raise RuntimeError("段落边界计数不正确")
        # 核对已见/未见历史的概率归一化，不用验证集或测试集调参。
        checks = []
        for context in ((), ("了",), (UNK,) * (order - 1)):
            probability_sum = sum(model.score(w, context) for w in model.vocab)
            if abs(probability_sum - 1.0) > 1e-8:
                raise RuntimeError(f"概率不归一化：{context}, {probability_sum}")
            checks.append({"context": list(context), "probability_sum": probability_sum})
        path = model_dir / f"{order}gram.pkl"
        saving_started = time.perf_counter()
        with path.open("wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        save_seconds = time.perf_counter() - saving_started
        with path.open("rb") as handle:
            restored = pickle.load(handle)
        if restored.score("的", ("了",)) != model.score("的", ("了",)):
            raise RuntimeError("保存再读取的概率不一致")
        context = tuple(["胜利", "召开", "了"][-(order - 1):])
        top_counts = model.context_counts(context).most_common(5)
        row = {"order": order, "train_seconds": round(elapsed, 6),
               "save_seconds": round(save_seconds, 6), "model_bytes": path.stat().st_size,
               "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "model_path": str(path), "ngram_types_by_order": types,
               "observations_by_order": observations, "probability_checks": checks,
               "example_context": list(context), "example_observed_followers": top_counts}
        metrics["runs"].append(row)
        save_json(result_dir / "training_metrics.json", metrics)
        print(f"{order}-gram: 训练{elapsed:.4f}秒，模型{path.stat().st_size / 1024**2:.2f} MiB，各阶类型数{types}", flush=True)
        del restored, model

    report = ["# 第一轮真实训练记录", "", "## 实验设置", "", "使用NLTK的WittenBellInterpolated（Witten–Bell插值平滑）；CPU运行。", "", "2/3/4-gram共用训练段落及词表。训练1—4月、验证5月、测试6月。按文章ID固定盐值哈希排序抽样，选完整文章直至预算；选择时对完全相同段落去重。近重复及同事件报道可能仍然存在。", "", "正文已分词，每行按段落处理。每段补n−1个BOS、一个EOS，不跨段落计数；BOS只作上下文、不作预测目标。每阶模型同时统计1至n阶。低频词及词表外词映射为UNK；词表只由训练集建立。", "", "## 数据规模", "", "| 集合 | 文章 | 段落 | 词数（未补边界） |", "|---|---:|---:|---:|"]
    for name, values in manifest["splits"].items():
        report.append(f"| {name} | {values['articles']} | {values['paragraphs']:,} | {values['tokens']:,} |")
    report += ["", f"共享词表大小：{len(kept_words) + 3:,}（含BOS/EOS/UNK）；训练集中映射到UNK的正文词：{unk_count:,}。", "", "## 实测训练结果", "", "| 阶数 | 训练秒 | pickle模型MiB | 各阶不同n-gram数量 |", "|---|---:|---:|---| 三种模型 |"]
    report[-1] = "|---|---:|---:|---|"
    for row in metrics["runs"]:
        report.append(f"| {row['order']} | {row['train_seconds']:.4f} | {row['model_bytes'] / 1024**2:.2f} | {row['ngram_types_by_order']} |")
    report += ["", "时间仅包含fit及n-gram遍历，不含数据准备、模型保存、归一化检查与评测。单次运行，不是重复测速的统计结论。模型大小是本机pickle文件大小，不是峰值内存。", "", "## 检查与当前状态", "", "文章ID集合互不相交；选中的完全重复段落已去重；各阶预测观测数一致；每段恰有一个EOS；三个历史的概率和检查通过；模型保存后重读检查通过。", "", "尚未评测验证/测试困惑度，尚未生成文本，因此本记录不能用于判断哪一阶生成质量最好。", "", "## 复现", "", "```powershell", "python .\\03_train_models.py", "```", "", "依赖：NLTK 3.9.1。重复执行会覆盖本轮划分、模型和训练记录。文件哈希、参数及文章列表见同目录JSON。"]
    (result_dir / "training_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"训练及检查完成，记录：{result_dir / 'training_report.md'}", flush=True)


if __name__ == "__main__":
    main()

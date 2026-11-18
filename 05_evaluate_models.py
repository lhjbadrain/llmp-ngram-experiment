"""留出集困惑度评测：python .\05_evaluate_models.py

统一词表、分词、数据划分及边界口径；评测不更新或保存模型。
"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import time

from ngram_model import BOS, EOS, UNK

BASE = Path(__file__).resolve().parent


def evaluate(model, records):
    started = time.perf_counter()
    negative_log_probability = 0.0
    predicted = raw_tokens = oov_tokens = highest_order_hits = 0
    minimum_probability = 1.0
    for row in records:
        original = row["tokens"]
        mapped = list(model.vocab.lookup(original))
        raw_tokens += len(original)
        oov_tokens += sum(word == UNK for word in mapped)
        padded = [BOS] * (model.order - 1) + mapped + [EOS]
        for index in range(model.order - 1, len(padded)):
            target = padded[index]
            context = tuple(padded[index - model.order + 1:index])
            probability = model.score(target, context)
            if not (0 < probability <= 1) or not math.isfinite(probability):
                raise RuntimeError(f"非法目标概率：{context} -> {target}: {probability}")
            negative_log_probability -= math.log(probability)
            minimum_probability = min(minimum_probability, probability)
            predicted += 1
            counts = model.counts[model.order]
            if context in counts and counts[context][target] > 0:
                highest_order_hits += 1
    if predicted != raw_tokens + len(records):
        raise RuntimeError("计分词数应等于正文词数加每段一个EOS")
    return {"paragraphs": len(records), "raw_body_tokens": raw_tokens,
            "predicted_tokens_including_eos": predicted, "oov_body_tokens": oov_tokens,
            "oov_rate_excluding_eos": oov_tokens / raw_tokens,
            "negative_log_probability_nats": negative_log_probability,
            "cross_entropy_bits": negative_log_probability / math.log(2) / predicted,
            "perplexity": math.exp(negative_log_probability / predicted),
            "highest_order_seen_targets": highest_order_hits,
            "highest_order_hit_rate_including_eos": highest_order_hits / predicted,
            "minimum_target_probability": minimum_probability,
            "evaluation_seconds": round(time.perf_counter() - started, 6)}


def main():
    result_dir = BASE / "results"
    training = json.loads((result_dir / "training_metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((result_dir / "split_manifest.json").read_text(encoding="utf-8"))
    heldout = {}
    for name in ("valid", "test"):
        path = BASE / "data/splits" / f"{name}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["splits"][name]["selected_sha256"]:
            raise RuntimeError(f"{name}划分文件与训练阶段记录不一致")
        heldout[name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    output = {"run_time_local": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "definition": "PPL=exp(-sum(log P(目标词|前n-1词))/计分目标数)",
              "scoring_rules": "每段独立；BOS只作历史；每段一个EOS计分；所有正文OOV映射到共享UNK；不屏蔽UNK或进行top-k/温度变换",
              "oov_denominator": "仅正文词，不含BOS/EOS",
              "highest_order_hit_definition": "目标词与最高阶历史形成的n-gram训练时出现过；含边界项和UNK映射，不等于纯历史覆盖率",
              "evaluation_timing_scope": "从遍历段落、词表映射到概率计算及统计，不含载入模型和读取文件",
              "vocabulary_sha256": training["vocabulary_sha256"],
              "split_sha256": {name: manifest["splits"][name]["selected_sha256"] for name in heldout},
              "runs": []}
    for trained in training["runs"]:
        path = Path(trained["model_path"])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != trained["model_sha256"]:
            raise RuntimeError("模型与训练记录哈希不一致")
        # 仅加载03脚本生成且已核对哈希的本地模型。
        model = pickle.loads(raw)
        before_counts = [model.counts[n].N() for n in range(1, model.order + 1)]
        row = {"order": model.order, "model_sha256": trained["model_sha256"]}
        for name, records in heldout.items():
            row[name] = evaluate(model, records)
            values = row[name]
            print(f"{model.order}-gram {name}: PPL={values['perplexity']:.4f}，正文OOV={values['oov_rate_excluding_eos']:.2%}，最高阶命中率={values['highest_order_hit_rate_including_eos']:.2%}", flush=True)
        after_counts = [model.counts[n].N() for n in range(1, model.order + 1)]
        if before_counts != after_counts:
            raise RuntimeError("评测改变了训练观测计数")
        if hashlib.sha256(path.read_bytes()).hexdigest() != trained["model_sha256"]:
            raise RuntimeError("评测改变了模型文件")
        output["runs"].append(row)
    for name in heldout:
        if len({r[name]["oov_body_tokens"] for r in output["runs"]}) != 1:
            raise RuntimeError("模型之间OOV映射不一致")
    output["verification"] = {"models_and_splits_match_training": True,
                              "counting_denominators_checked": True,
                              "all_target_probabilities_positive": True,
                              "training_observation_counts_unchanged": True,
                              "saved_models_unchanged": True,
                              "shared_oov_mapping": True}
    (result_dir / "evaluation_metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = ["# 留出集困惑度评测", "", "困惑度越低，表示在当前统一的数据、分词、词表与边界口径下，对真实后续词的平均预测更好；不能直接等同于指定开头的续写质量。", "", "PPL = exp(负对数概率总和 / 计分词数)。正文及每段一个EOS参与计分，BOS不参与计分。词表外词映射为UNK。评测使用原模型概率，不做温度/top-k变换，不屏蔽UNK。", "", "训练、词表、平滑及数据划分固定；没有根据测试集重新选参数或训练。", "", "| 阶数 | 验证PPL | 测试PPL | 测试最高阶命中率 |", "|---|---:|---:|---:|"]
    for row in output["runs"]:
        report.append(f"| {row['order']} | {row['valid']['perplexity']:.4f} | {row['test']['perplexity']:.4f} | {row['test']['highest_order_hit_rate_including_eos']:.2%} |")
    report += ["", "## 计分口径", ""]
    for name in heldout:
        values = output["runs"][0][name]
        report.append(f"- {name}：正文{values['raw_body_tokens']:,}词，{values['paragraphs']}段；加入EOS后计分{values['predicted_tokens_including_eos']:,}项；正文OOV {values['oov_body_tokens']:,}词，OOV率{values['oov_rate_excluding_eos']:.2%}（分母不含EOS）。")
    report += ["", "最高阶命中率指目标词加历史形成的最高阶n-gram训练中出现过，包括边界和UNK映射；不等于单独的历史覆盖率。", "", "较低困惑度可能同时受到分布匹配和UNK合并影响，不能与其他分词、词表、数据划分或EOS口径不同的报告直接比较。", "", "这是一次固定小规模划分的结果，不是重复抽样统计结论。完全相同段落已去重，近重复及同事件新闻仍可能存在。", "", "## 检查", "", "模型与划分哈希匹配训练记录；计分分母正确；所有目标词概率为正；训练观测计数及模型文件未改变；三种模型的OOV映射一致。", "", "## 复现", "", "```powershell", "python .\\05_evaluate_models.py", "```", "", "逐模型/逐集合结果、负对数概率和检查信息见evaluation_metrics.json。"]
    (result_dir / "evaluation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"完成：{result_dir / 'evaluation_report.md'}", flush=True)


if __name__ == "__main__":
    main()

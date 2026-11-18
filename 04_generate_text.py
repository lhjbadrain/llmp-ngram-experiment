"""指定开头续写：python .\04_generate_text.py

比较三种阶数与四组生成参数；固定种子；保留全部真实输出。
不重训模型，不读取验证/测试正文，不人工修改生成文本。
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import time

from ngram_model import BOS, EOS, UNK

BASE = Path(__file__).resolve().parent
PROMPT = "在阳光明媚的五月，我们学校胜利召开了"
PROMPT_TOKENS = "在 阳光 明媚 的 五月 ， 我们 学校 胜利 召开 了".split()
SEED = 20261118
MAX_NEW_TOKENS = 40
SETTINGS = [(0.0, 1), (0.7, 20), (1.0, 20), (1.0, 100)]


def has_local_loop(tokens):
    """检测长度1至6的词块是否在某处连续重复至少3次。"""
    for width in range(1, 7):
        for start in range(len(tokens) - 3 * width + 1):
            block = tokens[start:start + width]
            if block == tokens[start + width:start + 2 * width] == tokens[start + 2 * width:start + 3 * width]:
                return True
    return False


def rank_candidates(model, context, cache):
    # 不同参数遇到相同历史时复用评分；缓存只影响耗时，不改变概率。
    mapped = tuple(model.vocab.lookup(context))
    if mapped not in cache:
        scored = [(word, model.score(word, mapped)) for word in model.vocab if word not in (BOS, UNK)]
        if any(not math.isfinite(p) or p < 0 for _, p in scored):
            raise RuntimeError("模型出现非法概率")
        scored = [(word, p) for word, p in scored if p > 0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        cache[mapped] = scored[:100]
    return mapped, cache[mapped]


def generate(model, temperature, top_k, seed, cache):
    rng = random.Random(seed)
    history, generated, trace = list(PROMPT_TOKENS), [], []
    stop_reason = "max_new_tokens"
    for step in range(MAX_NEW_TOKENS):
        # n-gram只读取末尾n-1个词，前面的提示不会一直参与条件概率。
        context, ranked = rank_candidates(model, history[-(model.order - 1):], cache)
        if not ranked:
            raise RuntimeError("没有可用生成候选词")
        candidates = ranked[:top_k]
        if temperature == 0:
            word, raw_probability = candidates[0]
            sampling_probability = 1.0
        else:
            log_weights = [math.log(p) / temperature for _, p in candidates]
            maximum = max(log_weights)
            weights = [math.exp(value - maximum) for value in log_weights]
            index = rng.choices(range(len(candidates)), weights=weights, k=1)[0]
            word, raw_probability = candidates[index]
            sampling_probability = weights[index] / sum(weights)
        trace.append({"step": step + 1, "mapped_context": list(context), "selected": word,
                      "raw_model_probability": raw_probability, "sampling_probability": sampling_probability,
                      "top_five_raw_candidates": candidates[:5]})
        if word == EOS:
            stop_reason = "eos"
            break
        generated.append(word)
        history.append(word)
    pairs = list(zip(generated, generated[1:]))
    repeated_fraction = (len(pairs) - len(set(pairs))) / len(pairs) if pairs else 0.0
    if any(w in (BOS, EOS, UNK) for w in generated):
        raise RuntimeError("正文中混入模型特殊符号")
    if len(generated) > MAX_NEW_TOKENS:
        raise RuntimeError("超过生成长度上限")
    return {"order": model.order, "temperature": temperature, "top_k": top_k, "seed": seed,
            "max_new_tokens": MAX_NEW_TOKENS, "new_token_count": len(generated),
            "stop_reason": stop_reason, "generated_tokens": generated,
            "continuation_spaced": " ".join(generated),
            "full_text_spaced": " ".join(PROMPT_TOKENS + generated),
            "full_text_without_spaces": "".join(PROMPT_TOKENS + generated),
            "has_local_loop": has_local_loop(generated), "short_continuation_under_5_tokens": len(generated) < 5,
            "repeated_bigram_fraction": repeated_fraction, "trace": trace}


def main():
    results = BASE / "results"
    training = json.loads((results / "training_metrics.json").read_text(encoding="utf-8"))
    expected_hashes = {row["order"]: row["model_sha256"] for row in training["runs"]}
    started = time.perf_counter()
    output = {"run_time_local": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "prompt": PROMPT, "prompt_tokens": PROMPT_TOKENS, "prompt_segmentation": "人工按语料现有词级风格切分；未训练新分词器",
              "settings": SETTINGS, "seed": SEED, "max_new_tokens": MAX_NEW_TOKENS,
              "decoding_rules": "先屏蔽BOS/UNK，保留EOS；按原始模型概率取top-k；T>0按p^(1/T)重新归一化采样；T=0贪心；同概率按词字典序排序",
              "loop_definition": "续写中某处长度1至6的词块连续重复至少3次",
              "repeat_fraction_definition": "续写中相邻词对的重复出现次数除以词对总数；不含提示与EOS",
              "models": [], "runs": []}
    for order in (2, 3, 4):
        path = BASE / "models" / f"{order}gram.pkl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_hashes[order]:
            raise RuntimeError(f"{order}-gram与训练记录哈希不一致")
        # 这里只加载本实验03脚本生成并核对哈希的模型文件。
        with path.open("rb") as handle:
            model = pickle.load(handle)
        oov = [word for word in PROMPT_TOKENS if model.vocab.lookup(word) == UNK]
        cache = {}
        for temperature, top_k in SETTINGS:
            result = generate(model, temperature, top_k, SEED, cache)
            # 相同参数及种子必须复现完全相同的词序列和终止原因。
            repeated = generate(model, temperature, top_k, SEED, cache)
            if result != repeated:
                raise RuntimeError("固定种子复现检查失败")
            output["runs"].append(result)
            print(f"{order}-gram T={temperature:g}, top-k={top_k}: {result['new_token_count']}词，{result['stop_reason']}，局部循环={result['has_local_loop']}", flush=True)
            print(result["full_text_spaced"], flush=True)
        output["models"].append({"order": order, "sha256": digest, "prompt_oov_words": oov,
                                 "scored_unique_contexts": len(cache)})
    output["elapsed_seconds_including_repeat_checks"] = round(time.perf_counter() - started, 4)
    output["verification"] = {"model_hashes_match_training": True, "fixed_seed_reproduction": True,
                              "no_special_tokens_in_generated_body": True, "within_length_limit": True}
    (results / "generation_metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = ["# 指定开头续写与参数对比", "", f"开头：{PROMPT}……", "", f"固定种子：{SEED}；每组最多新增{MAX_NEW_TOKENS}词；共12组。使用03阶段保存的NLTK Witten–Bell模型，不重训。", "", "## 参数含义", "", "温度0采用贪心。其他组先按模型概率取top-k，再按p^(1/T)归一化采样。降低温度使相对高概率词更集中；增大top-k允许更多候选词参加采样。屏蔽BOS与UNK，保留EOS作为段落终止信号。", "", "固定top-k=20可对比温度0.7和1.0；固定温度1.0可对比top-k=20和100。每组只有一个种子，不能据此断言某组普遍更好。", "", "模型训练时学习的是原始分布；屏蔽特殊符号、top-k和温度会改变实际采样分布。停止于EOS或长度上限，不在句号处额外截断。", "", "## 提示词的词表情况", ""]
    for model_info in output["models"]:
        report.append(f"- {model_info['order']}-gram：提示中映射为UNK的词：{model_info['prompt_oov_words']}。显示文本保留原提示；计分历史使用模型词表映射。")
    report += ["", "## 全部原始结果", "", "输出只去除/保留分词空格用于展示，没有润色或挑选替换。EOS不显示在正文中。", ""]
    for run in output["runs"]:
        report += [f"### {run['order']}-gram，温度{run['temperature']:g}，top-k={run['top_k']}", "", f"新增{run['new_token_count']}词；终止={run['stop_reason']}；局部循环={run['has_local_loop']}；重复词对占比={run['repeated_bigram_fraction']:.1%}。", "", "```text", run["full_text_spaced"], "```", ""]
    report += ["## 观察边界", "", "局部循环指续写中长度1至6的某个词块连续重复至少3次。重复词对占比是描述指标，不等价于人工质量评分。模型只看末尾n−1个词，可能很快偏离校园会议主题；具体问题需结合上面的真实输出分析。", "", "本阶段未测验证/测试困惑度。生成总耗时含全词表评分、缓存建立及复现检查，不能当作优化后的在线生成速度。", "", "## 复现", "", "```powershell", "python .\\04_generate_text.py", "```", "", "相同环境、模型、代码和种子可复现本轮输出。逐词上下文、候选词概率、选择概率、模型哈希和检查结果见generation_metrics.json。"]
    (results / "generation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"完成：{results / 'generation_report.md'}", flush=True)


if __name__ == "__main__":
    main()

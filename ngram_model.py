"""各阶段共用的模型设置。段落之间不连接词对。"""
from nltk.lm import Vocabulary, WittenBellInterpolated

BOS, EOS, UNK = "<s>", "</s>", "<UNK>"


def training_ngrams(paragraphs, order):
    """每个目标词同时统计1至order阶；只补一个段落结束符。"""
    for tokens in paragraphs:
        padded = [BOS] * (order - 1) + tokens + [EOS]
        yield (
            tuple(padded[i - size + 1:i + 1])
            for i in range(order - 1, len(padded))
            for size in range(1, order + 1)
        )


def make_model(order, words):
    # 词表只来自训练集；低频词映射到UNK，不读取验证/测试词频。
    vocabulary = Vocabulary(list(words) + [BOS, EOS], unk_cutoff=1, unk_label=UNK)
    return WittenBellInterpolated(order, vocabulary=vocabulary)

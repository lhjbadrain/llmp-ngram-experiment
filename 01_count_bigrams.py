"""示例：统计相邻词对。以下句子是人工示例，不是正式实验语料。"""

from collections import Counter

sentences = [
    "今天 天气 很 好 。",
    "今天 天气 很 冷 。",
    "今天 我们 上课 。",
]

bigram_counts = Counter()

for sentence in sentences:
    words = sentence.split()
    for i in range(len(words) - 1):
        pair = (words[i], words[i + 1])
        bigram_counts[pair] += 1

print("词对及出现次数：")
for pair, count in bigram_counts.items():
    print(pair, count)
# 收集“今天”后面各个词的出现次数
next_counts = Counter()
for (previous, following), count in bigram_counts.items():
    if previous == "今天":
        next_counts[following] += count

# 用次数除以总次数，得到条件概率
total = sum(next_counts.values())
for word, count in next_counts.items():
    print(f"P({word} | 今天) = {count}/{total} = {count / total:.2%}")


input("\n按回车键退出……")

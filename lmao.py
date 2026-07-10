with open("data/raw.txt") as f:
    words = f.read().split()

with open("data/tiny.txt", "w") as f:
    f.write(" ".join(words[:10_000]))

with open("data/small.txt", "w") as f:
    f.write(" ".join(words[:100_000]))

with open("data/medium.txt", "w") as f:
    f.write(" ".join(words[:1_000_000]))
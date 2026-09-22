"""Точная дедупликация и детерминированный поиск по Жаккару словных шинглов.

Инвертированный индекс выдаёт все пары с общим шинглом. При положительном
пороге ни одна подходящая пара не теряется; ложные кандидаты отсеиваются
точным Жаккаром. Для нескольких тысяч строк это практичнее вероятностного LSH.
"""
from collections import Counter, defaultdict
from typing import Sequence
from datasketch import MinHash
from src.textnorm import shingles


def build_minhash(text: str, shingle_words: int, num_perm: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    mh.update_batch([s.encode('utf-8') for s in sorted(shingles(text, shingle_words))])
    return mh


def exact_duplicates(keys: Sequence[str]) -> list[int]:
    seen, duplicates = set(), []
    for i, key in enumerate(keys):
        if key in seen:
            duplicates.append(i)
        seen.add(key)
    return duplicates


class ShingleIndex:
    def __init__(self, threshold):
        if not 0 < threshold <= 1:
            raise ValueError('threshold должен быть в (0, 1]')
        self.threshold = threshold
        self.postings = defaultdict(list)
        self.sizes = {}
        self.empty = []

    def add(self, i, tokens):
        self.sizes[i] = len(tokens)
        if not tokens:
            self.empty.append(i)
        for token in sorted(tokens):
            self.postings[token].append(i)

    def query(self, tokens):
        if not tokens:
            return list(self.empty)
        counts = Counter(i for token in tokens for i in self.postings.get(token, ()))
        return sorted(i for i, intersection in counts.items()
                      if intersection / (len(tokens) + self.sizes[i] - intersection) >= self.threshold)


def near_duplicates(texts: Sequence[str], shingle_words: int, num_perm: int, threshold: float) -> list[int]:
    # num_perm оставлен для совместимости с интерфейсом заготовки; поиск точный.
    index, duplicates = ShingleIndex(threshold), []
    for i, text in enumerate(texts):
        tokens = shingles(text, shingle_words)
        if index.query(tokens):
            duplicates.append(i)
        else:
            index.add(i, tokens)
    return duplicates


def cross_near_duplicates(left: Sequence[str], right: Sequence[str], shingle_words: int,
                          num_perm: int, threshold: float) -> list[tuple[int,int]]:
    index = ShingleIndex(threshold)
    for i, text in enumerate(left):
        index.add(i, shingles(text, shingle_words))
    return [(i,j) for j,text in enumerate(right) for i in index.query(shingles(text, shingle_words))]

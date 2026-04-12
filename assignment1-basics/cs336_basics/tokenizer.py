from typing import Iterable

import regex


class Tokenizer:
    """My BPE tokenizer used to encode and decode text"""

    def __init__(self, vocab, merges, special_tokens=None) -> None:
        """
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        Your tokenizer should also support user-provided
        special tokens (appending them to the vocabulary if they aren’t already there)
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or ["<|endoftext|>"]

        self.id_to_bytes = vocab
        self.bytes_to_id = {v: k for k, v in vocab.items()}

        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            if special_token_bytes in self.bytes_to_id:
                # 如果这个 special token 在词汇表中
                continue
            # 如果这个 special token 不在词汇表中
            # update vocab
            self.bytes_to_id[special_token_bytes] = len(self.vocab)
            self.vocab[len(self.vocab)] = special_token_bytes
        self.id_to_bytes = self.vocab

        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # noqa: E501
        self.merge_rank = {
            (self.bytes_to_id[pair[0]], self.bytes_to_id[pair[1]]): i
            for i, pair in enumerate(merges)
        }

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        import json
        from pathlib import Path

        vocab_path = Path(vocab_filepath)
        raw_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        vocab: dict[int, bytes] = {
            int(k): bytes.fromhex(v) for k, v in raw_vocab.items()
        }
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                a_hex, b_hex = line.strip().split()
                merges.append((bytes.fromhex(a_hex), bytes.fromhex(b_hex)))
        return cls(vocab, merges, special_tokens or ["<|endoftext|>"])

    def single_merge(self, ids: list[int]):
        while True:
            tmp = []
            # min(pairs, key=lambda p: self.merge_rank[p])
            # 会因为 pair 不在 merge_rank 中报错
            # pair = zip(ids[:-1], ids[1:])
            # to_merge = min(pairs, key=lambda p: self.merge_rank[p])
            candidate_pairs = [
                (i, (ids[i], ids[i + 1]))
                for i in range(len(ids) - 1)
                if (ids[i], ids[i + 1]) in self.merge_rank
            ]
            if not candidate_pairs:
                break
            to_merge = min(candidate_pairs, key=lambda p: self.merge_rank[p[1]])
            _, (id_a, id_b) = to_merge
            to_merge_id = self.bytes_to_id[self.vocab[id_a] + self.vocab[id_b]]
            i = 0
            while i < len(ids) - 1:
                if (ids[i], ids[i + 1]) == (id_a, id_b):
                    tmp.append(to_merge_id)
                    i += 2
                else:
                    tmp.append(ids[i])
                    i += 1
            if i == len(ids) - 1:
                tmp.append(ids[i])
            ids = tmp
        return ids

    def encode(self, text: str) -> list[int]:
        result = []
        # 首先需要使用 pattern 进行分词
        if self.special_tokens:
            # 正则交替匹配（|）是从左到右短路的。如果 special_tokens 里
            # 同时有 "<|end|>" 和 "<|endoftext|>"，且前者排在前面，
            # 那么 "<|endoftext|>" 永远不会被匹配到。
            # 所以需要排序一下
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(regex.escape(tok) for tok in sorted_special)
            segments = regex.split(f"({special_pattern})", text)
        else:
            segments = [text]

        for segment in segments:
            if segment in self.special_tokens:
                special_id = self.bytes_to_id[segment.encode("utf-8")]
                result.append(special_id)
                continue
            for match in regex.finditer(self.pattern, segment):
                # 由于我们的 tokenizer 将 0～255 作为初始词汇表
                # 因此原字节序列 bytes，可以直接转变为 list[int]
                # 后续在此基础上进行 merge
                # !!!!
                # 这是不行的，直接映射到词汇表
                # 因为有一些词汇表没有这么映射，我们还是直接以词汇表为准
                token = [
                    self.bytes_to_id[bytes([b])] for b in match.group().encode("utf-8")
                ]
                result.extend(self.single_merge(token))
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for part in iterable:
            yield from self.encode(part)

    def decode(self, ids: list[int]) -> str:
        # decode
        result = b""
        for id in ids:
            result += self.vocab[id]
        return result.decode("utf-8", errors="replace")

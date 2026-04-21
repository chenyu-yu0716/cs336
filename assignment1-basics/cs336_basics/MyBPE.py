import heapq
import os
from collections import defaultdict
from dataclasses import dataclass, field

GPT2_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


# Step 1: Pre-tokenization
def pre_tokenization(input_path, special_tokens):
    """
    * 输入：输入文本 -- `str`
    * 输出：`Dict[Tuple[int, ...], int]`，即字节序列 → 频率。
    * 要求：高效并行 + 处理好 special tokens
    """
    import multiprocessing as mp

    workers = mp.cpu_count()
    # multiprocessing
    from cs336_basics.train_bpe import find_chunk_boundaries, process_chunks

    with open(input_path, "rb") as f:
        if special_tokens:
            boundaries = find_chunk_boundaries(
                f, workers, special_tokens[0].encode("utf-8")
            )
        else:
            f.seek(0, 2)
            file_size = f.tell()
            chunk_size = file_size // workers
            boundaries = [i * chunk_size for i in range(workers + 1)]
            boundaries[-1] = file_size

    chunk_args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk_args.append((start, end, input_path, special_tokens))

    with mp.Pool(workers) as pool:
        chunk_results = pool.map(process_chunks, chunk_args)
    pretokens = {}
    for chunk_result in chunk_results:
        for pretoken, freq in chunk_result.items():
            pretokens[pretoken] = pretokens.get(pretoken, 0) + freq

    return pretokens


# Step 2: Define Pretoken-seq DataStructure
@dataclass
class Node:
    val: int  # 这里的 val 指的是该 Node 节点对应词汇表的 ID
    prev: "Node | None" = field(default=None, repr=False)
    next: "Node | None" = field(default=None, repr=False)


class PretokenSeq:
    def __init__(self, head: Node | None, freq: int, id: int) -> None:
        self.head = head
        self.freq = freq
        self.pretoken_id = id

    def __repr__(self):
        vals = []
        cur = self.head
        while cur:
            vals.append(str(cur.val))
            cur = cur.next
        return (
            f"PretokenSeq(id={self.pretoken_id}, freq={self.freq}, "
            f"seq=[{', '.join(vals)}])"
        )


def build_pretoken_seq(pretoken_freq: dict[tuple[int, ...], int]) -> list[PretokenSeq]:
    pretokens: list[PretokenSeq] = []
    ids = 0
    for seq, freq in pretoken_freq.items():
        # seq: tuple[int, ...]
        # freq: int -- frequency
        head = None
        previous: Node | None = None
        for val in seq:
            new_node = Node(val=val)
            if not previous:
                head = new_node
                previous = head
            else:
                previous.next = new_node
                new_node.prev = previous
            previous = new_node
        pretokens.append(PretokenSeq(head=head, freq=freq, id=ids))
        ids += 1
    return pretokens


# Step 3: Pair 频率 & 倒排索引 & 堆
def build_pair_freq_and_index(pretokens: list[PretokenSeq]):
    pair_freq: dict[tuple[int, int], int] = {}
    pair_index: dict[tuple[int, int], set[int]] = {}
    for pretoken in pretokens:
        cur = pretoken.head
        while cur is not None and cur.next is not None:
            assert cur.val is not None and cur.next.val is not None
            pair: tuple[int, int] = (cur.val, cur.next.val)
            pair_freq[pair] = pair_freq.get(pair, 0) + pretoken.freq
            if pair not in pair_index:
                pair_index[pair] = set()
            pair_index[pair].add(pretoken.pretoken_id)
            cur = cur.next
    return (pair_freq, pair_index)


class NegBytes:
    __slots__ = ("b",)

    def __init__(self, b: bytes):
        self.b = b

    def __lt__(self, other):
        return self.b > other.b

    def __eq__(self, other):
        return self.b == other.b

    def __le__(self, other):
        return self.b >= other.b


def pair_freq_heapify(
    pair_freq: dict[tuple[int, int], int], vocab: dict[int, bytes]
) -> list[tuple[int, NegBytes, NegBytes, tuple[int, int]]]:
    max_heap = []
    for pair, freq in pair_freq.items():
        a_bytes, b_bytes = vocab[pair[0]], vocab[pair[1]]
        heapq.heappush(max_heap, (-freq, NegBytes(a_bytes), NegBytes(b_bytes), pair))
    return max_heap


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
):
    """
    输入:
        1. input_path -- text 文本
        2. 期望的词汇表大小
        3. special tokens
    输出:
        1. dict[int, bytes] -- vocabulary
        2. list[tuple[bytes, bytes] -- merges
    """
    pretoken_freq = pre_tokenization(
        input_path, special_tokens
    )  # pretoken_freq: dict[tuple[int,...], int]

    pretokens = build_pretoken_seq(pretoken_freq)  # pretokens: list[PretokenSeq]
    pair_freq, pair_index = build_pair_freq_and_index(pretokens)

    # Addition: Initialize The Original Vocabulary and Merge
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")
    base_vocab = 256 + len(special_tokens)
    nums_merge = vocab_size - base_vocab
    merges: list[tuple[bytes, bytes]] = []

    # 现在需要建 大根堆 -- 懒删除大根堆
    max_heap = pair_freq_heapify(pair_freq, vocab)
    # max_heap 中的 element: (-freq, NegBytes(a_bytes), NegBytes(b_bytes), pair)

    # Step 4: Merge Loop
    for i in range(nums_merge):
        # 首先需要找出出现频率最高的 pair
        while max_heap:
            # 懒删除需要确定验证条件，验证条件采用 pair_freq[pair] != freq
            negFreq, _, _, pair = max_heap[0]
            if pair not in pair_freq or -negFreq != pair_freq[pair]:
                heapq.heappop(max_heap)
                continue
            else:
                best_pair = pair
                break

        if not max_heap:
            break

        A, B = vocab[best_pair[0]], vocab[best_pair[1]]

        # 找到 best_pair 了，现在需要 merge
        # 1. 更新 vocabulary & merges
        new_id = base_vocab + i
        vocab[new_id] = A + B
        merges.append((A, B))

        # 2. 更新 pretokens: list[PretokenSeq]
        # 必须拷贝：当 best_pair=(X,X) 时，inner loop 中的 discard 会修改原 set
        affected_seq = list(pair_index.get(best_pair, set()))
        old_pairs = defaultdict(int)
        new_pairs = defaultdict(int)
        # 统计 merge 后受影响的 pair，affected_seq 全部更新完后统一更新pair数据和堆
        for id in affected_seq:
            # pretoken = pretokens[id]
            # 只需要更新 best_pair 前后的 pair
            freq = pretokens[id].freq
            cur = pretokens[id].head
            to_update_index = set()
            # 遍历一遍链表
            while cur is not None and cur.next is not None:
                # 找到 best_pair
                if cur.val == best_pair[0] and cur.next.val == best_pair[1]:
                    if cur.prev:
                        prev_pair = (cur.prev.val, best_pair[0])
                        to_update_index.add(prev_pair)
                        old_pairs[prev_pair] += freq
                        new_left = (cur.prev.val, new_id)
                        new_pairs[new_left] += freq
                        pair_index.setdefault(new_left, set()).add(id)
                    if cur.next.next:
                        next_pair = (best_pair[1], cur.next.next.val)
                        to_update_index.add(next_pair)
                        old_pairs[next_pair] += freq
                        new_right = (new_id, cur.next.next.val)
                        new_pairs[new_right] += freq
                        pair_index.setdefault(new_right, set()).add(id)
                    tmp = cur.next
                    cur.next = tmp.next
                    if cur.next:
                        cur.next.prev = cur  # 维护双向链表
                    del tmp
                    cur.val = new_id
                cur = cur.next
            cur = pretokens[id].head
            while cur is not None and cur.next is not None:
                to_update_index.discard((cur.val, cur.next.val))
                cur = cur.next
            for to_update in to_update_index:
                pair_index[to_update].discard(id)

        # best_pair 已完全合并，从 pair_freq 和 pair_index 中删除
        pair_freq.pop(best_pair, None)
        pair_index.pop(best_pair, None)

        # 合并 old_pairs 和 new_pairs 为 delta，统一更新 pair_freq 和 max_heap
        # 必须合并处理：当同一个 pair 同时出现在 old_pairs 和 new_pairs（如连续 best_pair
        # 情况 A B A B，中间产生的 transient pair），分开处理会导致 KeyError
        delta: dict[tuple[int, int], int] = {}
        for p, f in old_pairs.items():
            delta[p] = delta.get(p, 0) - f
        for p, f in new_pairs.items():
            delta[p] = delta.get(p, 0) + f

        for pair, d in delta.items():
            new_freq = pair_freq.get(pair, 0) + d
            if new_freq <= 0:
                pair_freq.pop(pair, None)
            else:
                pair_freq[pair] = new_freq
                heapq.heappush(
                    max_heap,
                    (
                        -new_freq,
                        NegBytes(vocab[pair[0]]),
                        NegBytes(vocab[pair[1]]),
                        pair,
                    ),
                )

    return (vocab, merges)


def main():
    input_path = "../data/TinyStories/TinyStories-valid.txt"
    # pretoken_freq = pre_tokenization(input_path, ["<|endoftext|>"])
    # pretokens = build_pretoken_seq(pretoken_freq)
    # print(f"pretoken_freq: {pretoken_freq}")
    # print(f"pretokens: {pretokens[:10]}")
    vocab, merge = train_bpe(input_path, 1500, ["<|endoftext|>"])
    print(vocab)
    print(merge[:10])


if __name__ == "__main__":
    main()

import os
from collections import defaultdict
from typing import BinaryIO

import regex

# TODO:
# 后续可以继续更新一下
# 将 training 中的关键部分使用 rust 编写
# 进一步提升效率


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def count_pretoken(text: str, special_tokens: list[str]) -> dict[bytes, int]:
    GPT2_PATTERN = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    freq = defaultdict(int)

    if special_tokens:
        special_pattern = (
            "(" + "|".join(regex.escape(tok) for tok in special_tokens) + ")"
        )
        parts = regex.split(special_pattern, text)
    else:
        parts = [text]

    for part in parts:
        if part in special_tokens:
            continue
        for match in regex.finditer(GPT2_PATTERN, part):
            token = match.group().encode("utf-8")
            freq[token] = freq.get(token, 0) + 1
    return freq


def train_bpe(  # noqa: C901
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 首先我需要从 input_path 中获取 text
    # input_path 是一个 str
    chunks = []
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(
            f, num_processes, split_special_token="<|endoftext|>".encode("utf-8")
        )
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
    # now run pre-tokenization on my chunk and store the counts for each pre-token
    # chunks的类型是 list[str]
    all_chunk_freq: dict[bytes, int] = defaultdict(int)
    # 我应该统计所有 chunk 中的 freq_token

    for chunk in chunks:
        chunk_freq = count_pretoken(chunk, special_tokens=special_tokens)
        for token, freq in chunk_freq.items():
            all_chunk_freq[token] += freq
    # all_chunk_freq 是所有 pre-token 的字典
    # pre-token 是指分词后状态比如 "hello world" 分词后得到 'hello' ' world'

    # 之后我需要将所有 pre-token 拆成字节序列(初始状态)
    # 然后统计所有相邻字节对的频率
    # 找出频率最高的 pair，merge 它
    # 重复直到达到目标 vocab_size

    # 初始化：把每个 pre-token 展开成 token id 列表
    # key: tuple[int,...] (token id 序列), value: 出现次数
    token_seqs: dict[tuple[int, ...], int] = {}
    for pre_token_bytes, freq in all_chunk_freq.items():
        seq = tuple(pre_token_bytes)
        token_seqs[seq] = token_seqs.get(seq, 0) + freq

    # 统计初始 pair 频率
    pair_freq: dict[tuple[int, int], int] = defaultdict(int)
    for seq, freq in token_seqs.items():
        for i in range(len(seq) - 1):
            pair_freq[(seq[i], seq[i + 1])] = (
                pair_freq.get((seq[i], seq[i + 1]), 0) + freq
            )
    # 同时我们也要建立反向的索引，对某一个 pair，哪些初始序列包含他 token_seqs
    pair_to_seqs: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
    for seq in token_seqs:
        for i in range(len(seq) - 1):
            pair_to_seqs[(seq[i], seq[i + 1])].add(seq)

    # 我们已经有 vocab size了
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    base_vocab = 256 + len(special_tokens)
    nums_merge = vocab_size - base_vocab
    merges: list[tuple[bytes, bytes]] = []

    for i in range(nums_merge):
        if not pair_freq:
            break

        best_pair = max(
            pair_freq,
            key=lambda p: (pair_freq[p], (vocab[p[0]], vocab[p[1]])),
        )
        # 后续应该加入当 best_pair 出现的频次，小于多少的时候应该停止 iterate
        new_id = base_vocab + i
        A, B = best_pair

        # TODO:
        # 更新 pair_freq -- 删除 best_pair 以及添加新的 pair
        # 接下来我们只遍历包含 best_pair 的序列
        affected_seqs = list(pair_to_seqs.pop(best_pair, set()))
        # 我们 pop 了原来 best_pair 的部分，后续需要再在 pair_to_seqs 加上
        # 合并后的新 id 和前后组成的 新 pair 的反向索引
        for old_seq in affected_seqs:
            freq = token_seqs.pop(old_seq)
            new_seq_list: list = []
            # 构建新的 seq
            j = 0
            while j < len(old_seq):
                if j < len(old_seq) - 1 and old_seq[j] == A and old_seq[j + 1] == B:
                    new_seq_list.append(new_id)
                    j += 2
                else:
                    new_seq_list.append(old_seq[j])
                    j += 1
            new_seq = tuple(new_seq_list)

            # 删除 old_seq 所有 pair 的计数和索引
            for k in range(len(old_seq) - 1):
                # 这里可以优化成只删除 best_pair 相邻的 pair 的计数和索引
                # 但是反正都需要遍历一边寻找 best_pair 就直接这么写
                # 并且代码统一也方便看，后续有空可以进一步优化
                pair = (old_seq[k], old_seq[k + 1])
                pair_freq[pair] -= freq
                if pair_freq[pair] <= 0:
                    del pair_freq[pair]
                pair_to_seqs[pair].discard(old_seq)

            # 更新 new_seq 所有 pair 的计数和索引
            token_seqs[new_seq] = token_seqs.get(new_seq, 0) + freq
            for k in range(len(new_seq) - 1):
                pair = (new_seq[k], new_seq[k + 1])
                pair_freq[pair] = pair_freq.get(pair, 0) + freq
                pair_to_seqs[pair].add(new_seq)
        merges.append((vocab[A], vocab[B]))
        vocab[new_id] = vocab[A] + vocab[B]

    return (vocab, merges)


# 接下来我们写一个 parallel 版本的
GPT2_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def process_chunks(args):
    start, end, input_path, special_tokens = args
    chunk_freq = defaultdict(int)

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk: str = f.read(end - start).decode("utf-8", errors="ignore")

        if special_tokens:
            pattern = "|".join(regex.escape(tok) for tok in special_tokens)
            segments = regex.split(f"({pattern})", chunk)
        else:
            segments = [chunk]

        for segment in segments:
            if segment in special_tokens:
                continue
            for match in regex.finditer(GPT2_PATTERN, segment):
                # 这里直接拆成 token-id 优化一下
                token = tuple(match.group().encode("utf-8"))
                chunk_freq[token] = chunk_freq.get(token, 0) + 1
    return chunk_freq


def parallel_train_bpe(  # noqa: C901
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 导入多线程库
    import multiprocessing as mp

    # 获取 cpu cores
    num_processes = mp.cpu_count()

    # 之后需要寻找 boundaries 以及构造 args
    with open(input_path, "rb") as f:
        if special_tokens:
            boundaries = find_chunk_boundaries(
                f, num_processes, special_tokens[0].encode("utf-8")
            )
        else:
            f.seek(0, 2)
            file_size = f.tell()
            chunk_size = file_size // num_processes
            boundaries = [i * chunk_size for i in range(num_processes + 1)]
            # 需要处理的是，boundaries 最后的一个一定就是 file_size，
            # 因为 i * chunk_size 很有可能会超过 file_size 导致出错
            boundaries[-1] = file_size
    # 找好所有的 boundaries 我们就可以调用所有的 cpu cores 来并行处理 pre-token
    # start, end, input_path, special_tokens = args
    chunk_args = []
    for start, end in zip(boundaries, boundaries[1:]):
        chunk_args.append((start, end, input_path, special_tokens))

    # 如此一来就构造好了所有的 args 我们可以开始并行处理了
    with mp.Pool(processes=num_processes) as pool:
        chunk_results = pool.map(process_chunks, chunk_args)

    # 处理完后，我们合并所有的结果

    # pre-token 是指分词后状态比如 "hello world" 分词后得到 'hello' ' world'
    # 初始化：把每个 pre-token 展开成 token id 列表
    # 在 process_chunks 我们已经这么做了
    # key: tuple[int,...] (token id 序列), value: 出现次数
    token_seqs: dict[tuple[int, ...], int] = {}
    for chunk_result in chunk_results:
        for token_seq, freq in chunk_result.items():
            token_seqs[token_seq] = token_seqs.get(token_seq, 0) + freq

    # 之后我需要将所有 pre-token 拆成字节序列(初始状态)
    # 然后统计所有相邻字节对的频率
    # 找出频率最高的 pair，merge 它
    # 重复直到达到目标 vocab_size

    # 统计初始 pair 频率
    pair_freq: dict[tuple[int, int], int] = defaultdict(int)
    for seq, freq in token_seqs.items():
        for i in range(len(seq) - 1):
            pair_freq[(seq[i], seq[i + 1])] = (
                pair_freq.get((seq[i], seq[i + 1]), 0) + freq
            )
    # 同时我们也要建立反向的索引，对某一个 pair，哪些初始序列包含他 token_seqs
    pair_to_seqs: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
    for seq in token_seqs:
        for i in range(len(seq) - 1):
            pair_to_seqs[(seq[i], seq[i + 1])].add(seq)

    # 我们已经有 vocab size了
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    base_vocab = 256 + len(special_tokens)
    nums_merge = vocab_size - base_vocab
    merges: list[tuple[bytes, bytes]] = []

    for i in range(nums_merge):
        if not pair_freq:
            break

        best_pair = max(
            pair_freq,
            key=lambda p: (pair_freq[p], (vocab[p[0]], vocab[p[1]])),
        )
        # 后续应该加入当 best_pair 出现的频次，小于多少的时候应该停止 iterate
        new_id = base_vocab + i
        A, B = best_pair

        # TODO:
        # 更新 pair_freq -- 删除 best_pair 以及添加新的 pair
        # 接下来我们只遍历包含 best_pair 的序列
        affected_seqs = list(pair_to_seqs.pop(best_pair, set()))
        # 我们 pop 了原来 best_pair 的部分，后续需要再在 pair_to_seqs 加上
        # 合并后的新 id 和前后组成的 新 pair 的反向索引
        for old_seq in affected_seqs:
            freq = token_seqs.pop(old_seq)
            new_seq_list: list = []
            # 构建新的 seq
            j = 0
            while j < len(old_seq):
                if j < len(old_seq) - 1 and old_seq[j] == A and old_seq[j + 1] == B:
                    new_seq_list.append(new_id)
                    j += 2
                else:
                    new_seq_list.append(old_seq[j])
                    j += 1
            new_seq = tuple(new_seq_list)

            # 删除 old_seq 所有 pair 的计数和索引
            for k in range(len(old_seq) - 1):
                # 这里可以优化成只删除 best_pair 相邻的 pair 的计数和索引
                # 但是反正都需要遍历一边寻找 best_pair 就直接这么写
                # 并且代码统一也方便看，后续有空可以进一步优化
                pair = (old_seq[k], old_seq[k + 1])
                pair_freq[pair] -= freq
                if pair_freq[pair] <= 0:
                    del pair_freq[pair]
                pair_to_seqs[pair].discard(old_seq)

            # 更新 new_seq 所有 pair 的计数和索引
            token_seqs[new_seq] = token_seqs.get(new_seq, 0) + freq
            for k in range(len(new_seq) - 1):
                pair = (new_seq[k], new_seq[k + 1])
                pair_freq[pair] = pair_freq.get(pair, 0) + freq
                pair_to_seqs[pair].add(new_seq)
        merges.append((vocab[A], vocab[B]))
        vocab[new_id] = vocab[A] + vocab[B]

    return (vocab, merges)


def save_vocab_and_merges(dir_path, vocab, merges):
    import json
    from pathlib import Path

    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    vocab_data = {str(k): v.hex() for k, v in vocab.items()}
    (dir_path / "vocab.json").write_text(
        json.dumps(vocab_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (dir_path / "merges.txt").open("w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")


def main():
    import time

    input_path = "./data/TinyStories/TinyStories-train.txt"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 1500
    start_time = time.time()
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    costs = time.time() - start_time
    save_vocab_and_merges("./results/non-parallel", vocab, merges)
    print(f"Train BPE on TinyStories costs {costs:.2f} seconds ON NON-PARALLEL")

    start_time = time.time()
    vocab, merges = parallel_train_bpe(input_path, vocab_size, special_tokens)
    costs = time.time() - start_time
    save_vocab_and_merges("./results/parallel", vocab, merges)
    print(f"Train BPE on TinyStories costs {costs:.2f} seconds ON PARALLEL")


if __name__ == "__main__":
    main()

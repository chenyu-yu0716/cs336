# MyBPE.py Debug 记录

本文档记录了 `train_bpe` 函数中发现并修复的 4 个 bug，每个 bug 均附有修改前后的代码、错误原因及修复思路。

---

## 数据结构背景

理解这些 bug 需要先了解实现中使用的几个核心数据结构：

- **`pretokens: list[PretokenSeq]`**：每个 pre-token（如 `" hello"`）表示为一个双向链表，每个节点的 `val` 是词汇表 ID。
- **`pair_freq: dict[(int, int), int]`**：记录每个相邻 pair 在整个语料库中的出现频率。
- **`pair_index: dict[(int, int), set[int]]`**：倒排索引，记录哪些 pretoken 序列（用 `pretoken_id` 表示）包含某个 pair。
- **`max_heap`**：基于 `pair_freq` 构建的懒删除最大堆，用于高效找到频率最高的 pair。

每次 merge 循环：
1. 从堆中弹出频率最高的 `best_pair = (A, B)`
2. 在所有包含该 pair 的链表中，将相邻的 A、B 节点合并为新节点 `new_id`
3. 更新 `pair_freq`、`pair_index`、堆

---

## Bug 1：将 best_pair 误加回 new_pairs

### 出错位置

merge 循环内，遍历链表找到 best_pair 匹配时。

### 修改前

```python
while cur is not None and cur.next is not None:
    if cur.val == best_pair[0] and cur.next.val == best_pair[1]:
        new_pairs[best_pair] += freq        # ← BUG
        pair_index[best_pair].add(id)       # ← BUG
        if cur.prev:
            ...
```

### 修改后

```python
while cur is not None and cur.next is not None:
    if cur.val == best_pair[0] and cur.next.val == best_pair[1]:
        # 不再有 new_pairs[best_pair] 和 pair_index[best_pair].add(id)
        if cur.prev:
            ...
```

### 为什么出错

`best_pair = (A, B)` 被 merge 后，语料库中所有的 `A B` 都被替换成了 `new_id`。也就是说，**pair `(A, B)` 应当在这次 merge 后彻底消失**，不应该再出现在 `pair_freq` 中。

但原代码在每次找到一个 `(A, B)` 匹配时都执行：

```python
new_pairs[best_pair] += freq
```

随后在循环末尾，`new_pairs` 中的所有 pair 都会被写回 `pair_freq`：

```python
for new_pair, freq in new_pairs.items():
    pair_freq[new_pair] = pair_freq.get(new_pair, 0) + freq
    heapq.heappush(max_heap, ...)
```

这导致 `(A, B)` 被重新加入了 `pair_freq`，并且频率不变。在下一次 merge 循环中，堆顶仍然是 `(A, B)`，堆的懒删除检查 `pair_freq[pair] == -negFreq` 依然通过，于是同一个 pair 被再次 merge，产生完全错误的结果（第二次 merge 会输出错误的 merges 顺序）。

`pair_index[best_pair].add(id)` 同理：best_pair 已经不存在于链表中了，还把 pretoken id 加回去，会让下次误以为该 pretoken 仍含有 best_pair。

### 修复思路

直接删除这两行。匹配到 best_pair 时只需要处理其**邻居** pair（左邻居和右邻居），best_pair 本身消失，无需任何"新增"处理。

---

## Bug 2：双向链表的 prev 指针未更新

### 出错位置

找到 best_pair 匹配后，执行链表节点合并时。

### 修改前

```python
tmp = cur.next          # tmp 指向 B 节点
cur.next = tmp.next     # 绕过 B，将 cur.next 指向 B 之后的节点
del tmp                 # 删除 B 节点
cur.val = new_id        # 将 A 节点的值改为 new_id
# 此时 cur.next（原 B 的后继节点）的 prev 指针仍然指向已删除的 B 节点！
```

### 修改后

```python
tmp = cur.next
cur.next = tmp.next
if cur.next:
    cur.next.prev = cur  # 维护双向链表：将后继节点的 prev 指向当前节点
del tmp
cur.val = new_id
```

### 为什么出错

合并 A、B 为 `new_id` 后，链表结构变化如下：

```
修改前：... <-> prev <-> [A] <-> [B] <-> next_node <-> ...
                          ↑ cur                ↑ cur.next.next

合并后期望：... <-> prev <-> [new_id] <-> next_node <-> ...
                              ↑ cur          ↑ cur.next

实际（有 bug）：
  cur.next = next_node  ✓（next 指针正确）
  next_node.prev = B（已删除！）  ✗（prev 指针仍然指向被删除的 B）
```

这个 bug 在**同一个 pretoken 中有连续的 best_pair** 时触发，例如：

```
序列：A B A B
      ↑1 ↑2 ↑3 ↑4（节点编号）
```

**第一次匹配**（节点1=A，节点2=B）：
- 合并：节点1.val = new_id，节点1.next = 节点3（原来的第二个 A）
- **但节点3.prev 仍然指向已删除的节点2（B）！**
- `cur = cur.next` → cur 移动到节点3

**第二次匹配**（节点3=A，节点4=B）：
- `cur.prev` 是节点3.prev，由于未更新，它仍然指向已删除的节点2
- `prev_pair = (cur.prev.val, best_pair[0]) = (B, A)` —— **错误！**
- 应该是 `(new_id, A)`，因为节点3的左邻是节点1（已被改为 new_id）

结果：
- `old_pairs[(B, A)]` 被错误地递增（B 节点已不存在于链表中）
- `new_pairs[(B, new_id)]` 被错误地创建（应为 `(new_id, new_id)`）
- `pair_freq` 中 `(B, A)` 被错误地减少，`(new_id, new_id)` 永远不会被创建

### 修复思路

在删除 B 节点后，立即将后继节点的 `prev` 指针更新为当前节点 `cur`。这样无论 best_pair 在序列中出现多少次，每次 merge 后链表的双向指针都保持正确。

---

## Bug 3：best_pair 对应的 pair_freq 未清除

### 出错位置

所有 affected_seq 处理完毕后，更新全局数据结构时。

### 修改前

```python
# 处理完所有受影响的 pretoken 后，直接进入 pair_freq 更新
for old_pair, freq in old_pairs.items():
    pair_freq[old_pair] -= freq
    ...

for new_pair, freq in new_pairs.items():
    pair_freq[new_pair] = pair_freq.get(new_pair, 0) + freq
    heapq.heappush(max_heap, ...)
# pair_freq[best_pair] 从未被删除！
```

### 修改后

```python
# best_pair 已完全合并，从 pair_freq 和 pair_index 中删除
pair_freq.pop(best_pair, None)
pair_index.pop(best_pair, None)

# 然后再更新 old_pairs / new_pairs
...
```

### 为什么出错

在修复 Bug 1（删除 `new_pairs[best_pair]`）之后，`best_pair` 不再会被重新加入 `new_pairs`。但 `old_pairs` 中也**从来不包含 best_pair 本身**（只包含它的左右邻居 pair），所以 `pair_freq[best_pair]` 在整个循环中**始终没有被修改**，仍然保持着 merge 前的旧频率值。

在下一次 merge 循环中，懒删除堆弹出 best_pair 时，检查：
```python
pair not in pair_freq  # False，因为 best_pair 仍在 pair_freq 里
-negFreq != pair_freq[pair]  # False，频率值也没变
```
两个条件都不满足，于是 best_pair 再次被选中为 best_pair，发生重复 merge。

在 Bug 1 存在时，这个问题被"掩盖"了：`new_pairs[best_pair]` 会把 best_pair 以旧频率重新加入堆，下次虽然也会选中它，但懒删除堆的行为恰好使得这个错误不那么明显（只是产生错误的 merge 顺序）。修复 Bug 1 后，这个 bug 的效果更加直接：pair_freq[best_pair] 未清除 → best_pair 被反复选中 → merge 循环产生大量重复/错误 merge。

### 修复思路

在处理完所有 affected_seq 之后，在更新 old_pairs/new_pairs 之前，显式地将 best_pair 从 `pair_freq` 和 `pair_index` 中删除：

```python
pair_freq.pop(best_pair, None)
pair_index.pop(best_pair, None)
```

---

## Bug 4：在遍历集合时修改集合导致 RuntimeError

### 出错位置

遍历 `affected_seq = pair_index[best_pair]` 时。

### 修改前

```python
affected_seq = pair_index[best_pair]  # 直接引用原集合
...
for id in affected_seq:               # 遍历时，集合可能被修改
    ...
    for to_update in to_update_index:
        pair_index[to_update].discard(id)  # 可能修改 pair_index[best_pair]！
```

### 修改后

```python
# 必须拷贝：当 best_pair=(X,X) 时，inner loop 中的 discard 会修改原 set
affected_seq = list(pair_index.get(best_pair, set()))  # 拷贝一份
...
for id in affected_seq:
    ...
    for to_update in to_update_index:
        pair_index[to_update].discard(id)  # 即使修改了原集合也无妨
```

### 为什么出错

`to_update_index` 中存放的是 best_pair 在链表中各个匹配位置的**邻居 pair**：

```python
prev_pair = (cur.prev.val, best_pair[0])
next_pair = (best_pair[1], cur.next.next.val)
```

通常情况下，这些邻居 pair 与 best_pair 不同，不会影响 `pair_index[best_pair]` 的内容。

但是，**当 `best_pair = (X, X)`（两个相同 token 的 pair）时**，存在一种特殊情况：

考虑序列 `... X X X ...`（三个相同 token 相邻），best_pair = (X, X)：

1. 第一次匹配（位置0-1的 X X）：
   - `next_pair = (X, X) = best_pair`，被加入 `to_update_index`
2. 第一次匹配合并后，序列变为 `... new_id X ...`
3. 内层 while 循环结束后，执行：
   ```python
   for to_update in to_update_index:
       pair_index[to_update].discard(id)
   ```
   其中 `to_update = (X, X) = best_pair`，所以执行：
   ```python
   pair_index[best_pair].discard(id)
   ```
   这**直接修改了 `affected_seq`（就是 `pair_index[best_pair]`）的内容**，而此时外层 `for id in affected_seq` 正在迭代它！

Python 的 `RuntimeError: Set changed size during iteration` 就此触发。

更直观地展示问题：

```python
s = {1, 2, 3}
affected_seq = s          # 直接引用，不是拷贝
for x in affected_seq:
    s.discard(x)          # 修改了正在遍历的集合 → RuntimeError!
```

### 修复思路

在开始迭代之前，先把集合**转换为 list 做一次拷贝**：

```python
affected_seq = list(pair_index.get(best_pair, set()))
```

这样 `for id in affected_seq` 遍历的是一个独立的 list，即使 `pair_index[best_pair]` 在循环中被修改，也不会影响迭代过程。

---

## Bug 3 延伸：old_pairs 与 new_pairs 必须用 delta 合并处理

### 出错位置

处理完所有 affected_seq 后，更新 `pair_freq` 时。

### 修改前

```python
for old_pair, freq in old_pairs.items():
    pair_freq[old_pair] -= freq   # ← 可能 KeyError
    ...

for new_pair, freq in new_pairs.items():
    pair_freq[new_pair] = pair_freq.get(new_pair, 0) + freq
    ...
```

### 修改后

```python
# 合并为 delta 统一处理
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
        heapq.heappush(max_heap, (-new_freq, NegBytes(vocab[pair[0]]),
                                   NegBytes(vocab[pair[1]]), pair))
```

### 为什么出错

当同一个 pretoken 中存在**连续的 best_pair**（如序列 `A B A B`）时，在修复了 Bug 2（prev 指针）后，处理该序列的 inner loop 会产生一个**"过渡期 pair"（transient pair）**：

以 `A B A B`（best_pair = (A, B)，merge 成 new_id）为例，逐步追踪：

**第一次匹配**（第一对 A B）：
- 无左邻，不产生 old/new 左侧 pair
- 右邻是第二个 A：`next_pair = (B, A)`，加入 old_pairs
- 右侧新 pair：`new_right = (new_id, A)`，加入 new_pairs

**第二次匹配**（第二对 A B，此时 cur.prev 已被修复为 new_id 节点）：
- 左邻是 new_id：`prev_pair = (new_id, A)`，加入 old_pairs
- 左侧新 pair：`new_left = (new_id, new_id)`，加入 new_pairs
- 无右邻，不产生 right pair

汇总结果：

| 数据结构    | 内容                                     |
|-------------|------------------------------------------|
| `old_pairs` | `{(B, A): freq, (new_id, A): freq}`      |
| `new_pairs` | `{(new_id, A): freq, (new_id, new_id): freq}` |

注意：**`(new_id, A)` 同时出现在 `old_pairs` 和 `new_pairs` 中**，净变化量为 0——这是正确的，因为 `A B A B` merge 后得到 `new_id new_id`，中间的 `(new_id, A)` 只是一个在同一次 merge 内生灭的过渡 pair，最终不应存在于 `pair_freq` 中。

**原代码先处理 old_pairs，再处理 new_pairs**：

```python
# 第一步：处理 old_pairs
pair_freq[(new_id, A)] -= freq
# ← KeyError！(new_id, A) 在 pair_freq 中根本不存在
#   因为 new_id 是本次 merge 刚创建的，它的相关 pair 还未被加入 pair_freq
```

`(new_id, A)` 作为一个 transient pair，在任何之前的迭代里都不可能出现过（new_id 是本次 merge 才创建的），所以 `pair_freq` 中不存在这个 key，直接 `pair_freq[(new_id, A)] -= freq` 触发 `KeyError`。

**修复思路**：将 old_pairs 和 new_pairs 合并为一个 `delta` 字典，计算每个 pair 的净变化量：

```
delta[(new_id, A)] = +freq（来自 new_pairs）- freq（来自 old_pairs）= 0
```

然后用 `pair_freq.get(pair, 0) + delta` 统一更新。这样：
- transient pair 的 delta = 0，`new_freq = 0`，不会被加入 `pair_freq`
- 真正消失的 pair（如 `(B, A)`）的 delta < 0，频率减少或删除
- 真正新增的 pair（如 `(new_id, new_id)`）的 delta > 0，被正确加入

---

## 修复后完整的 merge 循环核心代码

```python
for i in range(nums_merge):
    while max_heap:
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
    new_id = base_vocab + i
    vocab[new_id] = A + B
    merges.append((A, B))

    # 拷贝集合，防止迭代时被修改（Bug 4）
    affected_seq = list(pair_index.get(best_pair, set()))
    old_pairs = defaultdict(int)
    new_pairs = defaultdict(int)

    for id in affected_seq:
        freq = pretokens[id].freq
        cur = pretokens[id].head
        to_update_index = set()
        while cur is not None and cur.next is not None:
            if cur.val == best_pair[0] and cur.next.val == best_pair[1]:
                # Bug 1 修复：不再有 new_pairs[best_pair] 和 pair_index[best_pair].add(id)
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
                    cur.next.prev = cur  # Bug 2 修复：维护双向链表
                del tmp
                cur.val = new_id
            cur = cur.next
        cur = pretokens[id].head
        while cur is not None and cur.next is not None:
            to_update_index.discard((cur.val, cur.next.val))
            cur = cur.next
        for to_update in to_update_index:
            pair_index[to_update].discard(id)

    # Bug 3 修复：显式删除 best_pair
    pair_freq.pop(best_pair, None)
    pair_index.pop(best_pair, None)

    # Bug 3 延伸修复：用 delta 合并 old/new pairs，避免 transient pair 的 KeyError
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
                (-new_freq, NegBytes(vocab[pair[0]]), NegBytes(vocab[pair[1]]), pair),
            )
```

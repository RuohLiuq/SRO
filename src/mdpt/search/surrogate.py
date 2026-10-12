"""surrogate.py — 纯 Python 多输出随机森林（零依赖），给 BO acquisition 当代理模型。

为什么自带 RF 而不用 sklearn：
  整个 mdpt 是 stdlib + pydantic、可离线跑。RF 在我们这种"小样本 + 类别为主 + 结构化"
  的 recipe 空间里比 GP 稳，且我们需要"逐树预测"来做 Monte-Carlo 不确定性，自己实现
  反而更可控。

结构：
  _RegressionTree   单棵回归树（按方差/SSE 贪心分裂，特征子采样）。
  RandomForest      bootstrap + 特征子采样的树集成；暴露 tree_predictions() 供 MC。
  MultiOutputSurrogate  每个输出维度（各 target domain 的 gain）一个 RF；
                        sample() 通过"逐树预测里随机抽"产生 MC 联合样本。

不确定性来源：森林内各树预测的分歧（disagreement）。样本极少时分歧大 → σ 大 → 探索。
"""
from __future__ import annotations

import math
import random
from typing import Optional


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _sse(xs: list[float]) -> float:
    if not xs:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs)


def _var(xs: list[float]) -> float:
    return _sse(xs) / len(xs) if xs else 0.0


class _Node:
    __slots__ = ("feat", "thr", "left", "right", "value")

    def __init__(self) -> None:
        self.feat: Optional[int] = None
        self.thr: float = 0.0
        self.left: Optional["_Node"] = None
        self.right: Optional["_Node"] = None
        self.value: Optional[float] = None  # 叶子才有值


class _RegressionTree:
    """CART 风格回归树：在随机特征子集上找最小 SSE 的二分切分。"""

    def __init__(self, max_depth: int = 4, min_samples: int = 2,
                 feat_frac: float = 0.7, rng: Optional[random.Random] = None) -> None:
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.feat_frac = feat_frac
        self.rng = rng or random.Random(0)
        self.root: Optional[_Node] = None

    def fit(self, X: list[list[float]], y: list[float]) -> "_RegressionTree":
        self.root = self._build(list(range(len(X))), X, y, 0)
        return self

    def _build(self, idx: list[int], X: list[list[float]], y: list[float],
               depth: int) -> _Node:
        node = _Node()
        ys = [y[i] for i in idx]
        node_mean = _mean(ys)
        if (depth >= self.max_depth or len(idx) <= self.min_samples
                or _var(ys) < 1e-12):
            node.value = node_mean
            return node

        nfeat = len(X[0]) if X else 0
        k = max(1, int(self.feat_frac * nfeat))
        feats = self.rng.sample(range(nfeat), min(k, nfeat)) if nfeat else []

        best = None  # (sse, feat, thr, left_idx, right_idx)
        for fj in feats:
            vals = sorted({X[i][fj] for i in idx})
            for a, b in zip(vals, vals[1:]):
                thr = (a + b) / 2.0
                left = [i for i in idx if X[i][fj] <= thr]
                right = [i for i in idx if X[i][fj] > thr]
                if not left or not right:
                    continue
                sse = _sse([y[i] for i in left]) + _sse([y[i] for i in right])
                if best is None or sse < best[0]:
                    best = (sse, fj, thr, left, right)

        if best is None:
            node.value = node_mean
            return node

        _, fj, thr, left, right = best
        node.feat = fj
        node.thr = thr
        node.left = self._build(left, X, y, depth + 1)
        node.right = self._build(right, X, y, depth + 1)
        return node

    def predict(self, x: list[float]) -> float:
        n = self.root
        while n is not None and n.value is None:
            n = n.left if x[n.feat] <= n.thr else n.right  # type: ignore[index]
        return n.value if n is not None else 0.0


class RandomForest:
    """bootstrap + 特征子采样的回归森林。"""

    def __init__(self, n_estimators: int = 24, max_depth: int = 4,
                 feat_frac: float = 0.7, seed: int = 0) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feat_frac = feat_frac
        self.seed = seed
        self.trees: list[_RegressionTree] = []

    def fit(self, X: list[list[float]], y: list[float]) -> "RandomForest":
        self.trees = []
        n = len(X)
        if n == 0:
            return self
        rng = random.Random(self.seed)
        for t in range(self.n_estimators):
            # bootstrap：有放回采样
            boot = [rng.randrange(n) for _ in range(n)]
            Xb = [X[i] for i in boot]
            yb = [y[i] for i in boot]
            tree = _RegressionTree(
                max_depth=self.max_depth, feat_frac=self.feat_frac,
                rng=random.Random(self.seed * 1000 + t),
            )
            tree.fit(Xb, yb)
            self.trees.append(tree)
        return self

    def tree_predictions(self, x: list[float]) -> list[float]:
        """每棵树各给一个预测（MC 的样本来源）。"""
        return [tree.predict(x) for tree in self.trees]

    def predict(self, x: list[float]) -> float:
        preds = self.tree_predictions(x)
        return _mean(preds)


class MultiOutputSurrogate:
    """每个输出 key 一个 RF；sample() 产生跨输出的 Monte-Carlo 联合样本。"""

    def __init__(self, keys: list[str], n_estimators: int = 24,
                 max_depth: int = 4, feat_frac: float = 0.7, seed: int = 0) -> None:
        self.keys = list(keys)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feat_frac = feat_frac
        self.seed = seed
        self.forests: dict[str, RandomForest] = {}

    @property
    def fitted(self) -> bool:
        return bool(self.forests)

    def fit(self, X: list[list[float]],
            targets: list[dict[str, float]]) -> "MultiOutputSurrogate":
        self.forests = {}
        for ki, key in enumerate(self.keys):
            y = [float(t.get(key, 0.0)) for t in targets]
            rf = RandomForest(self.n_estimators, self.max_depth,
                              self.feat_frac, seed=self.seed + ki)
            rf.fit(X, y)
            self.forests[key] = rf
        return self

    def predict_mean(self, x: list[float]) -> dict[str, float]:
        return {k: rf.predict(x) for k, rf in self.forests.items()}

    def sample(self, x: list[float], n_samples: int = 64,
               rng: Optional[random.Random] = None) -> list[dict[str, float]]:
        """对每个 key 缓存逐树预测，再随机抽 → n_samples 个联合 y 向量。"""
        rng = rng or random.Random(0)
        tree_preds = {k: rf.tree_predictions(x) for k, rf in self.forests.items()}
        out: list[dict[str, float]] = []
        for _ in range(n_samples):
            sample = {}
            for k, preds in tree_preds.items():
                sample[k] = rng.choice(preds) if preds else 0.0
            out.append(sample)
        return out

"""HMM states, emission/transition probabilities, Viterbi decoding. Owner: C

Hidden states   : the candidate arcs of each GPS point (arcs within 50 m, step 02).
Emission        : log of the classifier probability for (point, arc), weighted by w_emit.
Transition      : based on how many arc-to-arc moves the network needs to get from
                  the previous arc to the next one (directed, following arc order):
                    same arc          -> 0 hops
                    next arc          -> 1 hop, and so on, searched up to max_hops
                  log P = -beta * hops, minus uturn_pen if the move reverses the arc.
                  If the next arc is not reachable within max_hops (different network
                  piece, or too far), the move still gets a finite score
                  -beta*(max_hops+1) - disconnect_pen. This keeps decoding from failing
                  on fragmented networks; the classifier then decides.
GPS gaps        : if the time gap between consecutive points exceeds gap_s
                  (decisions log D-003), the chain restarts: no transition across it.
Output          : one predicted arc per GPS point that has candidates.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class HMMParams:
    beta: float = 1.0           # cost per hop
    w_emit: float = 1.0         # weight on log classifier probability
    max_hops: int = 6           # search depth for reachability
    disconnect_pen: float = 10.0
    uturn_pen: float = 0.0
    gap_s: float = 300.0        # D-003 provisional threshold


class ArcGraph:
    """Directed arc-to-arc successor graph with cached bounded BFS."""

    def __init__(self, from_node, to_node, max_hops):
        self.fr = np.asarray(from_node)
        self.to = np.asarray(to_node)
        self.max_hops = max_hops
        out = {}
        for a, u in enumerate(self.fr):
            out.setdefault(int(u), []).append(a)
        self.out = out
        self._cache = {}

    def hops_from(self, a):
        """Dict arc -> hops for every arc reachable from arc a within max_hops."""
        h = self._cache.get(a)
        if h is not None:
            return h
        h = {a: 0}
        q = deque([a])
        while q:
            x = q.popleft()
            d = h[x]
            if d == self.max_hops:
                continue
            for y in self.out.get(int(self.to[x]), ()):
                if y not in h:
                    h[y] = d + 1
                    q.append(y)
        self._cache[a] = h
        return h

    def is_uturn(self, a, b):
        return self.fr[b] == self.to[a] and self.to[b] == self.fr[a]


def transition_matrix(graph, prev_arcs, cur_arcs, prm):
    far = -prm.beta * (prm.max_hops + 1) - prm.disconnect_pen
    T = np.full((len(prev_arcs), len(cur_arcs)), far)
    for i, a in enumerate(prev_arcs):
        h = graph.hops_from(int(a))
        for j, b in enumerate(cur_arcs):
            k = h.get(int(b))
            if k is not None:
                T[i, j] = -prm.beta * k - (prm.uturn_pen if k == 1 and graph.is_uturn(a, b) else 0.0)
    return T


def viterbi(points, graph, prm):
    """points: list of (t, arcs ndarray, probs ndarray) in time order.
    Returns list of predicted arc per point."""
    out = [None] * len(points)
    seg_start = 0
    delta, back = None, []

    def flush(end):
        # backtrack the segment points[seg_start:end]
        if delta is None:
            return
        j = int(np.argmax(delta))
        for k in range(end - 1, seg_start - 1, -1):
            out[k] = int(points[k][1][j])
            if k > seg_start:
                j = int(back[k - seg_start - 1][j])

    for k, (t, arcs, p) in enumerate(points):
        emit = prm.w_emit * np.log(np.clip(p, 1e-9, 1.0))
        if delta is None or t - points[k - 1][0] > prm.gap_s:
            flush(k)
            seg_start, back = k, []
            delta = emit
            continue
        T = transition_matrix(graph, points[k - 1][1], arcs, prm)
        scores = delta[:, None] + T
        bp = np.argmax(scores, axis=0)
        back.append(bp)
        delta = scores[bp, np.arange(len(arcs))] + emit
    flush(len(points))
    return out


def route_continuity(pred_arcs, times, graph, gap_s):
    """Share of consecutive point pairs (not across a GPS gap) whose predicted arcs
    are the same arc or directly connected (1 hop)."""
    ok = n = 0
    for k in range(1, len(pred_arcs)):
        if times[k] - times[k - 1] > gap_s:
            continue
        a, b = pred_arcs[k - 1], pred_arcs[k]
        n += 1
        ok += a == b or graph.hops_from(a).get(b, 99) <= 1
    return ok / n if n else float("nan")

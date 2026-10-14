"""Bounded, opt-in wall-clock profiling of the real online producer."""
from contextlib import contextmanager
import math
import time


class ChunkTimer:
    def __init__(self, enabled):
        self.enabled = enabled
        self.values = {}

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.values[name] = self.values.get(name, 0.) + (time.perf_counter() - start) * 1000


def summarize(rows, warmup_chunks=25):
    selected = [row for row in rows if row['chunk'] >= warmup_chunks]
    stats = {}
    for key in sorted({key for row in selected for key in row if key.endswith('_ms')}):
        values = sorted(row[key] for row in selected if key in row)
        stats[key] = dict(mean=sum(values) / len(values), p50=values[(len(values) - 1) // 2],
                          p95=values[math.ceil(len(values) * .95) - 1], maximum=values[-1])
    return dict(chunks=len(selected), warmup_chunks=warmup_chunks, stages=stats,
                note='Nested timings are not additive: publication is inside stream_wall for early publish. '
                     'Background preprocessing/native capture overlap the producer and must not be added to its cycle. '
                     'CUDA event intervals include host launch gaps; clean_kv includes the publish callback.')

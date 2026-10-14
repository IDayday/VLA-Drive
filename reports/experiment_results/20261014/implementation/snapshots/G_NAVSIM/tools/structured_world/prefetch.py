"""At most one ordered CPU batch ahead; no task RNG or data-position advance."""
from concurrent.futures import ThreadPoolExecutor


class OrderedScenePrefetch:
    def __init__(self, dataset, collate, scene_pool):
        self.dataset, self.collate, self.scene_pool = dataset, collate, scene_pool
        self.coordinator = ThreadPoolExecutor(max_workers=1)
        self.pending = None

    def _load(self, indices):
        return self.collate(list(self.scene_pool.map(self.dataset.__getitem__, indices)))

    def submit(self, location, indices):
        if self.pending is not None:
            raise RuntimeError('Only one unread prefetched batch is allowed')
        indices = tuple(int(i) for i in indices)
        self.pending = (location, indices, self.coordinator.submit(self._load, indices))

    def consume(self, location, indices):
        indices = tuple(int(i) for i in indices)
        if self.pending is None:
            self.submit(location, indices)
        key, expected, future = self.pending
        if key != location or expected != indices:
            raise RuntimeError('Prefetch scene order/data boundary changed')
        result = future.result()
        self.pending = None
        return result

    def close(self):
        if self.pending is not None:
            self.pending[2].cancel()
        self.coordinator.shutdown(wait=True, cancel_futures=True)
        self.pending = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

"""Ordered CPU-only fixed-bank IL prefetch; counters commit on consumption.

The worker builds deterministic immutable observations/targets, never Qwen or
policy features. Unconsumed futures do not advance the checkpoint cursor/calls.
Only one worker owns the cache readers; there is no concurrent builder access.
"""
from concurrent.futures import ThreadPoolExecutor


class OrderedUpdatePrefetch:
    def __init__(self,builder,request,*,start_update,start_cursor,depth=2,scenes_per_update=32):
        if depth not in(1,2):raise ValueError('Bounded depth1 or2 required')
        self.builder,self.request=builder,request
        self.update,self.cursor=start_update,start_cursor
        self.depth,self.scenes=depth,scenes_per_update
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='fixed_IL_CPU_prefetch')
        self.pending={};self.closed=False;self.consumed_updates=0
        self._fill()

    def _build(self,update,cursor):
        return [self.builder.build(tokens,positions,update,account=False)
                for tokens,positions in self.request(update,cursor)]

    def _fill(self):
        for offset in range(self.depth):
            update=self.update+offset
            if update not in self.pending:
                self.pending[update]=self.pool.submit(self._build,update,self.cursor+offset*self.scenes)

    def next(self,update,cursor):
        if self.closed or(update,cursor)!=(self.update,self.cursor):raise ValueError('Prefetch cursor/order mismatch')
        batches=self.pending.pop(update).result() # Exceptions reach the normal distributed data-error path.
        self.builder.calls+=len(batches)
        self.update+=1;self.cursor+=self.scenes;self.consumed_updates+=1
        self._fill();return batches

    def close(self):
        if self.closed:return
        self.closed=True
        for future in self.pending.values():future.cancel()
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.pending.clear()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()

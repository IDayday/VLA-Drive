"""Ordered CPU-only fixed-bank IL prefetch; counters commit on consumption.

The worker builds deterministic immutable observations/targets, never Qwen or
policy features. Unconsumed futures do not advance the checkpoint cursor/calls.
Only one worker owns the cache readers; there is no concurrent builder access.
"""
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import time


class OrderedUpdatePrefetch:
    def __init__(self,builder,request,*,start_update,start_cursor,depth=2,scenes_per_update=32,builder_factory=None):
        if depth not in(1,2):raise ValueError('Bounded depth1 or2 required')
        # Candidate/evidence SQLite connections belong to their creating
        # thread. Build independent read-only readers inside the worker; never
        # disable SQLite's ownership check or share mutable reader caches.
        if builder_factory is None and any(isinstance(getattr(getattr(builder,name,None),'db',None),sqlite3.Connection)
                for name in('bank','evidence')):
            raise ValueError('Database-backed prefetch requires a worker-local builder_factory')
        self.builder,self.request=builder,request
        self.builder_factory=builder_factory;self.worker_builder=None
        self.update,self.cursor=start_update,start_cursor
        self.depth,self.scenes=depth,scenes_per_update
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='fixed_IL_CPU_prefetch')
        self.pending={};self.closed=False;self.consumed_updates=0
        self.prepared_updates=0;self.failed_preparations=0;self.preparation_seconds=0.;self.attempted_microbatches=0
        self._fill()

    def _build(self,update,cursor):
        start=time.perf_counter()
        try:return self._build_update(update,cursor)
        except BaseException:
            self.failed_preparations+=1;raise
        finally:self.preparation_seconds+=time.perf_counter()-start

    def _build_update(self,update,cursor):
        builder=self.builder
        if self.builder_factory is not None:
            if self.worker_builder is None:
                self.worker_builder=self.builder_factory()
                expected={k:v for k,v in builder.state_dict().items()if k!='calls'}
                actual={k:v for k,v in self.worker_builder.state_dict().items()if k!='calls'}
                if expected!=actual:raise ValueError('Worker builder fixed-bank identity mismatch')
            builder=self.worker_builder
        result=[]
        for tokens,positions in self.request(update,cursor):
            self.attempted_microbatches+=1
            result.append(builder.build(tokens,positions,update,account=False))
        self.prepared_updates+=1
        return result

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
        # Dispose connections on their creating thread after any active build.
        try:
            if self.builder_factory is not None:
                self.pool.submit(self._close_worker).result()
        finally:
            self.pool.shutdown(wait=True,cancel_futures=True)
            self.pending.clear()

    def _close_worker(self):
        if self.worker_builder is not None:
            self.worker_builder.close();self.worker_builder=None

    def statistics(self):
        return dict(consumed_updates=self.consumed_updates,prepared_updates=self.prepared_updates,
            unconsumed_prepared_updates=self.prepared_updates-self.consumed_updates,
            attempted_microbatches=self.attempted_microbatches,failed_preparations=self.failed_preparations,
            CPU_reader_worker_seconds=self.preparation_seconds,
            semantics='Physical CPU-only preparation includes unconsumed work; only consumption advances native builder/data progress.')

    def __enter__(self):return self
    def __exit__(self,*_):self.close()

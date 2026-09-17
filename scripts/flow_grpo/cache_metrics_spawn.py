"""Official NAVSIM cache entry, spawn workers to avoid inheriting the full scene index.

Only the multiprocessing start method changes. Official scene selection,
MetricCacheProcessor, maps, traffic/reference data and serialization are intact.
"""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    from navsim.planning.script.run_metric_caching import main

    main()

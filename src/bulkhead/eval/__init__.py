"""Bulkhead part 7: the attack suite, overhead and usability benchmarks, and the results report.

- ``attacks``: the versioned attack dataset (``evaluation/data/bulkhead_attacks_v1.json``),
  its strict loader, the verdict judge, and the resumable per-profile runner;
- ``overhead``: serial cold-start and scripted whole-episode benchmarks from raw samples;
- ``usability``: ASK rate, over-denial, and utility loss under declared approval modes;
- ``report``: ``results/bulkhead/RESULTS.md`` generated only from those records.

Nothing here weakens a sandbox or chooses one. Every number comes from records
written by these modules; a profile this host cannot run is reported as
untested, never as passed.
"""

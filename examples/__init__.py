"""Configuration-driven entry points: the one layer that reads a RunConfig.

Each module here turns a parsed config into the explicit arguments a `pipeline/` function takes, loads the files that
run names, and decides what to save and print. Everything below this layer is callable without a config file at all.
This is the same split WILDS draws at `examples/run_expt.py` and detectron2 at `from_config`.
"""

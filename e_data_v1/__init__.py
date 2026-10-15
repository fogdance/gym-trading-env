"""Standalone S3-E Data V1. Importing this package never starts an Env or GPU."""
from .schema import Decision, EDataBundle, EDataConfig, SCHEMA
from .core import EDataError, MarketData, aggregate_buckets, build_e_data_v1

__all__ = ["Decision", "EDataBundle", "EDataConfig", "SCHEMA", "EDataError",
           "MarketData", "aggregate_buckets", "build_e_data_v1"]

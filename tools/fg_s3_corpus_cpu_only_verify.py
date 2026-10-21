"""Run the legacy verifier with an explicit CPU-only scheduling override.

Import the original producer from its working directory/PYTHONPATH. Keep its
source inventory, data checks and teacher calculations intact. The caller
records this scheduling adapter's path and SHA in the verification receipt.
"""
import os
from pathlib import Path

from gym_trading_env.research import fg_s3_e_quality_corpus_cli as producer


if __name__ == '__main__':
    if os.environ.get('FG_CORPUS_MEMORY_CHECKS') != '0':
        raise ValueError('CPU-only verifier requires an explicit memory-check override')
    original_limits = producer._worker_limits

    def cpu_only_limits(jobs, **kwargs):
        # Reuse the producer's CPU/account/teacher allocation. Its available
        # memory argument cannot restrict CPU slots in this explicit mode.
        kwargs['available'] = 2**127
        limits = original_limits(jobs, **kwargs)
        limits['available_memory_bytes'] = int(next(
            line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith('MemAvailable:'))) * 1024
        limits['memory_checks'] = False
        limits['worker_limit_reason'] = 'cpu_budget_forced'
        return limits

    producer._worker_limits = cpu_only_limits
    raise SystemExit(producer.main())

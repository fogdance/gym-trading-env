"""CPU consumer example. No model construction and no training integration."""
from e_data_v1 import Decision, EDataConfig, build_e_data_v1
from e_data_v1.sources import pool_entries, read_fg_source


def market_inputs(source_pool, contract, cutoff):
    entries,pool_record=pool_entries(source_pool,[contract])
    source=read_fg_source(entries[0],pool_record)
    bundle=build_e_data_v1(source.bars,[Decision(contract,cutoff)],source.schedule,
                           EDataConfig(audit_1m=True))
    # Future E-Model review must explicitly choose this minute input version.
    return dict(market_seq_e_shared=bundle.market_seq_e_shared[0],
                market_context_15m=bundle.context_15m[0]),bundle

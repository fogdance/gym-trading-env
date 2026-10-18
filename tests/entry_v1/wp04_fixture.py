"""Settings for controlled F1 CPU tests, never real TRAIN authority."""
from gym_trading_env.research.fg_s3_e_quality_foundation import CONFIG_SCHEMA


def settings(root, *, target=8, burnin=4):
    return dict(schema=CONFIG_SCHEMA,stage='FOUNDATION_WM',sampling_stage='WM_NATURAL',
        sampling_spec=dict(mode='NATURAL'),model_init=dict(seed=202,presets=['size1m']),
        batch_size=1,target_steps=target,burnin_steps=burnin,optimizer_updates=4,
        report_every_updates=1,save_every_updates=2,device='cpu',
        corpus_root=str(root),prepared_dir=str(root.parent/'prepared'),run_dir=str(root.parent/'run'))

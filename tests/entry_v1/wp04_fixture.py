"""Settings for controlled F1 CPU tests, never real TRAIN authority."""
from gym_trading_env.research.fg_s3_e_quality_foundation import CONFIG_SCHEMA


def settings(root, *, target=8, burnin=4):
    return dict(schema=CONFIG_SCHEMA,stage='FOUNDATION_WM',sampling_stage='WM_NATURAL',
        sampling_spec=dict(mode='NATURAL'),model_init=dict(seed=202,presets=['size1m']),
        batch_size=1,target_steps=target,burnin_steps=burnin,optimizer_updates=4,
        report_every_updates=1,save_every_updates=2,device='cpu',
        corpus_root=str(root),prepared_dir=str(root.parent/'prepared'),run_dir=str(root.parent/'run'))


def native_reader(root, *, target=8, burnin=4, corpus=None):
    from wp03_fixture import native_corpus
    from gym_trading_env.research.fg_s3_e_quality_supervision import SupervisionSource,QualitySupervisionAdmission
    from gym_trading_env.research.fg_s3_e_quality_reader import build_join_index,EQualityCourseReader
    corpus=corpus or native_corpus(root/'corpus',minutes=210,chunk=240)
    admission=QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(corpus))
    index=root/'join.sqlite';build_join_index(admission,index)
    reader=EQualityCourseReader.open(admission,stage='WM_NATURAL',sampling_spec=dict(mode='NATURAL'),seed=202,index_path=index)
    return settings(corpus,target=target,burnin=burnin),reader


def multiscale_reader(root, *, target=64, burnin=32):
    """Original feature builders, visible history, six genuine native accounts."""
    import pandas as pd
    import numpy as np
    from wp02_fixture import native_factory
    from wp03_fixture import native_corpus
    from gym_trading_env.research.fg_replay import digest
    bundle=native_factory(root/'market',minutes=420,chunk=500)
    factory,_,profile=bundle;path=root/'market/market.csv'
    frame=pd.read_csv(path)
    shift=(np.arange(len(frame))//3)%9-4  # Causal integer-tick synthetic prices.
    for key in ('Open','High','Low','Close'):frame[key]+=shift
    frame.to_csv(path,index=False)
    profile.source_receipt['sources']['CZCE.FG2701']['market']['sha256']=digest(path)
    env=factory()
    try:
        for _ in range(320):env.step(0)
        initial=env.export_state()  # Keep the complete financial/E history.
    finally:env.close()
    corpus=native_corpus(root/'corpus',factory_bundle=bundle,initial_state=initial)
    return native_reader(root,target=target,burnin=burnin,corpus=corpus)


def synthetic_foundation(root, *, multiscale=False, device='cpu'):
    """Build current-contract engineering evidence through the original F1 run."""
    import json
    from pathlib import Path
    import subprocess
    import sys
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if multiscale:
        s,r=multiscale_reader(root)
    else:
        from wp03_fixture import native_corpus
        corpus=native_corpus(root/'corpus',minutes=100,chunk=120)
        s,r=native_reader(root,corpus=corpus,target=64,burnin=32)
    index=r.index_path;r.close();s['device']=device
    s['run_dir']=str(root/'run')
    path=root/'settings.json';path.write_text(json.dumps(s))
    argv=[sys.executable,'-B',str(Path(__file__).with_name('wp04_cold_worker.py')),
          str(path),s['run_dir'],str(index),'start','4']
    with (root/'build.log').open('x') as log:
        result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('SYNTHETIC_F1_BUILD_FAILED: '+(root/'build.log').read_text()[-4000:])
    return s

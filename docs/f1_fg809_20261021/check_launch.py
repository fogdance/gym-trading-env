"""Read config, contracts and frozen identities only. No Agent or training."""
import json
from pathlib import Path

from gym_trading_env.research.fg_replay import digest, read_manifest
from gym_trading_env.research.fg_s3_corpus_compatibility import CorpusCompatibility
from gym_trading_env.research.fg_s3_e_quality_foundation import read_settings, model_config
from gym_trading_env.research.fg_s3_e_quality_supervision import frozen_producer_audit


def main():
    repo=Path(__file__).resolve().parents[2]
    config=repo/'configs/experiments/fg_s3_v2/e_quality_f1_fg809_size12m_u10000_20261021.yaml'
    lock=json.loads((repo.parent/'release/LAUNCH_LOCK.json').read_text())
    settings=read_settings(config)
    if digest(config)!=lock['config']['sha256']:
        raise ValueError('Launch configuration differs from the frozen selection')
    root=Path(settings['corpus_root'])
    manifest=read_manifest(root/'corpus_manifest.json')
    source=read_manifest(root/'source_lock.json')
    if (source['contract']!='CZCE.FG809' or source['fold']!='TRAIN'
            or manifest['content_hash']!=lock['corpus']['identity']):
        raise ValueError('This Run is restricted to the selected FG809 TRAIN Corpus')
    compatibility=CorpusCompatibility.resolve(manifest,json.loads((root/'course_v2.json').read_text()),
        read_manifest(root/'teacher_manifest.json'),read_manifest(root/'model_contract.json'))
    audit=frozen_producer_audit(repo,producer_lock=lock['verifier_pin']['path'],
        producer_lock_sha256=lock['verifier_pin']['sha256'])
    cfg=model_config(settings)
    prepared=Path(settings['prepared_dir'])
    ready=prepared/'f1_preparation.json'
    return dict(status='F1_LAUNCH_CONFIGURATION_READY',settings=settings,
        compatibility=compatibility.as_dict(),corpus_identity=manifest['content_hash'],
        contract=source['contract'],selection=source.get('selection'),
        verifier_audit=audit,model_rssm=cfg['agent']['dyn']['rssm'],
        sequence_rows=1+settings['burnin_steps']+settings['target_steps'],
        target_positions_budget=settings['batch_size']*settings['target_steps']*settings['optimizer_updates'],
        f1_preparation=read_manifest(ready)['status'] if ready.is_file() else 'NOT_RUN',
        agent_created=False,gpu_execution=False,optimizer_updates_by_this_check=0,
        reader_readiness='USER_INSTRUCTED_TO_TREAT_AS_READY; NOT_RECHECKED',
        actual_f1_data_admission='REQUIRED_BY_EXISTING_NATIVE_F1_ENTRYPOINT')


if __name__=='__main__':
    print(json.dumps(main(),indent=2,ensure_ascii=False))

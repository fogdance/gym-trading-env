"""Compare full logical state across independent Runs; prove exact graph resume."""
import json
import pickle
from gym_trading_env.research.fg_s3_reader_versions import materialize_prefetch


def runtime_evidence(run):
    state=run.runtime.save()
    state['streams']=[materialize_prefetch(run.source,state['streams'][0])]
    return state


def prove_graph_restore(run):
    saved=pickle.loads((run.cp.latest()/'runtime.pkl').read_bytes())['streams'][0]['source']
    with run.stream.pause():current=run.stream.save_full()['source']
    assert saved==current,'COLD_RESTORE_CHANGED_VERSION_GRAPH'
    (run.root/'exact_graph_restore.json').write_text(json.dumps(dict(
        protocol=current['protocol'],graph_sha256=current['graph_sha256'],session=current['session'],
        complete_graph_byte_structure_equal=True,optimizer_updates_before_continuation=int(run.agent.n_updates)),indent=2)+'\n')

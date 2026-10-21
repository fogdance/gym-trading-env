"""Run selected CPU tests in an isolated cwd and retain exact command receipts."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('tests', nargs='+')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = root/'artifacts/batch3'/f'{args.label}_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    receipts = root/'docs/batch3_r2_r8_r3_20261021/receipts'
    receipts.mkdir(parents=True, exist_ok=True)
    for name in ('wp04', 'wp05', 'wp06', 'wp07', 'wp08'):
        (out/'docs/receipts'/name).mkdir(parents=True)
    initial = root/'docs/receipts/wp04/B_initial_layout.json'
    if initial.is_file():
        shutil.copyfile(initial, out/'docs/receipts/wp04/B_initial_layout.json')
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(map(str,
        (root, root/'src', root.parent/'dreamerv3', root/'tests/entry_v1'))),
        'JAX_PLATFORMS': 'cpu', 'CUDA_VISIBLE_DEVICES': '',
        'PYTHONDONTWRITEBYTECODE': '1', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
        'MPLCONFIGDIR': str(out/'.matplotlib'),
        'FG_WP07_SOURCE_ROOT': os.environ.get('FG_BATCH3_SOURCE_ROOT',str(root/'artifacts/batch3/synthetic_f3_source')),
        'FG_WP07_C_ROOT': str(root/'artifacts/fg_s3_v2/e_quality_wp07'/f'batch3_{stamp}'/'live'),
        'FG_WP07_G_ROOT': str(root/'artifacts/fg_s3_v2/e_quality_wp07'/f'batch3_{stamp}'/'cold'),
        'FG_WP07_B_ROOT': str(root/'artifacts/fg_s3_v2/e_quality_wp07'/f'batch3_{stamp}'/'shadow'),
        'FG_WP07_H_PROBE': str(root/'artifacts/fg_s3_v2/e_quality_wp07'/f'batch3_{stamp}')}
    env.pop('LD_LIBRARY_PATH', None)  # Avoid the CLI's deliberate process restart inside pytest.
    xml = receipts/f'{args.label}_{stamp}.xml'
    log = receipts/f'{args.label}_{stamp}.log'
    command = [sys.executable, '-B', '-m', 'pytest', '-p', 'no:cacheprovider',
        '-o', 'addopts=', '-q', '--junitxml='+str(xml),
        *[str(root/t) if not Path(t.split('::')[0]).is_absolute() else t for t in args.tests]]
    repositories = {}
    for name, path in (('gym',root), ('dreamer',root.parent/'dreamerv3')):
        repositories[name] = dict(path=str(path), sha=subprocess.check_output(
            ['git','-C',str(path),'rev-parse','HEAD'],text=True).strip(),
            status=subprocess.check_output(['git','-C',str(path),'status','--porcelain=v1'],text=True))
    began = time.monotonic()
    with log.open('x') as stream:
        result = subprocess.run(command, cwd=out, env=env, stdout=stream, stderr=subprocess.STDOUT)
    receipt = dict(argv=command, cwd=str(out), environment={k:env[k] for k in (
        'PYTHONPATH','JAX_PLATFORMS','CUDA_VISIBLE_DEVICES','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
        repositories=repositories, exit_code=result.returncode, seconds=time.monotonic()-began,
        log=str(log), log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(), junit=str(xml),
        scope='SELECTED_CPU_SYNTHETIC_NOT_FORMAL')
    if xml.is_file():
        import xml.etree.ElementTree as ET
        suites = ET.parse(xml).getroot()
        receipt['cases'] = [dict(node.attrib) for node in suites.iter('testcase')]
        receipt['summary'] = [dict(node.attrib) for node in suites.iter('testsuite')]
    receipt_path = log.with_suffix('.command.json')
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(receipt=str(receipt_path),exit_code=result.returncode,seconds=receipt['seconds'])))
    print(log.read_text()[-6000:])
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Collate signed CPU receipts into reviewable docs; never open price data.

Incomplete stages remain explicit. A completion summary requires every CPU
receipt and verifies the linked report bytes before publishing it. Stage CLIs
are responsible for their complete native artifact verification.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import json
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gym_trading_env.research.fg_replay import digest, read_manifest


STAGES = {
    'E': ('env/run_20261011_04', 'E'),
    'R': ('reward/run_20261011_02', 'R'),
    'Input': ('input_audit/run_20261011_01', 'INPUT_AUDIT'),
    'O': ('oracle/run_20261011_01', 'O'),
    'D generation': ('replay/run_20261011_01', 'D_GENERATED'),
    'D coverage': ('coverage/run_20261011_01', 'D_AUDIT'),
    'H': ('horizon/run_20261011_01', 'H'),
    'Pilot preparation': ('pilot/run_20261011_01', 'PILOT_PREPARED'),
}


def table(headers, rows):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return ['| ' + ' | '.join(map(cell, headers)) + ' |',
            '| ' + ' | '.join('---' for _ in headers) + ' |',
            *['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows]]


def publish(path, text):
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def checked_report(root, receipt, filename):
    path = root / filename
    if receipt['files'].get(filename) != digest(path):
        raise ValueError(f'Signed report bytes changed: {path}')
    return path.read_text(encoding='utf-8')


def collate(root, docs, *, require_complete=False, stages=None):
    stages = stages or STAGES
    root = root.resolve(); docs = docs.resolve(); docs.mkdir(parents=True, exist_ok=True)
    found = {}
    for name, (relative, expected) in stages.items():
        directory = root / relative; path = directory / 'manifest.json'
        if not path.exists():
            continue
        receipt = read_manifest(path)
        if receipt.get('status') != 'PASS' or receipt.get('stage') != expected:
            raise ValueError(f'Unexpected CPU receipt: {path}')
        if name in ('Input', 'O', 'D generation', 'D coverage') and not receipt['full_training_fold']:
            raise ValueError(f'Partial TRAIN cannot pass the formal Gate: {path}')
        found[name] = (directory, receipt)
    complete = len(found) == len(stages)
    if require_complete and not complete:
        raise ValueError('CPU stages incomplete: ' + ', '.join(set(stages) - set(found)))
    if 'O' in found:
        directory, oracle = found['O']
        if oracle['oracle_optimality_status'] != 'exact_proven' or len(oracle['contracts']) != 18:
            raise ValueError('Formal Oracle proof is incomplete')
        oracle_doc = docs / 'FG_Single3_Oracle_Validation.md'
        design = oracle_doc.read_text(encoding='utf-8').split('\n## 实际全TRAIN验收\n')[0]
        old_status = '状态：IN_PROGRESS；本文件目前是算法与验收记录，不能作为O-Gate完成报告。'
        design = design.replace(old_status, '状态：PASS；完整TRAIN精确O-Gate与四候选原Env达到性验收已完成。')
        rows = []; plans = []
        for contract in oracle['contracts']:
            verify = contract['candidate_verification']
            if verify['all_candidates_exhaustively_replayed']:
                raise ValueError('Unexpected exhaustive reference claim')
            rows.append((contract['contract'], contract['candidates'], len(verify['sampled_cursors']),
                         contract['funds_certificate']['required_equity_upper_bound'],
                         round(contract['elapsed_seconds'], 3), contract['peak_rss_bytes']))
            for trace in contract['collection']:
                actual = trace['objective_replay']; candidate = trace['manifest']['reward_candidate']
                if not (actual['every_trade_matched'] and actual['every_feedback_matched']):
                    raise ValueError('Oracle has unmatched actual feedback')
                plan = read_manifest(contract['H333_schedules'][candidate])
                gamma1 = contract['gamma1_diagnostics'][candidate]
                plans.append((contract['contract'], candidate, plan['selected_trades'],
                              plan['value_interval'], actual['actual_discounted_float64'],
                              actual['actual_cash_gain'], gamma1['selected_trades'], gamma1['value_interval']))
        total = sum(c['candidates'] for c in oracle['contracts'])
        samples = sum(len(c['candidate_verification']['sampled_cursors']) for c in oracle['contracts'])
        lines = [design.rstrip(), '', '## 实际全TRAIN验收', '',
                 f'签名O收据：`{directory}/manifest.json`；content hash：`{oracle["content_hash"]}`。', '',
                 f'18合约、{total}个候选；{samples}个分层候选另行完整参考Env重放。72条正式H333账户完整原Driver采集及逐行reconcile通过；每笔最终交易、每步反馈、实际净现金增量与精确计划相同。γ1为精确优化诊断，未声明全部γ1动作表另行完整Driver重放。', '',
                 *table(['contract', 'candidates', 'reference samples', 'required equity upper bound', 'seconds', 'peak RSS bytes'], rows), '',
                 *table(['contract', 'candidate', 'H333 trades', 'H333 certified objective', 'replayed float64 diagnostic', 'net cash gain', 'gamma1 trades', 'gamma1 objective'], plans), '',
                 '完整候选表、最优性/资金证书、动作表、抽样游标、实际账本、Replay/Sidecar和执行源码归档均留在运行目录；其字节哈希在签名收据。抽样参考路径没有覆盖每个候选；有限TRAIN最优后验策略不是可部署模型，未用于反选Reward。']
        publish(oracle_doc, '\n'.join(lines) + '\n')
    copies = [('D coverage', 'replay_coverage.md', 'FG_Single3_Replay_R0_R2_Coverage.md'),
              ('H', 'horizon_analysis.md', 'FG_Single3_Horizon_Credit_Assignment.md'),
              ('Pilot preparation', 'pilot_comparison.md', 'FG_Single3_Model_Pilot_Comparison.md')]
    for stage, source, target in copies:
        if stage in found:
            directory, receipt = found[stage]
            publish(docs / target, checked_report(directory, receipt, source))
    if 'D coverage' in found:
        directory, receipt = found['D coverage']
        coverage = checked_report(directory, receipt, 'replay_coverage.md')
        reward = docs / 'FG_Single3_Reward_Research.md'
        marker = '\n## 全TRAIN候选比较（D完整账户实测）\n'
        before = reward.read_text(encoding='utf-8').split(marker)[0]
        section = coverage.split('## 全Fold Reward与真实账户比较\n', 1)[1].split('\n## 年份/合约/方向/日夜盘', 1)[0]
        publish(reward, before.rstrip() + marker + '\n' + section + '\n\n'
                + 'early10条件概率、MAE、TP时长及年份/合约/方向/日夜盘完整表见 `FG_Single3_Replay_R0_R2_Coverage.md`；训练Reward尚未选定。\n')
    rows = [(name, 'PASS' if name in found else 'NOT_COMPLETED',
             found[name][1]['content_hash'] if name in found else '-') for name in stages]
    lines = ['# FG Single3 CPU 实施与验收汇总', '',
             '状态：' + ('CPU阶段验收及模型先导准备完成。' if complete else 'IN_PROGRESS，以下缺失Gate仍未完成。'),
             '模型NOT_RUN；gradient_updates=0；择时能力NOT_ESTABLISHED。', '',
             f'报告生成UTC：{datetime.now(timezone.utc).isoformat()}。', '',
             '正式依据：`FG_DreamerV3_Single3_TP5_SL30_Reward_OracleReplay_Execution_Plan_V1.md`；原文件SHA256 `9e3287a5bbe9eec93788275ae5b3ee082677c637b266690bad08732823679744`。业务协议和实施范围见 `FG_Single3_Implementation_Plan_and_Progress.md`。', '',
             *table(['stage', 'actual status', 'signed receipt content hash'], rows), '',
             '输入源为冻结18合约TRAIN白名单；保留旧6手Replay、Checkpoint和实验结果。原账户/撮合、Driver/Replay、Uniform采样和Dreamer网络继续复用。所有新价格轨迹来自Single3真实Env；同一自然策略四Reward视图冻结动作后分别执行Env，不把旧6手反馈重新标记为新数据。', '',
             '实际确认并修复P3跨CHUNK_END的等待计时重置；原Reward R01收据撤销为失败证据，R02重新采集验收。最终验证失败时正式完成收据/PASS报告不发布；失败资料保留。', '', '## 实际测试及命令证据', '']
    for name, (directory, receipt) in found.items():
        tests = receipt.get('tests')
        if name == 'D generation':
            tests = read_manifest(directory / 'inputs.json').get('tests')
        if tests:
            lines += [f'- {name}：`{" ".join(tests["command"])}`；returncode={tests["returncode"]}；输出 `{tests["output"].strip()}`。']
    lines += ['', '## 真实产物与范围', '']
    if 'Input' in found:
        input_receipt = found['Input'][1]
        cuts = sum(len(c['exact_float32_prefix_stops']) for c in input_receipt['contracts'])
        lines += [f'18合约{cuts}个实际截点，60×18观察与当时行情前缀精确float32相同；随机seed202、日边界及chunk边界单列。这是抽样因果证据，未来Oracle标签只在Sidecar，不在模型张量。', '']
    if 'D generation' in found:
        data = found['D generation'][1]
        lines += [f'完整新账户{len(data["traces"])}条；{len(data["views"])}个单Reward Replay池；P0独立。原生chunk、执行源码、输入及逐行对账身份见 `{found["D generation"][0]}/manifest.json` 和 `inputs.json`。', '']
    if 'D coverage' in found:
        coverage = found['D coverage'][1]
        lines += [f'实际Replay行{coverage["totals"]["records"]}、推进{coverage["totals"]["advanced"]}；访问唯一市场分钟{coverage["unique_advanced_minutes"]}；{len(coverage["pools"])}池各{coverage["batches"]}真实B16×65批次。非法ADD={coverage["totals"]["illegal_add"]}、6手决策={coverage["totals"]["lots6"]}。详见 `FG_Single3_Replay_R0_R2_Coverage.md`。', '']
    if 'H' in found:
        lines += [f'H实际{found["H"][1]["pairs"]}个公平同账户快照ADD/WAIT分叉；H333/1000/3000/γ1比较见 `FG_Single3_Horizon_Credit_Assignment.md`。这是冻结后续P1的有限反事实，Critic预测误差NOT_MEASURED。', '']
    if 'Pilot preparation' in found:
        pilot = found['Pilot preparation'][1]
        if pilot['model_status'] != 'NOT_RUN' or pilot['gradient_updates'] != 0:
            raise ValueError('Unexpected model execution in CPU report')
        lines += [f'{len(pilot["configurations"])}份M0/M1模板已CPU预检；authorized=False，在Agent初始化前阻止启动。现有Mask warmup/ADD召回门控与完整Checkpoint恢复通过FakeAgent接线测试；实际GPU网络权重与训练能力未验证。', '']
    lines += ['Reward候选未选定，不按P4高分选参数；保持horizon333/imag15/原网络作为第一组基线。当前工程证据不证明Agent学会WAIT/ADD或快速TP。', '',
              '用户确认Reward、完整Replay覆盖和Oracle验收后，才可授权一组Reward的M0/M1同预算先导。先导结果仍需独立评估；本次没有访问验证/封存Test价格。']
    publish(docs / 'FG_Single3_CPU_Implementation_and_Verification.md', '\n'.join(lines) + '\n')
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/fg_single3_v1'))
    parser.add_argument('--docs', type=Path, default=Path('docs'))
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--wait', action='store_true', help='Wait for the signed pilot Gate, then publish complete CPU reports')
    parser.add_argument('--pipeline-log', type=Path, help='Stop waiting when the CPU pipeline records a failed command')
    for name, (relative, _) in STAGES.items():
        parser.add_argument('--' + name.lower().replace(' ', '-'), default=relative)
    args = parser.parse_args()
    stages = {name: (getattr(args, name.lower().replace(' ', '_')), expected)
              for name, (_, expected) in STAGES.items()}
    if args.wait:
        print('WAITING_FOR_CPU_REPORT_GATES', flush=True)
        gate = args.root / stages['Pilot preparation'][0] / 'manifest.json'
        while not gate.exists():
            if args.pipeline_log and args.pipeline_log.exists():
                for line in args.pipeline_log.read_text().splitlines():
                    if not line.startswith('{'):
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get('status') == 'FAIL' and event.get('returncode'):
                        raise RuntimeError('CPU pipeline failed; completion report refused')
            time.sleep(30)
    complete = collate(args.root, args.docs, require_complete=args.require_complete or args.wait, stages=stages)
    print('CPU_REPORTS_COMPLETE_GPU_NOT_RUN' if complete else 'CPU_REPORTS_IN_PROGRESS')


if __name__ == '__main__':
    main()

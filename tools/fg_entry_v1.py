#!/usr/bin/env python3
"""Explicit FG-only DB export, contract construction and baseline evaluation."""
import argparse
import json
from pathlib import Path
from gym_trading_env.envs.entry_contract import read_yaml
from gym_trading_env.research.fg_entry_v1 import export_fg, build_contract, run_policy


def main():
    parser=argparse.ArgumentParser(__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    inspect=sub.add_parser("inspect-db",help="Read-only DB schema and FG coverage inspection")
    inspect.add_argument("--output",help="Save complete inspection JSON; terminal shows a compact summary")
    exp=sub.add_parser("export")
    exp.add_argument("--provider")
    exp.add_argument("--config",default="configs/fg_entry_v1/fg.yaml")
    exp.add_argument("--start");exp.add_argument("--end")
    exp.add_argument("--output",required=True)
    build=sub.add_parser("build")
    build.add_argument("--export-manifest",required=True);build.add_argument("--config",default="configs/fg_entry_v1/fg.yaml")

    build.add_argument("--output",required=True)
    base=sub.add_parser("baselines")
    base.add_argument("--config",required=True);base.add_argument("--output",required=True)
    base.add_argument("--seed",type=int,default=0)
    args=parser.parse_args()
    if args.command in ("inspect-db","export"):
        from gym_trading_env.utils.juejin_db import JuejinDBClient
        from gym_trading_env.config.settings import DB_CONFIG
        import pymysql
        client=JuejinDBClient.from_settings()
        if args.command=="inspect-db":
            with pymysql.connect(host=DB_CONFIG.host,port=DB_CONFIG.port,user=DB_CONFIG.user,
                    password=DB_CONFIG.password,database=DB_CONFIG.db,connect_timeout=5,
                    read_timeout=30,cursorclass=pymysql.cursors.DictCursor) as conn:
                with conn.cursor() as cur:
                    cur.execute("SHOW TABLES LIKE 'fut%'")
                    tables=cur.fetchall()
                    cur.execute("SHOW COLUMNS FROM fut_bar_1m_v2")
                    columns=[r["Field"] for r in cur.fetchall()]
                    cur.execute("SELECT symbol, MIN(eob) first_eob, MAX(eob) last_eob, COUNT(*) rows_count FROM fut_bar_1m_v2 WHERE symbol LIKE %s GROUP BY symbol",("CZCE.FG%",))
                    coverage=cur.fetchall()
                    metadata={}
                    for table in ("fut_contract","fut_contract_v2","fut_underlying_v2"):
                        cur.execute(f"SHOW COLUMNS FROM {table}")
                        fields=[r["Field"] for r in cur.fetchall()]
                        metadata[table]=dict(columns=fields)
                        if "symbol" in fields:
                            cur.execute(f"SELECT * FROM {table} WHERE symbol=%s LIMIT 1",("CZCE.FG701",))
                            metadata[table]["sample_fg701"]=cur.fetchall()
                    result=dict(tables=tables,columns=columns,fg_coverage=coverage,metadata=metadata)
                    if args.output:
                        from gym_trading_env.research.fg_entry_v1 import write_json
                        write_json(args.output,result)
                    print(json.dumps(dict(metadata=metadata,fg_symbols=len(coverage),
                        selected=[r for r in coverage if r["symbol"] in ("CZCE.FG609","CZCE.FG701")]),default=str,indent=2))
            return
        import pandas as pd
        master=read_yaml(args.config)
        start=args.start or master['data']['start']
        end=args.end or str(pd.Timestamp(master['data']['end_inclusive'])+pd.Timedelta(days=1))
        result=export_fg(client,start,end,args.output,args.provider or master["data"]["provider"],
            excluded_contracts=master['data']['excluded_contracts'])
    elif args.command=="build":
        from gym_trading_env.research.fg_entry_v1 import prepare_main_contracts
        master=read_yaml(args.config)
        manifest=json.loads(Path(args.export_manifest).read_text())
        contract_paths=prepare_main_contracts(manifest,master['costs'],Path(args.output)/'contracts')
        result=build_contract(manifest,args.config,contract_paths,args.config,args.output)
    else:
        result={strategy:run_policy(args.config,Path(args.output)/strategy,strategy=strategy,seed=args.seed)
                for strategy in ("always_add","random_add","rule_add")}
    print(json.dumps(result,default=str,indent=2))


if __name__=="__main__":
    main()

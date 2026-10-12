"""Only AST/YAML/report validation; never executes report training commands."""
import ast
import json
from pathlib import Path
import re
import subprocess
import yaml

GYM=Path(__file__).resolve().parents[3]; DREAMER=Path('/home/v/Documents/work/dreamerv3-dev')
OUT=Path(__file__).parent
python_files=[]
for root in (GYM,DREAMER):
    names=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root,text=True).splitlines()
    for name in names:
        if name.endswith('.py'):
            ast.parse((root/name).read_text(),filename=str(root/name)); python_files.append(str(root/name))
configs=[]
for root,folder in [(GYM,'configs/experiments/fg_s3b'),(DREAMER,'configs/fg_s3b')]:
    for p in (root/folder).glob('*.yaml'):
        yaml.safe_load(p.read_text()); configs.append(str(p))
report=GYM/'docs/FG_S3B_Implementation_and_Verification.md'; body=report.read_text()
for term in ('B0','B1','B2','B3','B4','NOT_RUN','NOT_ACCESSED','16','55','396,542','DRIFT_OBSERVED'):
    assert term in body,term
blocks=re.findall(r'```bash\n(.*?)```',body,re.S)
for block in blocks:
    result=subprocess.run(['bash','-n'],input=block,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
receipts=json.loads((OUT/'B4_delivery_audit.json').read_text())
assert receipts['unique_new_tests']==16 and receipts['actual_account_forks']==55
assert receipts['development_existing_source_drift']==[]
result=dict(status='PASS',python_ast_files=python_files,yaml_configs=configs,
            bash_syntax_blocks=len(blocks),formal_commands_executed=False,
            receipt_status=receipts['status'],report_scope='code/CPU; future GPU only')
(OUT/'B4_document_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status='PASS',python_ast_files=len(python_files),yaml_configs=len(configs),bash_blocks=len(blocks))))

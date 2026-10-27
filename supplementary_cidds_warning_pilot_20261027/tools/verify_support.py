from pathlib import Path
import sys,json,hashlib,importlib.util
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'SHA256SUMS.json').read_text())
for name,expected in manifest.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, name
spec=importlib.util.spec_from_file_location('s20_verifier',ROOT/'code/verify_report.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
report=ROOT/'archives/cidds_warning_pilot_report_20261027_104430_393215.zip'
print(json.dumps(module.verify(report),indent=2))
print('Support manifest PASS:',len(manifest),'files')

from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
paths=['tools/round2_v4_s5_orchestrator.py','tools/shape_contract_search.py','tools/deliver_coverage102_pilot_p3a_p3b_20260910.py','tools/render_finite_summary_20260909.py','tools/freeze_coverage102_pilot_budget_20260910.py','tools/run_shape_guided_budget_search_20260913.py','tools/audit_old_balance_shapes_20260908.py','src/runtime_bank_calendar.py','src/runtime_daily_aggregation.py','src/runtime_transaction_normalizer.py','src/runtime_field_mapping.py','configs','assets/first_round_published_model','assets/templates']
rows=[]
for item in paths:
 p=ROOT/item
 for f in ([p] if p.is_file() else sorted(p.rglob('*'))):
  if f.is_file():rows.append({'path':str(f.relative_to(ROOT)).replace('\\','/'),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size})
(ROOT/'data/manifests/资产复制清单.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print(len(rows))

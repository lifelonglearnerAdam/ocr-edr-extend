"""Offline replay/metric recomputation, appended after original generation receipts."""
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

project = Path.cwd()
sys.path.insert(0, str(project / 'src'))
from ocr_edr.native_table_repair import native_repair_inputs, verify_native_calls
from ocr_edr.native_tables import load_native_table_sources
from ocr_edr.dashboard_evidence import verify_native_coverage
from ocr_edr.official_tables import load_official_table_normalizer, load_official_teds, verify_official_table_sources
from ocr_edr.sft import sha256, verify_model_files
from ocr_edr.table_evaluation import evaluate_tables
from ocr_edr.table_sft_screen import table_messages

root = project / 'experiments/runs/native-table-sft-20261009'
dataset = project / 'data/processed/pubtabnet-four-roles-20261007'
training = project / 'experiments/runs/table-nf4-screen-recovery-20261009/training'
read = lambda p: json.loads(p.read_text())
rows = lambda p: [json.loads(line) for line in p.read_text().splitlines()]
pipeline = read(root / 'pipeline-status.json')
seals = {r['stage']: r['receipt_sha256'] for r in pipeline['completed_stages']}
assert len(seals) == 4 and pipeline['status'] == 'completed'
frozen = root / 'frozen-native-inputs'
assert sha256(frozen / 'predictions.jsonl') == '6bfe68cd225e3cac1f645849174209725936367b31fe8a91bf7a248c93ccd2f3'
inputs = native_repair_inputs(rows(frozen / 'predictions.jsonl'), load_native_table_sources(dataset, role='model_dev'), role='model_dev')
observed_hashes = {}
actions = {}
for arm in ['base', 'all', 'no_explicit_preservation']:
 folder = root / 'inference' / arm
 receipt = read(folder / 'run.json')
 assert sha256(folder / 'run.json') == seals['inference_' + arm]
 assert receipt['status'] == 'completed' and receipt['completed_calls'] == 32 and receipt['reference_access'] == 'none'
 assert sha256(folder / 'calls.jsonl') == receipt['calls_sha256']
 calls = rows(folder / 'calls.jsonl')
 verify_native_calls(calls, inputs, arm)
 for call in calls:
  messages, prompt = table_messages(call['prediction'], prompt_format='descriptive_schema')
  assert call['prompt'] == prompt and call['messages'] == messages
 for filename, digest in receipt['source_sha256'].items():
  assert sha256(folder / 'source' / filename) == digest
 if arm != 'base':
  trained = read(training / arm / 'run.json')
  assert receipt['adapter_sha256'] == trained['checkpoint_sha256']
  for filename, digest in receipt['adapter_sha256'].items():
   assert sha256(training / arm / 'checkpoint' / filename) == digest
 observed_hashes[arm + '_receipt'] = sha256(folder / 'run.json')
 observed_hashes[arm + '_calls'] = sha256(folder / 'calls.jsonl')
 assert receipt['finished_at'] < read(root / 'evaluation/run.json')['started_at']
# Original generator did not record the base receipt hash. Verify the controller's
# explicit path and current model files; do not pretend this was a historical seal.
model_receipt_path = project / 'experiments/runs/server-selection-20261006/model-receipt.json'
model_receipt = read(model_receipt_path)
model_dir = project / ('data/raw/model-cache/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/' + model_receipt['revision'])
assert model_receipt['revision'] == '895c3a49bc3fa70a340399125c650a463535e71c'
assert str(model_dir.relative_to(project)) in (root / 'continue_pipeline.py').read_text()
verify_model_files(model_dir, model_receipt)
evaluation = root / 'evaluation'
receipt = read(evaluation / 'run.json')
assert sha256(evaluation / 'run.json') == seals['evaluation']
for filename, key in [('evaluation.json','evaluation_sha256'), ('predictions.jsonl','predictions_sha256')]:
 assert sha256(evaluation / filename) == receipt[key]
 observed_hashes[filename] = sha256(evaluation / filename)
report = read(evaluation / 'evaluation.json')
verify_native_coverage(report)
predictions = rows(evaluation / 'predictions.jsonl')
for arm in ['base','all','no_explicit_preservation']:
 calls = {r['sample_id']: r for r in rows(root / 'inference' / arm / 'calls.jsonl')}
 for prediction in [p for p in predictions if p['arm'] == arm]:
  trace = prediction['trace']
  assert len(trace) == 1
  assert all(trace[0].get(k) == v for k,v in calls[prediction['sample_id']].items())
 refs = []
for source in rows(dataset / 'model_dev-references.jsonl'):
 assert source['role'] == 'model_dev'
meta = read(dataset / 'dataset.json')
assert sha256(dataset / 'model_dev-references.jsonl') == meta['file_sha256']['model_dev-references.jsonl']
by_family = {}
for source in rows(dataset / 'model_dev-references.jsonl'):
 identity = (source['document_id'],source['reference'])
 assert by_family.setdefault(source['family_id'],identity) == identity
for item in inputs:
 refs.append({'sample_id':item['sample_id'],'family_id':item['family_id'],
 'parent_page':by_family[item['family_id']][0],'reference':by_family[item['family_id']][1],
 'variant':'native','annotation_id':None,'gt_position':None})
official = project / 'data/raw/omnidocbench-source'
verify_official_table_sources(official, 'f133a71e9e91c3621c7ce8994200a7b394a06eb3')
normalize = load_official_table_normalizer(official)
metric = load_official_teds(official)
teds, structure = metric(), metric(structure_only=True)
recomputed = evaluate_tables(predictions, inputs, refs, ['unchanged_0','base','all','no_explicit_preservation'], normalize=normalize,
 score=lambda p,r:{'teds':teds.evaluate(p,r),'teds_structure':structure.evaluate(p,r)})
assert recomputed == report
numerical_cases = []
indexed = {(r['sample_id'],r['arm']):r for r in predictions}
for r in report['cases']:
 p = indexed[r['sample_id'],r['arm']]
 trace = p['trace']
 action = (trace[0].get('action') or {}) if trace else {}
 if 'text' in action:
  action = {k:v for k,v in action.items() if k != 'text'} | {'text_sha256':hashlib.sha256(trace[0]['action']['text'].encode()).hexdigest()}
 numerical_cases.append({k:v for k,v in r.items() if 'html' not in k} | {
 'action':action, 'input_sha256':hashlib.sha256(p['initial_prediction'].encode()).hexdigest(),
 'output_sha256':hashlib.sha256(p['final_prediction'].encode()).hexdigest(),
 'raw_output_sha256':hashlib.sha256(trace[0]['raw_output'].encode()).hexdigest() if trace else None})
for arm in ['base','all','no_explicit_preservation']:
 group = [r for r in numerical_cases if r['arm'] == arm]
 actions[arm] = dict(Counter(r['action'].get('action','rejected') for r in group))
summary = [r for r in report['summary'] if r['variant'] == 'all']
verification = {'status':'audit_completed','verified_at':datetime.now(timezone.utc).isoformat(),
 'complete_call_source_prompt_and_checkpoint_checks':True,'full_metric_recomputation_exact':True,
 'original_receipts_preserved':True,'historical_base_receipt_hash_missing':True,
 'current_model_files_and_controller_path_verified':True,'current_model_receipt_sha256':sha256(model_receipt_path),
 'evaluation_sha256':sha256(evaluation/'evaluation.json'),'evidence_sha256':observed_hashes,
 'calls':96,'case_arm_results':128,'cases':32,'documents':32,'calibration_locked_access':False,
 'action_counts':actions,'summary':summary,'audit_script_sha256':sha256(Path(__file__)),
 'limits':['post-hoc model-dev diagnostic, not independent visual gold','same previously inspected public PubTabNet articles; pretraining overlap possible','current base hash verification cannot recover an absent historical receipt binding']}
(root / 'audit.json').write_text(json.dumps(verification,indent=2)+'\n')
(root / 'numerical-cases.json').write_text(json.dumps(numerical_cases,indent=2)+'\n')
print(json.dumps({'verification':verification['status'],'calls':96,'all_metrics_recomputed_exact':True,'actions':actions},indent=2))

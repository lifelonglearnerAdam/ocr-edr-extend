"""Replay the32 already-frozen all calls after an evidence projection bug; no model."""
import json,sys,hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
from PIL import Image
project=Path.cwd();sys.path.insert(0,str(project/'src'))
from ocr_edr.sft import sha256
from ocr_edr.native_table_repair import blank_table_observation,native_repair_inputs,verify_native_calls
from ocr_edr.native_tables import load_native_table_sources
from ocr_edr.table_sft_screen import adapt_table_call
from ocr_edr.table_pilot import HTMLTableRenderer
from ocr_edr.official_tables import load_official_table_normalizer,verify_official_table_sources
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(l) for l in p.read_text().splitlines()]
failed=project/'experiments/runs/table-source-evidence-20261009'
native=project/'experiments/runs/native-table-sft-20261009'
audit=read(native/'audit.json');old=read(failed/'run.json')
assert old['status']=='failed' and old['completed_calls']=={'all':32}
assert old['script_sha256']==sha256(failed/'execution.py')
assert old['native_audit_sha256']==sha256(native/'audit.json')
assert old['protocol_sha256']==sha256(project/'docs/research/TABLE_SOURCE_EVIDENCE_PROTOCOL_20261009.md')
assert old['model_receipt_sha256']==sha256(project/'experiments/runs/server-selection-20261006/model-receipt.json')
assert sha256(native/'evaluation/predictions.jsonl')==audit['evidence_sha256']['predictions.jsonl']
assert sha256(native/'inference/all/calls.jsonl')==audit['evidence_sha256']['all_calls']
inputs=native_repair_inputs(rows(native/'frozen-native-inputs/predictions.jsonl'),load_native_table_sources(project/'data/processed/pubtabnet-four-roles-20261007',role='model_dev'),role='model_dev')
original_calls=rows(native/'inference/all/calls.jsonl');verify_native_calls(original_calls,inputs,'all')
generated=rows(failed/'all/calls.jsonl');assert len(generated)==32 and len({r['sample_id'] for r in generated})==32
native_receipt=read(native/'inference/all/run.json')
assert old['adapter_sha256']['all']==native_receipt['adapter_sha256']
assert old['original_call_sha256']['all']==audit['evidence_sha256']['all_calls']
original={r['sample_id']:r for r in rows(native/'evaluation/predictions.jsonl') if r['arm']=='all'}
official=project/'data/raw/omnidocbench-source';verify_official_table_sources(official,'f133a71e9e91c3621c7ce8994200a7b394a06eb3')
normalize=load_official_table_normalizer(official)
renderer=HTMLTableRenderer(failed/'recovery-renders')
paired=[]
for item,call,real_call in zip(inputs,generated,original_calls):
 blank=failed/'blank'/(item['family_id']+'.png')
 assert call['condition']=='all' and call['ordered_image_sha256']==[sha256(blank)]
 assert all(call[k]==real_call[k] for k in ['prompt','messages','image_grid_thw','input_tokens'])
 with Image.open(blank) as image,Image.open(project/'data/processed/pubtabnet-four-roles-20261007'/item['source_image']) as source:
  assert image.size==source.size and image.getextrema()==((255,255),)*3
 observation,execution=blank_table_observation(item,call)
 result=adapt_table_call(observation,execution,renderer=renderer.render,max_new_tokens=192,prompt_format='descriptive_schema')
 trace=result['trace'][0];real=original[item['sample_id']]
 paired.append({'sample_id':item['sample_id'],'arm':'all','raw_output_equal':call['raw_output']==real_call['raw_output'],
 'action_equal':trace.get('action')==real['trace'][0].get('action'),
 'final_html_equal':result['final_prediction']==real['final_prediction'],
 'normalized_html_equal':normalize(result['final_prediction'])==normalize(real['final_prediction']),
 'blank_action':{k:v for k,v in (trace.get('action') or {}).items() if k!='text'},
 'real_action':{k:v for k,v in (real['trace'][0].get('action') or {}).items() if k!='text'},
 'blank_contract':trace['contract'],'hit_length_cap':trace['hit_length_cap'],
 'input_tokens':call['input_tokens'],'output_tokens':call['output_tokens'],'generation_seconds':call['generation_seconds'],
 'original_image_sha256':item['source_sha256'],'blank_image_sha256':call['intervention_image_sha256']})
summary={'arm':'all','cases':32,**{k:sum(r[k] for r in paired) for k in ['raw_output_equal','action_equal','final_html_equal','normalized_html_equal','hit_length_cap','input_tokens','output_tokens','generation_seconds']},
 'blank_actions':dict(Counter(r['blank_action'].get('action','rejected') for r in paired)),
 'contracts':dict(Counter(r['blank_contract'] for r in paired)),'calls_sha256':sha256(failed/'all/calls.jsonl')}
comparison={'summary':[summary],'cases':paired}
(failed/'replayed-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
receipt={'status':'comparison_recovered','verified_at':datetime.now(timezone.utc).isoformat(),'cases':32,'new_model_calls':0,
 'failed_receipt_sha256':sha256(failed/'run.json'),'original_script_sha256':old['script_sha256'],
 'saved_calls_sha256':sha256(failed/'all/calls.jsonl'),'comparison_sha256':sha256(failed/'replayed-comparison.json'),
 'recovery_script_sha256':sha256(Path(__file__)),'source_projection_sha256':sha256(project/'src/ocr_edr/native_table_repair.py'),
 'original_receipts_and_calls_modified':False,'native_predictions_sha256':sha256(native/'evaluation/predictions.jsonl')}
(failed/'recovery.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(summary,indent=2))

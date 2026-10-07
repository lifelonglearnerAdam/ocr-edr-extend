from pathlib import Path
import json,sys,hashlib
import numpy as np
from PIL import Image
from lxml import etree,html
repo=Path(__file__).resolve().parents[3];sys.path.insert(0,str(repo/'src'))
from ocr_edr.official_tables import load_official_table_normalizer
from ocr_edr.table_pilot import parse_table
root=Path(__file__).parent
old=[json.loads(l) for l in (root/'inference/predictions.jsonl').read_text().splitlines()]
new=[json.loads(l) for l in (root/'compat-inference/predictions.jsonl').read_text().splitlines()]
refs={r['family_id']:r for r in map(json.loads,(repo/'data/processed/pubtabnet-four-roles-20261007/model_dev-references.jsonl').read_text().splitlines())}
normalize=load_official_table_normalizer((repo/'data/raw/omnidocbench-source').resolve())
def skeleton(markup):
    tree,_=parse_table(markup)
    for cell in tree.xpath('.//td|.//th'):
        for child in list(cell):cell.remove(child)
        cell.text=''
    return etree.tostring(tree)
comparisons=[]
for a,b in zip(old,new):
    assert a['family_id']==b['family_id'] and a['source_sha256']==b['source_sha256']
    with Image.open(repo/'data/processed/pubtabnet-four-roles-20261007'/a['source_image']) as image: w,h=image.size
    x=np.array(a['details']['cell_bbox']);y=np.array(b['details']['cell_bbox'])
    scaled=x*np.array([w/max(w,h),h/max(w,h),w/max(w,h),h/max(w,h)])
    comparisons.append({'family_id':a['family_id'],'raw_dom_skeleton_equal':skeleton(a['prediction'])==skeleton(b['prediction']),
        'ocr_text_confidence_equal':a['details']['ocr_recognition']==b['details']['ocr_recognition'],
        'ocr_boxes_equal':a['details']['ocr_boxes']==b['details']['ocr_boxes'],
        'expected_coordinate_rescale_equal':x.shape==y.shape and bool(np.allclose(scaled,y,rtol=1e-5,atol=1e-4)),
        'max_coordinate_rescale_error':float(np.max(np.abs(scaled-y))) if x.shape==y.shape else None})
family='p0141';a=next(r for r in old if r['family_id']==family);b=next(r for r in new if r['family_id']==family)
metrics=[]
for label,rows in [('original',old),('compat',new)]:
    row=next(r for r in rows if r['family_id']==family)
    norm=normalize(row['prediction']);ref=normalize(refs[family]['reference'])
    pred_table=html.fromstring(norm).xpath('body/table')[0];ref_table=html.fromstring(ref).xpath('body/table')[0]
    evaluation=json.loads((root/('evaluation' if label=='original' else 'compat-evaluation')/'evaluation.json').read_text())
    score=next(r for r in evaluation['cases'] if r['family_id']==family)['teds_structure']
    denominator=max(len(pred_table.xpath('.//*')),len(ref_table.xpath('.//*')))
    metrics.append({'condition':label,'normalized_descendant_denominator':denominator,'teds_structure':score,
        'distance_recovered_from_score':(1-score)*denominator})
result={'sources':len(comparisons),'comparison':comparisons,
    'p0141_normalized_td_skeleton_equal':skeleton(normalize(a['prediction']))==skeleton(normalize(b['prediction'])),
    'p0141_official_teds_s_denominator':metrics,
    'escaped_format_marker_sources':sum(any('&lt;'+tag in row['prediction'] for tag in ['b','i','sub','sup']) for row in new),
    'meaning':'decoded raw structure and OCR output equivalence; no claim of captured model-logit bit equality',
    'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root/'coordinate-condition-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='comparison'},indent=2))

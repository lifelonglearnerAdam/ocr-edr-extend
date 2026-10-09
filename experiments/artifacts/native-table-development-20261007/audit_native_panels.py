from pathlib import Path
import json,sys,subprocess,hashlib
from PIL import Image,ImageOps,ImageDraw,ImageFont
repo=Path(__file__).resolve().parents[3];sys.path.insert(0,str(repo/'src'))
from ocr_edr.loop import Observation
from ocr_edr.table_pilot import HTMLTableRenderer
root=Path(__file__).parent;out=root/'audit-panels';out.mkdir(exist_ok=False)
rows={r['family_id']:r for r in map(json.loads,(root/'compat-inference/predictions.jsonl').read_text().splitlines())}
refs={r['family_id']:r for r in map(json.loads,(repo/'data/processed/pubtabnet-four-roles-20261007/model_dev-references.jsonl').read_text().splitlines())}
renderer=HTMLTableRenderer(out/'renders');font=ImageFont.truetype(subprocess.check_output(['fc-match','-f','%{file}','sans-serif'],text=True),22)
selected=['p0130','p0135','p0141','p0143']
(out/'selection.json').write_text(json.dumps({'selection':'metric full-match escaping example; two structure mismatches; one raw-text-match but nonfull TEDS case','families':selected,'scope':'diagnostic assistant audit, not random sample or human gold labels','reference_changes':False},indent=2)+'\n')
for family in selected:
    row=rows[family]
    paths=[repo/'data/processed/pubtabnet-four-roles-20261007'/row['source_image'],Path(renderer.render(Observation(family,'table','',row['prediction'])).path),Path(renderer.render(Observation(family,'table','',refs[family]['reference'])).path)]
    images=[]
    for p in paths:
        with Image.open(p) as im:images.append(ImageOps.contain(im.convert('RGB'),(920,900)))
    sheet=Image.new('RGB',(2880,max(im.height for im in images)+110),'white');d=ImageDraw.Draw(sheet)
    d.text((10,5),family+' / source vs raw parser HTML vs published reference',font=font,fill='black')
    for i,(title,im) in enumerate(zip(['SOURCE','RAW PARSER (before metric normalization)','REFERENCE (offline)'],images)):
        d.text((i*960+10,42),title,font=font,fill='black');sheet.paste(im,(i*960+10,90))
    sheet.save(out/(family+'.png'))
print('saved',selected)

"""Offline, self-contained review page and strict import of explicit human decisions."""
from pathlib import Path
import json
import numpy as np
from .common import (read, write, manifest, load_xyz, annotation, blank_annotation,
                     motion_hash, digest, now, DEFECTS, JOINTS, CHAINS, FEET)
from .measure import measure, mask_intervals, rule_feedback

def review_items(work):
    work = Path(work)
    items = {}
    for row in manifest(work):
        for phase in ('raw', 'initial'):
            if f'{phase}_xyz' in row['paths']:
                key = row['clip_id'] + '/' + phase
                items[key] = dict(path=row['paths'][f'{phase}_xyz'], caption=row['caption'], fps=row['fps'],
                                  clip_id=row['clip_id'], phase=phase,
                                  duplicate_generated_raw=phase == 'raw' and row['origin'] == 'generated'
                                  and row.get('raw_hash') == row.get('initial_hash'))
    for path in sorted((work / 'runs').glob('*/cases/*/*/round_*/result.json')):
        result = read(path)
        if result.get('status') == 'complete':
            items[result['output_key']] = dict(path=result['output_xyz'], caption=result['caption'],
                fps=result['fps'], clip_id=result['clip_id'], phase='after',
                before=result['input_xyz'], before_key=result['input_key'], result_path=str(path.relative_to(work)))
    return items

def validate_annotation(entry, xyz):
    if entry['motion_hash'] != motion_hash(xyz) or entry['frames'] != len(xyz):
        raise ValueError('Stale annotation or changed clip; export a fresh review')
    for field in ('reviewed', 'units_confirmed', 'floor_verified', 'support_reviewed', 'labels_complete'):
        if not isinstance(entry.get(field), bool):
            raise ValueError(f'{field} must be a JSON boolean')
    if entry['reviewed'] and not entry.get('reviewer', '').strip():
        raise ValueError('Reviewed annotations require a reviewer name/ID')
    for field in ('support', 'required_support', 'airborne', 'labels'):
        mask_intervals(entry[field], len(xyz))
    for event in entry['labels'] + entry['feedback']['defects']:
        if event['defect'] not in DEFECTS or event.get('body_part') not in [*JOINTS, *FEET, 'both_feet']:
            raise ValueError('Defects need a supported defect type and body part')
        mask_intervals([event], len(xyz))
    feedback = entry['feedback']
    if not isinstance(feedback.get('approved'), bool) or feedback['decision'] not in ('defect', 'no_defect', 'uncertain'):
        raise ValueError('Feedback requires boolean approved and a valid decision')
    if feedback['approved']:
        if not entry['reviewed']:
            raise ValueError('Approved feedback requires an explicit reviewed annotation')
        if feedback['decision'] == 'defect' and (not feedback['defects'] or not feedback['instruction'].strip()):
            raise ValueError('Defect feedback requires reviewed locations and an instruction')
        if feedback['decision'] == 'no_defect' and feedback['defects']:
            raise ValueError('No-defect feedback cannot also list defects')

def export_review(work, config, output, mode='clips', run_id=None):
    work, output = Path(work), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    entries, cards, mapping = [], [], {}
    for key, item in review_items(work).items():
        if (mode == 'pairs') != (item['phase'] == 'after'):
            continue
        if item.get('duplicate_generated_raw'):
            continue  # Keep both measurements, but do not ask for the same generated-clip review twice.
        if run_id and item['phase'] == 'after' and not key.startswith(run_id + '/'):
            continue
        xyz = load_xyz(work / item['path'])
        ann = annotation(work, key) or blank_annotation(key, xyz, item['fps'], item['caption'])
        # Re-exporting never overwrites accepted annotations; templates are copies.
        ann = json.loads(json.dumps(ann))
        metric = measure(xyz, item['fps'], ann, config['thresholds'])
        if mode == 'pairs':
            # Stable anonymous IDs hide condition, run paths, feedback and defect proposals.
            code = digest([config['seed'], key])[:12]
            mapping[code] = key
            ann['key'] = code
            ann['feedback'] = {'approved': False, 'decision': 'uncertain', 'instruction': '', 'defects': []}
            ann.setdefault('outcome', dict(reviewed=False, target_resolved=None, new_defect=None,
                           action_preserved=None, unnecessary_edit=None, notes=''))
            before = load_xyz(work / item['before']).round(5).tolist()
            proposals = None
        else:
            code, before = key, None
            proposals = rule_feedback(metric)
        entries.append(ann)
        cards.append(dict(code=code, caption=item['caption'], fps=item['fps'], xyz=xyz.round(5).tolist(),
                          before=before, floor=ann['floor_m'], proposals=proposals,
                          target_labels=annotation(work, item['clip_id'] + '/initial').get('labels', []) if mode == 'pairs' else None))
    if mode == 'pairs':
        # Keep this file outside the review directory; reviewer need not see the condition mapping.
        write(work / 'private' / f'{output.name}_mapping.json', mapping)
        order = np.random.default_rng(config['seed']).permutation(len(cards))
        cards, entries = [cards[i] for i in order], [entries[i] for i in order]
    payload = dict(schema_version=1, mode=mode, entries=entries)
    write(output / 'annotations_template.json', payload)
    data = json.dumps(dict(cards=cards, template=payload, chains=CHAINS)).replace('<', '\\u003c')
    page = VIEWER.replace('__DATA__', data)
    (output / 'index.html').write_text(page, encoding='utf-8')
    return len(entries)

def import_review(work, path, config, mapping_path=None):
    work = Path(work)
    payload = read(path)
    mapping = read(mapping_path) if mapping_path else {}
    items = review_items(work)
    existing = read(work / 'annotations.json') if (work / 'annotations.json').exists() else {}
    updates = {}
    for entry in payload['entries']:
        if not entry.get('reviewed'):
            continue  # Draft/unreviewed records never erase completed reviews.
        key = mapping.get(entry['key'], entry['key'])
        if key not in items:
            raise ValueError(f'Unknown review key: {key}')
        entry['key'] = key
        item = items[key]
        xyz = load_xyz(work / item['path'])
        if entry['fps'] != item['fps']:
            raise ValueError('Review cannot change the recorded frame rate')
        validate_annotation(entry, xyz)
        if item['phase'] == 'after':
            before_ann = annotation(work, item['before_key'])
            if before_ann.get('floor_verified') and entry['floor_verified'] and not np.isclose(entry['floor_m'], before_ann['floor_m'], atol=1e-8, rtol=0):
                raise ValueError('Before/after review must retain the same fixed floor reference')
        measure(xyz, item['fps'], entry, config['thresholds'])  # overlap/geometry validation
        outcome = entry.get('outcome', {})
        if outcome.get('reviewed'):
            if payload['mode'] != 'pairs':
                raise ValueError('Correction outcomes must come from the pair review')
            for field in ('target_resolved', 'new_defect', 'action_preserved', 'unnecessary_edit'):
                if outcome.get(field) is not None and not isinstance(outcome[field], bool):
                    raise ValueError('Outcome values must be true, false or null (uncertain)')
        entry['imported_at'] = now()
        updates[key] = entry
    if updates:
        write(work / 'review_history' / f'{digest(updates)}.json', updates)
        existing.update(updates)
        write(work / 'annotations.json', existing)
        rows = manifest(work)
        for row in rows:
            ann = existing.get(row['clip_id'] + '/initial')
            if ann:
                row.update(annotation_status='reviewed' if ann['labels_complete'] else 'partially_reviewed',
                           defect_labels=ann['labels'])
        from .common import save_manifest
        save_manifest(work, rows)
    return len(updates)

VIEWER = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Motion pilot review</title><style>
body{font:16px system-ui;margin:24px;max-width:1250px;background:#f4f6f8;color:#182536}
article{background:white;padding:18px;margin:20px 0;border-radius:10px}canvas{background:#edf2f7;max-width:48%;border:1px solid #ddd}
textarea{width:98%;height:220px;font:13px monospace}button,input{margin:8px}pre{white-space:pre-wrap}summary{cursor:pointer}
.tip{padding:12px;background:#e0edf9}label{display:inline-block}</style>
<h1>Motion pilot review</h1><p class="tip">Offline viewer. Blue = anatomical left; red = right.
The floor stays fixed. Scrub frames and rotate the camera; screen-left is not anatomical left.
All intervals use zero-based frames [start, end), with end excluded. Missing information stays uncertain.</p>
<p>Review floor and nominal metre scale. Mark support from the action and visible planted foot,
not from low speed alone. Choose ankle/toe/both anchors; exclude heel/toe rolling where that anchor moves.
Mark legitimate flight separately. No contact is required for a raised swing foot.</p>
<details><summary>Annotation examples and review instructions</summary><pre>
support: [{"body_part":"left_foot","anchor":"toe","start":10,"end":30}]
required_support: [{"start":0,"end":40}]
airborne: [{"start":40,"end":50}]
labels: [{"defect":"sliding","body_part":"left_foot","start":12,"end":27,"severity":"moderate"}]
Use [] for no defects ONLY after setting labels_complete:true and reviewed:true.
Human feedback: explicitly set feedback.approved:true, add your reviewer ID,
decision (defect/no_defect/uncertain), locations in feedback.defects, and the instruction.
Changing a proposal is allowed; proposals are NOT ground truth.
Pair reviews hide the condition. Left is before; right is after. Playbacks use actual fps.
Set outcome booleans only when supported; null means uncertain. Add newly observed labels.
Review support again after editing: timing may change. Do not copy old frame intervals blindly.
Click Save current page after editing, then import the downloaded JSON with the CLI.
</pre></details><button id="save">Save current page as reviewed_annotations.json</button><span id="notice"></span>
<main></main><script>const DATA=__DATA__; const editors=[];
function viewer(parent,xyz,fps,floor,bounds){const c=document.createElement('canvas');c.width=540;c.height=410;parent.append(c);
const ctx=c.getContext('2d');let yaw=.6; const all=bounds; const lo=[0,1,2].map(k=>Math.min(...all.map(p=>p[k])));
const hi=[0,1,2].map(k=>Math.max(...all.map(p=>p[k]))); const centre=lo.map((v,k)=>(v+hi[k])/2);
const scale=280/Math.max(1.5,...hi.map((v,k)=>v-lo[k]));
function draw(frame){ctx.clearRect(0,0,540,410);const project=p=>{let x=p[0]-centre[0],z=p[2]-centre[2],y=p[1]-centre[1];
return [270+scale*(Math.cos(yaw)*x-Math.sin(yaw)*z),205-scale*(y*.87-(Math.sin(yaw)*x+Math.cos(yaw)*z)*.3)];};
ctx.strokeStyle='#c7d2dc';ctx.lineWidth=1;for(let g=-2;g<=2;g+=.5){for(let q=0;q<2;q++){let a=[centre[0]-2,floor,centre[2]+g],b=[centre[0]+2,floor,centre[2]+g];
if(q){a=[centre[0]+g,floor,centre[2]-2];b=[centre[0]+g,floor,centre[2]+2];}ctx.beginPath();ctx.moveTo(...project(a));ctx.lineTo(...project(b));ctx.stroke();}}
DATA.chains.forEach((chain,k)=>{ctx.strokeStyle=['#c54040','#176abb','#354a5f','#c54040','#176abb'][k];ctx.lineWidth=3;ctx.beginPath();
chain.forEach((j,i)=>{const p=project(xyz[Math.min(frame,xyz.length-1)][j]);if(i)ctx.lineTo(...p);else ctx.moveTo(...p);});ctx.stroke();});
ctx.fillStyle='#172b43';ctx.fillText('Frame '+Math.min(frame,xyz.length-1)+' / '+(xyz.length-1)+' | '+fps+' fps',15,22);
[7,8,10,11].forEach(j=>{ctx.fillText(String(j),...project(xyz[Math.min(frame,xyz.length-1)][j]));});}
return {draw,rotate:v=>{yaw=v;}};}
DATA.cards.forEach((card,i)=>{const article=document.createElement('article');const title=document.createElement('h2');title.textContent=card.code;article.append(title);
const text=document.createElement('p');text.textContent=card.caption;article.append(text);const canvases=document.createElement('div');article.append(canvases);
if(card.target_labels){const target=document.createElement('p');target.textContent='Target set from initial review (resolve all): '+JSON.stringify(card.target_labels);article.insertBefore(target,canvases);}
const views=[];const bounds=[...card.xyz,...(card.before||[])].flat();if(card.before)views.push(viewer(canvases,card.before,card.fps,card.floor,bounds));views.push(viewer(canvases,card.xyz,card.fps,card.floor,bounds));
const slider=document.createElement('input');slider.type='range';slider.min=0;slider.max=Math.max(card.xyz.length,card.before?.length||0)-1;slider.value=0;slider.style.width='65%';article.append(slider);
const camera=document.createElement('input');camera.type='range';camera.min=-3.14;camera.max=3.14;camera.step=.02;camera.value=.6;camera.title='Camera yaw';article.append(camera);
let timer=null;const draw=()=>views.forEach(v=>v.draw(+slider.value));slider.oninput=draw;camera.oninput=()=>{views.forEach(v=>v.rotate(+camera.value));draw();};
const play=document.createElement('button');play.textContent='Play / pause';play.onclick=()=>{if(timer){clearInterval(timer);timer=null;}else timer=setInterval(()=>{slider.value=(+slider.value+1)%(+slider.max+1);draw();},1000/card.fps);};article.append(play);
if(card.proposals){const p=document.createElement('pre');p.textContent='Unverified rule proposal: '+JSON.stringify(card.proposals,null,2);article.append(p);}
const edit=document.createElement('textarea');edit.value=JSON.stringify(DATA.template.entries[i],null,2);editors.push(edit);article.append(edit);document.querySelector('main').append(article);draw();});
document.querySelector('#save').onclick=()=>{try{const payload={...DATA.template,entries:editors.map(e=>JSON.parse(e.value))};const a=document.createElement('a');
const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));a.href=url;a.download='reviewed_annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
document.querySelector('#notice').textContent='Downloaded. Import using pilot review-import.';}catch(e){alert('Invalid JSON: '+e.message);}};
</script></html>'''

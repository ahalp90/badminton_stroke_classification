"""Build a self-contained per-video view from the saved comparison tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = '''<!doctype html>
<html lang="en-AU"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Where each video's annotations succeed and fail</title>
<style>
body{font:17px/1.5 system-ui,sans-serif;color:#202b35;background:#f6f8fa;margin:0}
main{max-width:1060px;margin:32px auto;padding:0 24px 48px}h1{font-size:30px;line-height:1.2}h2{font-size:23px;margin-top:30px}
p{max-width:850px}.intro{color:#435465}.controls{display:flex;gap:24px;flex-wrap:wrap;margin:24px 0}
label{font-weight:650}select{display:block;margin-top:6px;padding:10px;font:inherit;background:white;border:1px solid #71808a;border-radius:5px}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.card{background:white;padding:18px;border:1px solid #dae2e8;border-radius:7px}
.big{font-size:29px;font-weight:700;display:block;color:#00659e}.detail{font-size:15px;color:#435465}.note{padding:14px 18px;background:#fff0db;border-left:4px solid #a65b00}
table{width:100%;border-collapse:collapse;background:white}caption{text-align:left;margin-bottom:10px;font-weight:650}
th,td{padding:14px 10px;border:1px solid #d5dde3;text-align:right}th{background:#edf2f6}th:first-child{text-align:left}thead th{text-align:center}
.matrix td{font-size:23px;min-width:110px;text-align:center}.scroll{overflow-x:auto}.footnote{font-size:15px;color:#435465}.axis{font-size:15px;margin:12px 0 5px}
@media(max-width:700px){.cards{grid-template-columns:1fr}h1{font-size:26px}.matrix td{min-width:65px}th,td{padding:10px 5px}}
@media print{body{background:white}main{margin:0}.controls{margin:10px 0}.card{break-inside:avoid}table{break-inside:avoid}}
</style></head><body><main>
<h1>Where each video's annotations succeed and fail</h1>
<p class="intro">47 previously examined ShuttleSet22 videos · cleaned labels · ±10 frames at 30 fps.
Choose a video to see its contact, player and rally results. These counts compare saved outputs with labels; they are not a fresh test on unseen matches.</p>
<div class="controls"><label>Video<select id="video"></select></label><label>Output<select id="model"><option value="learned">Final learned detector</option><option value="heuristic">Ordinary heuristic</option></select></label></div>
<div id="warning"></div><div class="cards" id="cards"></div>
<h2>Which player was named for each labelled hit?</h2>
<p>The rows say who the label names. The columns say what the detector returned within the timing allowance.
A correct hit needs both the right time and the right player. Far and near refer to the image, not an athlete's identity across court-end changes.</p>
<p class="axis"><strong>Rows:</strong> labelled player &nbsp; · &nbsp; <strong>Columns:</strong> detector's answer</p>
<div class="scroll"><table class="matrix"><thead><tr><th scope="col">Labelled player</th><th scope="col">Far player</th><th scope="col">Near player</th><th scope="col">Hit found;<br>no player</th><th scope="col">No timing<br>match</th></tr></thead><tbody id="matrix"></tbody></table></div>
<p class="footnote">Every labelled hit appears once. Shading shows the share within each row; numbers are counts. An unknown label cannot confirm the player.</p>
<p id="extras" class="note"></p>
<h2>What was available at the missed contact times?</h2>
<div class="scroll"><table><caption>Saved input state at each labelled frame</caption><thead><tr><th>Input state</th><th>Timing matched</th><th>Missed</th><th>Total labels</th></tr></thead><tbody id="inputs"></tbody></table></div>
<p class="footnote">These are linked pipeline stages, not separate proven causes. A rejected frame can still match a nearby event. Label errors can coexist with court rejection.</p>
<h2>How much of each rally is usable?</h2>
<table><thead><tr><th>Requirement</th><th>Labelled rallies meeting it</th></tr></thead><tbody id="rallies"></tbody></table>
<p class="footnote">A clip can contain the whole rally yet have extra or missing hits. An exact sequence has every labelled hit matched once and no extra hit. Fully correct also requires the right players.</p>
<p id="selection"></p>
<p class="footnote">The source tables also contain all-source labels and ±5-frame results. Video IDs belong to ShuttleSet22 and differ from original ShuttleSet IDs.</p>
<script id="data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent);
const video=document.getElementById('video'),model=document.getElementById('model');
const number=value=>Number(value).toLocaleString('en-AU');
const rate=(count,total)=>(100*count/total).toFixed(1)+'%';
const rows=data.filter(row=>row.model==='learned').sort((a,b)=>a.fully_correct_rate_percent-b.fully_correct_rate_percent);
for(const row of rows){const option=document.createElement('option');option.value=row.fixture;option.textContent='Video '+row.fixture;video.append(option);}
video.value='53';
function update(){
const row=data.find(item=>item.fixture===Number(video.value)&&item.model===model.value);
document.getElementById('warning').innerHTML=row.fixture===15?'<p class="note"><strong>Video 15: labels and footage disagree.</strong> Treat these counts as recorded disagreements, not verified detector errors.</p>':'';
const cards=[['Fully correct rallies',row.fully_correct_rallies,row.labelled_rallies],['Contacts: time + player',row.confirmed_contacts,row.labelled_contacts],['Serves: time + player',row.confirmed_serves,row.labelled_serves]];
document.getElementById('cards').innerHTML=cards.map(([title,count,total])=>'<div class="card">'+title+'<span class="big">'+rate(count,total)+'</span><span class="detail">'+number(count)+' / '+number(total)+'</span></div>').join('');
const targets=['Far','Near','Unknown'];const predictions=['Far','Near','Unassigned','Missed prediction'];
document.getElementById('matrix').innerHTML=targets.map(target=>{const counts=predictions.map(prediction=>row.matrix[target+'|'+prediction]||0);const total=counts.reduce((sum,value)=>sum+value,0);return '<tr><th scope="row">'+(target==='Unknown'?'Unknown label':target+' player')+'</th>'+counts.map(count=>'<td style="background:rgba(0,114,178,'+(total?0.05+0.22*count/total:0)+')">'+number(count)+'</td>').join('')+'</tr>';}).join('');
document.getElementById('extras').textContent=number(row.emitted_unmatched_events)+' emitted events have no cleaned-label match. These are separate from the matrix, which starts from labelled hits. Missing labels can create unmatched events; they are not all proven false physical hits.';
const states=[['Court rejected','court_rejected'],['Court accepted; a player pick missing','accepted_missing_pick'],['Court accepted; both players picked','accepted_both_picked']];
document.getElementById('inputs').innerHTML=states.map(([title,key])=>'<tr><th scope="row">'+title+'</th><td>'+number(row['matched_'+key])+'</td><td>'+number(row['missed_'+key])+'</td><td>'+number(row['labelled_'+key])+'</td></tr>').join('');
const rallyRows=[['Whole rally fits in a clip',row.contained_rallies],['Exact contact sequence; players not required',row.timing_complete_rallies],['Fully correct contact sequence and players',row.fully_correct_rallies]];
document.getElementById('rallies').innerHTML=rallyRows.map(([title,count])=>'<tr><th scope="row">'+title+'</th><td>'+number(count)+' / '+number(row.labelled_rallies)+' ('+rate(count,row.labelled_rallies)+')</td></tr>').join('');
document.getElementById('selection').textContent=row.model==='learned'?'The fixed selection keeps '+number(row.selected_proposals)+' clips: '+number(row.selected_correct)+' correct, '+number(row.selected_wrong)+' wrong and '+number(row.selected_unknown)+' unknown.':'No learned selection rule is applied to the ordinary heuristic output.';
}
video.addEventListener('change',update);model.addEventListener('change',update);update();
</script></main></body></html>'''


def run() -> None:
    results = pd.read_csv(ROOT / "results/video_outcome_breakdown.csv.gz")
    confusion = pd.read_csv(ROOT / "results/video_player_confusion.csv.gz")
    results = results[(results.population == "retained") & (results.tolerance_base30 == 10)]
    confusion = confusion[(confusion.population == "retained") & (confusion.tolerance_base30 == 10)]
    if "omission" in results:
        results = results[results.omission == "all47"]
        confusion = confusion[confusion.omission == "all47"]
    assert len(results) == 94
    records = json.loads(results.to_json(orient="records"))
    for row in records:
        cells = confusion[(confusion.fixture == row["fixture"]) & (confusion.model == row["model"])]
        assert cells.contacts.sum() == row["labelled_contacts"]
        row["matrix"] = {f"{cell.target_player}|{cell.predicted_player}": int(cell.contacts)
                         for cell in cells.itertuples(index=False)}
    payload = json.dumps(records, separators=(",", ":"), allow_nan=False)
    (ROOT / "VIDEO_BREAKDOWN.html").write_text(TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    print("Wrote 94 video/output combinations; every matrix sums to its labelled contact count")


if __name__ == "__main__":
    run()

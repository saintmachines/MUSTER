#!/usr/bin/env python3
"""Generate thirteen scoped SDK README/page rows and the two-reader Auto CSV."""
import argparse
import csv
from decimal import Decimal,ROUND_HALF_UP
import html
import io
import json
from pathlib import Path
import re
import sys
from native_mab_contract import (AUTO_ADDITIONAL_CSV_FIELDS, AUTO_ADDITIONAL_CSV_PATH,
    automatic_records, number, validate_data)

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'docs/data/native_mab_0_1_2.json'
METRICS=('AR','TTL','LRU','SF','Overall')
RESULT_NOTE=(
    'Overall is higher for all five readers with SDK paper settings than with the original implementation. '
    'Relative to the original implementation, automatic settings raise SF and lower LRU for each measured reader. '
    'Overall rises for GPT-4o-mini and GPT-5-mini and falls for GPT-4.1-mini.'
)
AUTO_SCOPE=(
    'Paper settings retain the cell representation originally derived from up to the first three stores of each cell. '
    'Automatic settings derive a plan per store at the benchmark reference chunk size P and retain the original LongMemEval rounds adapter.'
)

def display(x):
    return str(Decimal(str(x)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))

def scoped_rows(data):
    output=[]
    for record in data['paper_configuration_readers']:
        output.append(dict(reader_id=record['id'],label=record['label'],scope='original',configuration='Original implementation',values=record['original']))
    for record in data['paper_configuration_readers']:
        output.append(dict(reader_id=record['id'],label=record['label'],scope='sdk-paper',configuration='SDK (paper settings)',values=record['native']))
    for record in automatic_records(data):
        output.append(dict(reader_id=record['id'],label=record['label'],scope='sdk-auto',configuration='SDK (automatic settings)',values=record['native']))
    return output

def rows(data):
    return [(r['label'],r['configuration'],r['values']) for r in scoped_rows(data)]

def auto_extension_accounting(data):
    record=automatic_records(data)[1]
    reads=record['read_provenance']
    return (
        'For each of GPT-4.1-mini and GPT-5-mini in Auto, '
        f'{reads["completed_extension_reader_response"]:,} non-FC slots use completed extension responses, '
        f'{reads["reused_original_saved_answer_identical_full_request"]:,} reuse original answers, and '
        f'{reads["explicit_saved_raw_reuse_identical_request"]:,} reuses a saved raw response. '
        f'Each reader reuses {record["latest_SDK_FC_saved_request_reuses"]:,} SDK paper-settings FC answers '
        f'after full-request matching. All {record["judge_jobs"]:,} judge jobs per reader were resolved and are tracked separately.'
    )

def markdown(data):
    groups=data['provenance']['response_accounting_groups']
    gpt,frontier=groups['gpt_readers'],groups['frontier_readers']
    paper=data['paper']
    lines=['<a id="native-mab"></a>','<details>',
        '<summary><b>Library 0.1.2 on all fourteen MemoryAgentBench tasks</b></summary>','',
        'SDK (paper settings) holds the paper cell settings fixed. SDK (automatic settings) is measured separately for GPT-4o-mini, GPT-4.1-mini and GPT-5-mini. Both configurations are compared with the Original implementation using the same source stores and questions; selected reader contexts can differ.','',
        '| Reader | Configuration | AR | TTL | LRU | SF | Overall |',
        '|---|---|---:|---:|---:|---:|---:|']
    for reader,condition,values in rows(data):
        lines.append('| '+' | '.join([reader,condition]+[display(values[k]) for k in METRICS])+' |')
    lines+=['',
        RESULT_NOTE,'',
        f'Each reader covers {data["benchmark"]["questions_per_reader"]:,} questions and {data["benchmark"]["raw_slots_per_reader"]:,} response slots in 14 cells; DetectiveQA retains three reads. SDK encoder and reranker input limits are 2,048 tokens. {AUTO_SCOPE}','',
        f'SDK paper-settings response accounting: the three GPT readers use {gpt["fresh_validated_reader_response_rows"]:,} fresh reader answers and {gpt["full_message_reused_question_reader_score_units"]:,} reused question-reader score units. The two frontier readers use {frontier["fresh_reader_response_slots"]:,} fresh response slots and {frontier["reused_raw_response_slots"]:,} reused raw slots. API attempt counts are tracked separately. Reuse requires identical complete messages and request settings.','',
        auto_extension_accounting(data),'',
        '[All task cells and conditions](https://github.com/saintmachines/MUSTER/blob/main/docs/data/native_mab_0_1_2.json) · '
        f'[Updated paper](https://github.com/saintmachines/MUSTER/blob/main/{paper["pdf"]}) · '
        f'[Marked changes](https://github.com/saintmachines/MUSTER/blob/main/{paper["marked_pdf"]})','',
        '[Additional automatic-setting task cells](https://github.com/saintmachines/MUSTER/blob/main/paper/auto_additional_mab_cells.csv) records 28 task rows and two Overall rows for GPT-4.1-mini and GPT-5-mini.','',
        'The numbered-fact rules were developed on the reported FC data. The 71.45 six-setting overview, original build timings and original Infini-style results retain their original-implementation scope.','',
        '</details>']
    return '\n'.join(lines)

def landing(data):
    groups=data['provenance']['response_accounting_groups']
    gpt,frontier=groups['gpt_readers'],groups['frontier_readers']
    paper=data['paper']
    parts=['<section class="sec" id="native-mab">','<div class="wrap">',
        '<div class="eyebrow">Library 0.1.2 · all fourteen MemoryAgentBench tasks</div>',
        '<h2>The released library, measured across the benchmark.</h2>',
        '<p class="lead">Original implementation and SDK (paper settings) are compared at five readers. SDK (automatic settings) is measured separately for GPT-4o-mini, GPT-4.1-mini and GPT-5-mini. Source stores and questions are shared; selected contexts can differ.</p>',
        '<div class="table-wrap"><table class="t" style="min-width:760px"><caption class="small">Original benchmark metrics, 0–100 scale. Overall averages the four competencies.</caption>',
        '<thead><tr>'+''.join('<th scope="col">'+x+'</th>' for x in ('Reader','Configuration',*METRICS))+'</tr></thead><tbody>']
    for row in scoped_rows(data):
        parts.append('<tr data-result-scope="'+row['scope']+'" data-reader="'+row['reader_id']+'"><th scope="row">'+html.escape(row['label'])+'</th><td>'+html.escape(row['configuration'])+'</td>'+''.join('<td class="n">'+display(row['values'][k])+'</td>' for k in METRICS)+'</tr>')
    parts+=['</tbody></table></div>',
        '<p class="note">'+RESULT_NOTE+'</p>',
        f'<p class="note">{data["benchmark"]["questions_per_reader"]:,} questions and {data["benchmark"]["raw_slots_per_reader"]:,} response slots per reader · 14 task cells · three DetectiveQA reads · SDK encoder and reranker input limits 2,048 tokens.</p>',
        '<p class="note"><a href="data/native_mab_0_1_2.json">All task cells and conditions</a> · '
        f'<a href="https://github.com/saintmachines/MUSTER/blob/main/{html.escape(paper["pdf"])}">Updated paper</a> · '
        f'<a href="https://github.com/saintmachines/MUSTER/blob/main/{html.escape(paper["marked_pdf"])}">Changes in blue</a> · '
        '<a href="https://github.com/saintmachines/MUSTER/blob/main/paper/auto_additional_mab_cells.csv">Additional automatic-setting task cells</a></p>',
        '<details class="more"><summary>Evaluation scope and response accounting</summary><div><p>'+AUTO_SCOPE+' The numbered-fact rules were developed on the reported FC600 and 262K data. Archived answers require complete-message and request-setting equality.</p>'
        f'<p>SDK paper-settings accounting: three GPT readers use {gpt["fresh_validated_reader_response_rows"]:,} fresh reader answers and {gpt["full_message_reused_question_reader_score_units"]:,} reused question-reader score units. Two frontier readers use {frontier["fresh_reader_response_slots"]:,} fresh response slots and {frontier["reused_raw_response_slots"]:,} reused raw slots. API attempt counts are tracked separately.</p>'
        '<p>'+auto_extension_accounting(data)+'</p>'
        '<p>The 71.45 six-setting overview, original build timings and original Infini-style columns retain their original-implementation scope.</p></div></details>',
        '</div>','</section>']
    return '\n'.join(parts)


def render_auto_additional_csv(data):
    """Render checked canonical values without file access; accepts pending paper metadata."""
    validate_data(data)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=AUTO_ADDITIONAL_CSV_FIELDS, lineterminator='\n')
    writer.writeheader()
    for record in automatic_records(data)[1:]:
        for cell in (*data['benchmark']['cell_order'], 'Overall'):
            if cell == 'Overall':
                n = repeats = ''
                old, new = number(record['original']['Overall']), number(record['native']['Overall'])
            else:
                entry = record['cells'][cell]
                n, repeats = entry['n'], entry['repeats']
                old, new = number(entry['original_percent']), number(entry['native_percent'])
            writer.writerow(dict(sdk_version='0.1.2', configuration='per_store_auto_native',
                                 reader=record['id'], cell=cell, n=n, repeats=repeats,
                                 old_percent=str(old), new_percent=str(new), delta_pp=str(new-old),
                                 metric_source_sha256=data['provenance']['auto_extension_metric_sha256'][record['id']]))
    return stream.getvalue()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--check',action='store_true');a=ap.parse_args()
    data=json.loads(DATA.read_text(),parse_float=Decimal);validate_data(data,require_final_paper=True)
    changes=[]
    for path,content in ((ROOT/'README.md',markdown(data)),(ROOT/'docs/index.html',landing(data))):
        before=path.read_text();pattern=re.compile(r'(?<=<!-- BEGIN native-mab -->\n).*?(?=\n<!-- END native-mab -->)',re.S)
        assert len(pattern.findall(before))==1,(path,'Need exactly one SDK result block')
        after=pattern.sub(lambda _:content,before)
        if before!=after:
            changes.append(str(path.relative_to(ROOT)))
            if not a.check:path.write_text(after)
    csv_path=ROOT/AUTO_ADDITIONAL_CSV_PATH
    csv_text=render_auto_additional_csv(data)
    if not csv_path.is_file() or csv_path.read_text(encoding='utf-8')!=csv_text:
        changes.append(AUTO_ADDITIONAL_CSV_PATH)
        if not a.check:csv_path.write_text(csv_text,encoding='utf-8')
    if a.check and changes:
        print('SDK MAB blocks are stale: '+', '.join(changes));return 1
    print('SDK MAB README/page data match' if a.check else 'Updated: '+', '.join(changes))
    return 0

if __name__=='__main__':sys.exit(main())

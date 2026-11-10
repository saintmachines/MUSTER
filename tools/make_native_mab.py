#!/usr/bin/env python3
"""Generate scoped five-reader SDK README/page blocks from checked cell data."""
import argparse
from decimal import Decimal,ROUND_HALF_UP
import html
import json
from pathlib import Path
import re
import sys
from native_mab_contract import validate_data

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'docs/data/native_mab_0_1_2.json'

def display(x):
    return str(Decimal(str(x)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))

def scoped_rows(data):
    output=[]
    for record in data['paper_configuration_readers']:
        output.append(dict(reader_id=record['id'],label=record['label'],scope='original',configuration='Original implementation',values=record['original']))
    for record in data['paper_configuration_readers']:
        output.append(dict(reader_id=record['id'],label=record['label'],scope='sdk-paper',configuration='SDK (paper settings)',values=record['native']))
    record=data['protocol_auto_4omini']
    output.append(dict(reader_id=record['id'],label=record['label'],scope='sdk-auto',configuration='SDK (automatic settings)',values=record['native']))
    return output

def rows(data):
    return [(r['label'],r['configuration'],r['values']) for r in scoped_rows(data)]

def markdown(data):
    groups=data['provenance']['response_accounting_groups']
    gpt,frontier=groups['gpt_readers'],groups['frontier_readers']
    paper=data['paper']
    lines=['<a id="native-mab"></a>','<details>',
        '<summary><b>Library 0.1.2 on all fourteen MemoryAgentBench tasks</b></summary>','',
        'SDK (paper settings) holds the paper cell settings fixed. SDK (automatic settings) is measured separately for GPT-4o-mini. Both are compared with the Original implementation using the same source stores and questions; selected reader contexts can differ.','',
        '| Reader | Configuration | AR | TTL | LRU | SF | Overall |',
        '|---|---|---:|---:|---:|---:|---:|']
    for reader,condition,values in rows(data):
        lines.append('| '+' | '.join([reader,condition]+[display(values[k]) for k in ('AR','TTL','LRU','SF','Overall')])+' |')
    lines+=['',
        'Overall is higher for all five readers with SDK paper settings; individual cells do not all improve. With automatic settings, GPT-4o-mini improves SF and Overall while AR, TTL and LRU decrease versus the original implementation.','',
        f'Each reader covers {data["benchmark"]["questions_per_reader"]:,} questions and {data["benchmark"]["raw_slots_per_reader"]:,} response slots in 14 cells; DetectiveQA retains three reads. SDK encoder and reranker input limits are 2,048 tokens. Automatic settings derive a plan per store at the benchmark reference chunk P and retain the original LongMemEval rounds adapter.','',
        f'The three GPT readers use {gpt["fresh_validated_reader_response_rows"]:,} fresh reader answers and {gpt["full_message_reused_question_reader_score_units"]:,} reused question-reader score units. The two frontier readers use {frontier["fresh_reader_response_slots"]:,} fresh response slots and {frontier["reused_raw_response_slots"]:,} reused raw slots. These accounting units are kept separate; neither is an API POST count. Reuse requires identical complete messages and request settings.','',
        '[All task cells and conditions](https://github.com/saintmachines/MUSTER/blob/main/docs/data/native_mab_0_1_2.json) · '
        f'[Updated paper](https://github.com/saintmachines/MUSTER/blob/main/{paper["pdf"]}) · '
        f'[Marked changes](https://github.com/saintmachines/MUSTER/blob/main/{paper["marked_pdf"]})','',
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
        '<p class="lead">Original implementation and SDK (paper settings) are compared at five readers. SDK (automatic settings) is measured separately for GPT-4o-mini. Source stores and questions are shared; selected contexts can differ.</p>',
        '<div class="table-wrap"><table class="t" style="min-width:760px"><caption class="small">Original benchmark metrics, 0–100 scale. Overall averages the four competencies.</caption>',
        '<thead><tr>'+''.join('<th scope="col">'+x+'</th>' for x in ('Reader','Configuration','AR','TTL','LRU','SF','Overall'))+'</tr></thead><tbody>']
    for row in scoped_rows(data):
        parts.append('<tr data-result-scope="'+row['scope']+'" data-reader="'+row['reader_id']+'"><th scope="row">'+html.escape(row['label'])+'</th><td>'+html.escape(row['configuration'])+'</td>'+''.join('<td class="n">'+display(row['values'][k])+'</td>' for k in ('AR','TTL','LRU','SF','Overall'))+'</tr>')
    parts+=['</tbody></table></div>',
        '<p class="note">Overall is higher for all five readers with SDK paper settings; individual tasks are not universally improved. GPT-4o-mini automatic settings improve SF and Overall while AR, TTL and LRU decrease.</p>',
        f'<p class="note">{data["benchmark"]["questions_per_reader"]:,} questions and {data["benchmark"]["raw_slots_per_reader"]:,} response slots per reader · 14 task cells · three DetectiveQA reads · SDK encoder and reranker input limits 2,048 tokens.</p>',
        '<p class="note"><a href="data/native_mab_0_1_2.json">All task cells and conditions</a> · '
        f'<a href="https://github.com/saintmachines/MUSTER/blob/main/{html.escape(paper["pdf"])}">Updated paper</a> · '
        f'<a href="https://github.com/saintmachines/MUSTER/blob/main/{html.escape(paper["marked_pdf"])}">Changes in blue</a></p>',
        '<details class="more"><summary>Evaluation scope and response accounting</summary><div><p>SDK automatic settings derive a plan per store at the benchmark reference chunk P and retain the original LongMemEval rounds adapter. The numbered-fact rules were developed on the reported FC600 and 262K data. Archived answers require complete-message and request-setting equality.</p>'
        f'<p>Three GPT readers: {gpt["fresh_validated_reader_response_rows"]:,} fresh reader answers and {gpt["full_message_reused_question_reader_score_units"]:,} reused question-reader score units. Two frontier readers: {frontier["fresh_reader_response_slots"]:,} fresh response slots and {frontier["reused_raw_response_slots"]:,} reused raw slots. The units are not combined into an API call count.</p>'
        '<p>The 71.45 six-setting overview, original build timings and original Infini-style columns retain their original-implementation scope.</p></div></details>',
        '</div>','</section>']
    return '\n'.join(parts)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--check',action='store_true');a=ap.parse_args()
    data=json.loads(DATA.read_text());validate_data(data,require_final_paper=True)
    changes=[]
    for path,content in ((ROOT/'README.md',markdown(data)),(ROOT/'docs/index.html',landing(data))):
        before=path.read_text();pattern=re.compile(r'(?<=<!-- BEGIN native-mab -->\n).*?(?=\n<!-- END native-mab -->)',re.S)
        assert len(pattern.findall(before))==1,(path,'Need exactly one SDK result block')
        after=pattern.sub(lambda _:content,before)
        if before!=after:
            changes.append(str(path.relative_to(ROOT)))
            if not a.check:path.write_text(after)
    if a.check and changes:
        print('SDK MAB blocks are stale: '+', '.join(changes));return 1
    print('SDK MAB README/page data match' if a.check else 'Updated: '+', '.join(changes))
    return 0

if __name__=='__main__':sys.exit(main())

"""One existing 32-codepoint lexer guard; native missing lines keep source identity."""
import pytest
import muster
from muster.stats import LEAD,lead_match,lead_serial

def test_shared_lexer_keeps_existing_bounds_and_unicode_digits():
    valid='X'*30+' 1: body';invalid='X'*31+' 1: body'
    assert lead_match(valid).end(2)==32 and lead_match(invalid) is None
    assert int(lead_match('Step ١: body').group(2))==1
    assert lead_match('Step 123456789: body') is None
    assert lead_match('user: Step 1: body') is None
    assert LEAD.match(invalid) is not None  # existing grammar allows it; shared cap is the semantic guard

def test_native_raw_outlier_is_kept_in_source_position_without_revision_clock():
    lines=['Step %d: The Atlas project is led by Alice.'%i for i in range(100,220)]
    lines[60]='X'*40+' 999: An original literal line, not a trusted serial.'
    text='\n'.join(lines);share,median=lead_serial(text)
    assert share>=.95 and median<=20
    mem=muster.build(text,models=muster.Models.fake(16))
    assert mem.plan.branch=='numbered' and mem.plan.serial=='lead'
    assert mem.cut.units[60]==lines[60] and mem.cut.prefixes[60]=='' and mem.cut.order[60]==-1
    assert mem.parent[60]=='u:rawline60' and not mem.index.structural and not mem.index.newest_row
    assert mem.cut.diagnostics['unmarked_records']==[60]
    context=mem.context('original literal line',U=120)
    assert context.ids==tuple(range(120)) and context.text==text
    assert context.text.index('Step 159:')<context.text.index(lines[60])<context.text.index('Step 161:')

def test_more_than_twenty_raw_lines_keep_detected_lead_prefix_literal():
    lines=['#%d EntityAA owns a blue key.'%i for i in range(23)]
    lines.append('Y'*40+' 777: This tail has a prefix outside the known bound.')
    mem=muster.build('\n'.join(lines),embed=False)
    assert mem.plan.serial=='lead' and mem.cut.order[-1]==-1
    assert mem.render(list(range(24))).text=='\n'.join(lines)

def test_dot_native_and_exact_do_not_inherit_the_lead_prefix_cap():
    text='\n'.join(' '*40+'%d. EntityAA owns a blue key.'%i for i in range(24))
    plan=muster.derive(text);assert plan.branch=='numbered' and plan.serial=='dot'
    native=muster.build(text,plan,embed=False);exact=muster.build(text,plan.with_(mode='exact'),embed=False)
    assert native.cut.units==exact.cut.units and native.cut.order==exact.cut.order==tuple(range(24))
    assert native.cut.diagnostics==exact.cut.diagnostics and native.render(range(24)).text==exact.render(range(24)).text

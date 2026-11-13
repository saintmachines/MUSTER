"""Flat text keeps the reader bytes and still exposes source blocks to applications."""
import muster
from muster.render import flat


def test_flat_source_blocks_preserve_text_order_and_parent_identity():
    units = ('first source', 'second source', 'third source')
    parent = ('episode-a', 'episode-a', 'episode-b')
    context = flat([2, 0], units, parent)
    assert context.text == 'first source\nthird source'
    assert context.ids == (0, 2)
    assert [(block.index, block.parent, block.ids, block.text) for block in context.blocks] == [
        (1, 'episode-a', (0,), 'first source'), (2, 'episode-b', (2,), 'third source')]


def test_automatic_record_context_names_its_original_records():
    records = [muster.Record('The launch is scheduled for Friday.', id='source-a'),
               muster.Record('The rehearsal is scheduled for Thursday.', id='source-b')]
    memory = muster.build(records, embed=False)
    assert memory.plan.render == 'flat'
    context = memory.render(range(len(memory)))
    assert context.blocks and context.text == '\n'.join(memory.units)
    recovered = {record.id for block in context.blocks for i in block.ids for record in memory.unit_records(i)}
    assert recovered == {'source-a', 'source-b'}

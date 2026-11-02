"""(r4 variant, 2026-10-03: evq/det/summ OFF legs = r4 per-chunk re-reads gOA_nm_<reader>_<cell>, first read) Board at the new lengths (2048/2048) vs the published NO_EXTRACT board, three readers. $0, reads files only.
Runs noxc_board.py unchanged except that the OFF legs of the seven cells whose contexts changed are replaced by the full
re-reads gOA_l2k_<reader>_<cell>; ICL x5 and FC x2 keep the published legs (byte-identical contexts)."""
import os, sys
os.chdir(os.environ.get('MUSTER_HOME', os.path.dirname(os.path.abspath(__file__))))
src = open('noxc_board.py').read()
anchor = "NOTAG = {'4o'"
assert src.count(anchor) == 1
inject = ("if os.environ.get('L2K') == '1':\n"
          "    for _rd, _m in (('4o', '4om'), ('41m', '41m'), ('5m', '5m')):\n"
          "        OFF[_rd].update({_c: 'gOA_l2k_%s_%s' % (_m, _c) for _c in ('shqa', 'mhqa', 'lme', 'evq', 'rec', 'summ', 'det')})\n"
          "        OFF[_rd].update({_c: 'gOA_r4_%s_%s' % (_m, _c) for _c in ('evq', 'summ', 'det')})\n")
src = src.replace(anchor, inject + anchor)
src = src.replace("not off_t.startswith('gOA_5m')", "not off_t.startswith(('gOA_5m', 'gOA_l2k_5m', 'gOA_r4_5m'))")
src = src.replace("if cell == 'rec' or not tag.startswith('gOA_5m'):", "if cell == 'rec' or not tag.startswith(('gOA_5m', 'gOA_l2k_5m', 'gOA_r4_5m')):")
exec(compile(src, 'noxc_board_l2k', 'exec'))

"""derive_config.stats()/derive() verbatim on the LoCoMo store, reference scale c=4096 (as BEAM: no chunk prescribed)."""
import sys; sys.argv=['x']
import derive_config as DC
for prot in (4096,):
    cov, num, tok = DC.stats("beam/locomo.parquet", list(range(10)), prot)
    print("SAMPLE_ROWS=%d prot=%d atom_cov=%.4f numbered=%.4f tok/row=%d" % (DC.SAMPLE_ROWS, prot, cov, num, tok))
    DC.SAMPLE_ROWS = 10
    cov, num, tok = DC.stats("beam/locomo.parquet", list(range(10)), prot)
    print("SAMPLE_ROWS=10 prot=%d atom_cov=%.4f numbered=%.4f tok/row=%d" % (prot, cov, num, tok))
    DC.REG.CELLS["locomo"] = ("beam/locomo.parquet", list(range(10)), 1540, "J", prot)
    DC.SAMPLE_ROWS = 3
    print(DC.derive("locomo"))

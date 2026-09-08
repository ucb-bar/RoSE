"""Pull the per-op cycle dict out of a curated-verify PASS line."""
import ast, re, sys
def cycles(path):
    t = open(path, errors="replace").read()
    out = None
    for m in re.finditer(r"cycles=(\{[^}]*\})", t):
        out = ast.literal_eval(m.group(1))
    return out
a = cycles(sys.argv[1]); b = cycles(sys.argv[2]) if len(sys.argv) > 2 else None
if b is None:
    tot = sum(a.values())
    print(f"{'op':<18}{'Mcycles':>10}{'share':>8}")
    for k, v in sorted(a.items(), key=lambda x: -x[1]):
        print(f"{k:<18}{v/1e6:>10.1f}{100*v/tot:>7.1f}%")
    print(f"{'TOTAL':<18}{tot/1e6:>10.1f}")
else:
    ta, tb = sum(a.values()), sum(b.values())
    print(f"{'op':<18}{'A Mcyc':>10}{'B Mcyc':>10}{'speedup':>9}")
    for k in sorted(set(a) | set(b), key=lambda k: -(a.get(k, 0))):
        va, vb = a.get(k, 0), b.get(k, 0)
        sp = f"{va/vb:.2f}x" if vb else "-"
        print(f"{k:<18}{va/1e6:>10.1f}{vb/1e6:>10.1f}{sp:>9}")
    print(f"{'TOTAL':<18}{ta/1e6:>10.1f}{tb/1e6:>10.1f}{ta/tb:>8.2f}x")

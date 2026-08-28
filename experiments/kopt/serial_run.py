import sys, os, time
sys.path.insert(0,"/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw")
from modelblaster.optimize.firesim_eval import fq_transport as T
OUT="/scratch/dima/rose-infra/RoSE/experiments/kopt/logs"
cfg=T.FqConfig()
for a in sys.argv[1:]:
    tag,_,elf=a.partition("=")
    model=os.environ.get("EXPECT_MODEL") or None
    for attempt in (1,2,3):
        try:
            u=T.run_fq(elf, tag=tag, cfg=cfg, expected_model=model)
            open(f"{OUT}/{tag}.uartlog","w").write(u)
            v=[l for l in u.splitlines() if "max_abs_err" in l]
            print(f"OK   {tag} (try {attempt}) {v[0].strip() if v else ''}", flush=True)
            break
        except Exception as e:
            print(f"retry {tag} ({attempt}): {e}", flush=True)
            time.sleep(20)
    else:
        print(f"FAIL {tag}", flush=True)

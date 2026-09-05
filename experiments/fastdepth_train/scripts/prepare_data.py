#!/usr/bin/env python3
"""Fetch NYU Depth V2 (FastDepth HDF5 repack) shard by shard.

Each shard is downloaded, extracted, and DELETED before the next one starts.
Downloading all 14 first would need ~35 GB of tars alongside ~35 GB of
extracted .h5 on a 144 GB disk that already holds 51 GB.
"""
import os, sys, tarfile, shutil
from huggingface_hub import list_repo_files, hf_hub_download

REPO, OUT = "sayakpaul/nyu_depth_v2", "/home/ubuntu/nyu_h5"
os.makedirs(OUT, exist_ok=True)
tars = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".tar"))
print(f"{len(tars)} shards", flush=True)
total = 0
for i, t in enumerate(tars):
    p = hf_hub_download(REPO, t, repo_type="dataset")
    with tarfile.open(p) as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".h5")]
        tf.extractall(OUT, members=members)
    total += len(members)
    sz = os.path.getsize(p)
    os.remove(p)                      # blob
    try:                              # and the symlink farm pointing at it
        shutil.rmtree(os.path.join(os.environ.get("HF_HOME", ""), "hub",
                      "datasets--sayakpaul--nyu_depth_v2", "snapshots"), ignore_errors=True)
    except Exception:
        pass
    print(f"  [{i+1}/{len(tars)}] {os.path.basename(t)}  +{len(members)} files "
          f"({sz/1e9:.1f} GB tar)  total={total}", flush=True)
n_tr = sum(len(f) for _, _, f in os.walk(f"{OUT}/train"))
n_va = sum(len(f) for _, _, f in os.walk(f"{OUT}/val"))
print(f"DONE train={n_tr} val={n_va}")
if (n_tr, n_va) != (47584, 654):
    print(f"WARNING: expected 47584/654, got {n_tr}/{n_va}")

# /*
#  * Cicada Emergence Optimizer: A Hybrid Success-History Differential Evolution with Staggered Brood Restarts
#  * Authors: Van Van Chau (1,2), Minh Tuan Nguyen (1,*), and Phayung Meesad (2,*)
#  * (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#  *     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam
#  *     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
#  * (2) Department of Information Technology Management, Faculty of Information Technology and Digital Innovation,
#  *     King Mongkut's University of Technology North Bangkok (KMUTNB),
#  *     1518 Pracharat 1 Rd, Wongsawang, Bangsue, Bangkok 10800, Thailand
#  *     phayung.m@itd.kmutnb.ac.th
#  * (*) Corresponding authors
#  * Repository: https://github.com/vancv43/Cicada-Emergence-Optimizer (MIT License)
#  */
"""Shard packing and the batch-run worker loop for the RC22-RC33 extension (PROTOCOL_rwx.md).
On a workstation: python rwx_runner.py build -> worklist_rwx.json ; batch-run: rwx_runner.main(shard, out_dir, prior_dirs)."""
import glob, json, os, sys, time
import multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
CHUNK_CAP_H = 1.5
SESSION_CAP_H = 11.0          # batch-run CPU sessions stop at 12 h
N_SHARDS = int(os.environ.get("RWX_SHARDS", 6))


def run_hours(alg, code):
    """batch-run hours for one run, from timing_rwx.json (dry runs scaled to the full budget)."""
    t = json.load(open(os.path.join(HERE, "timing_rwx.json")))
    return t[f"{alg}|{code}"] / 3600.0


def build():
    import rwx_lib as X
    cs = []
    for alg in X.ALGS:
        for code in X.CODES:
            h = run_hours(alg, code)
            per = max(1, min(30, int(CHUNK_CAP_H / max(h, 1e-6))))
            for s0 in range(0, 30, per):
                s1 = min(30, s0 + per)
                cs.append(dict(alg=alg, code=code, s0=s0, s1=s1, h=round(h * (s1 - s0), 4)))
    cs.sort(key=lambda c: -c["h"])
    shards, load = [[] for _ in range(N_SHARDS)], [0.0] * N_SHARDS
    for c in cs:
        k = load.index(min(load)); shards[k].append(c); load[k] += c["h"]
    wl = [dict(round="rwx", shard=k + 1, hours=round(load[k], 2), chunks=shards[k]) for k in range(N_SHARDS)]
    json.dump(wl, open(os.path.join(HERE, "worklist_rwx.json"), "w"), indent=0)
    print(f"{len(cs)} chunks, {sum(load):.1f} core-h, {N_SHARDS} shards:", [round(x, 1) for x in load])


def _work(args):
    c, out_dir = args
    import rwx_lib as X
    t0 = time.time()
    try:
        X.run_unit(c["alg"], c["code"], c["s0"], c["s1"], out_dir)
        return c, time.time() - t0, None
    except Exception as e:
        import traceback
        return c, time.time() - t0, repr(e) + " | " + traceback.format_exc().replace("\n", " / ")[-600:]


def main(shard, out_dir, prior_dirs=(), workers=4):
    import rwx_lib as X
    start = time.time()
    wl = json.load(open(os.path.join(HERE, "worklist_rwx.json")))
    assert wl[0]["round"] == "rwx", "stale worklist"
    have = set()
    for d in [out_dir, *prior_dirs]:
        have |= {os.path.basename(p) for p in glob.glob(os.path.join(d, "RWX__*.csv"))}
    todo = [c for c in wl[shard - 1]["chunks"] if X.name(c["alg"], c["code"], c["s0"], c["s1"]) not in have]
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, f"rwx_shard{shard:02d}_log.txt"), "a")
    print(f"shard {shard}: {len(todo)} chunks to run", file=log, flush=True)
    pending, running = list(todo), []
    with mp.Pool(workers, maxtasksperchild=20) as pool:
        while pending or running:
            while pending and len(running) < workers:
                c = pending.pop(0)
                if (time.time() - start) / 3600 + c["h"] * 1.5 > SESSION_CAP_H:
                    print(f"skip (cap) {X.name(c['alg'], c['code'], c['s0'], c['s1'])}", file=log, flush=True)
                    continue
                running.append(pool.apply_async(_work, ((c, out_dir),)))
            time.sleep(5)
            for r in [r for r in running if r.ready()]:
                running.remove(r)
                c, dt, err = r.get()
                print(f"{time.strftime('%H:%M:%S')} {X.name(c['alg'], c['code'], c['s0'], c['s1'])} {dt:.0f}s {err or 'ok'}",
                      file=log, flush=True)
    print(f"SHARD {shard} DONE {(time.time() - start) / 3600:.2f} h", file=log, flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "build":
    build()

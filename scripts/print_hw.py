import pandas as pd

m = pd.read_parquet(
    "outputs/runs/benchmark_20261119T222606Z_a1361f0f/run_meta.parquet"
)
row = m.iloc[0]
print("hostname", row.get("hostname"))
print("platform", row.get("platform"))
print("cpu_count_logical", row.get("cpu_count_logical"))
print("python_version", row.get("python_version"))
for phase in [
    "timing_sequential_summary",
    "timing_ray_workers_1_summary",
    "timing_ray_workers_2_summary",
    "timing_ray_workers_4_summary",
]:
    sub = m[m["phase"] == phase]
    if len(sub):
        r = sub.iloc[0]
        print(
            phase,
            "workers=",
            r.get("workers"),
            "startup_mean=",
            r.get("startup_s_mean"),
            "warm_mean=",
            r.get("warm_eval_s_mean"),
        )

timing = m[m["phase"].astype(str).str.startswith("timing_ray_workers_2") & ~m["phase"].astype(str).str.endswith("summary")]
if len(timing):
    print("batch_size", timing.iloc[0].get("batch_size"))
    print("node_ids", timing.iloc[0].get("node_ids"))
    print("topology", timing.iloc[0].get("topology"))

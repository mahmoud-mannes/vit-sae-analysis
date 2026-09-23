"""Rebuild w3_per_block.csv and w3_gate_report.csv from the merged JSON.
The curve and summary CSVs do not include baseline and floor, so they are unchanged.
"""
import json, glob, os, shutil
import pandas as pd

MAIN = glob.glob("w3_output/w3_windows_*.json")[0]
J = json.load(open(MAIN))
R = J["results"]

pb = []
for lab, e in R.items():
    if e.get("status") != "ok":
        continue
    for nm, r in e.get("conditions", {}).items():
        if "ssdc" not in r:
            continue
        for b, v in enumerate(r["ssdc"]):
            pb.append({"run": lab, "condition": nm, "block": b, "ssdc": v})
pd.DataFrame(pb).to_csv("w3_output/w3_per_block.csv", index=False)

gdf = pd.DataFrame(J["gates"])
gdf.to_csv("w3_output/w3_gate_report.csv", index=False)

print(pd.DataFrame(pb).groupby("run").condition.nunique().to_string())
print()
print(gdf.groupby(["scope", "status"]).size().to_string())

archive = shutil.make_archive("w3_output", "zip", "w3_output")
print(f"\narchive: {archive} ({os.path.getsize(archive):,} bytes)")

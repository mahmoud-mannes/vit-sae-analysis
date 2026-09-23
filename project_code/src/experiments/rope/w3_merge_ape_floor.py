"""Merge the re-measured APE floors into the main W3 output and add the
denominators that depend on them.
"""
import json, glob, shutil, time

MAIN = glob.glob("w3_output/w3_windows_*.json")[0]
PATCH = "w3_output/ape_floor_patch.json"

shutil.copy(MAIN, MAIN + ".pre_ape_fix")
J = json.load(open(MAIN))
P = json.load(open(PATCH))
R = J["results"]
A = J["w2_anchors"]

added = []
for label, r in P["conditions"].items():
    e = R[label]
    old = e["conditions"].get("floor", {})
    assert old.get("status") == "failed", f"{label}: floor is not the failed one, refusing"
    e["conditions"]["floor"] = r

    PL = A[label]["probe_layer"]
    b = e["conditions"]["baseline"]["ssdc"]
    f = r["ssdc"]
    e["baseline_ssdc"], e["floor_ssdc"] = b, f
    e["achievable_range"] = b[PL] - f[PL]
    e["floor_remeasured"] = {"reason": P["reason"], "when": P["generated_utc"]}

    for name, passed, detail in [
        ("floor hits its W1-B anchor",
         abs(f[PL] - A[label]["floor"]) < 0.01,
         f"{f[PL]:+.6f} vs anchor {A[label]['floor']}"),
        ("floor is below baseline", f[PL] < b[PL] - 0.01,
         f"{b[PL]:+.4f} -> {f[PL]:+.4f}"),
    ]:
        J["gates"].append({"gate": name, "scope": label,
                           "status": "PASS" if passed else "FAIL", "detail": detail})
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {name}: {detail}")
    added.append(label)

fails = [g for g in J["gates"] if g["status"] == "FAIL"]
J["gate_summary"] = {"total": len(J["gates"]),
                     "pass": len(J["gates"]) - len(fails),
                     "fail": len(fails)}
J["ape_floor_fix"] = {"applied_utc": time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime()),
                      "models": added, "patch_file": PATCH}

json.dump(J, open(MAIN, "w"), indent=2)
print(f"\nmerged {added} into {MAIN}")
print(f"gates now {J['gate_summary']['pass']} PASS / {J['gate_summary']['fail']} FAIL")
for g in fails:
    print("  FAIL", g["scope"], "|", g["gate"], "|", g["detail"])

"""Merge the W7 runs that machine restarts split into several files.

reg1-gap (RoPE, R12): the 04:06 rerun supplies heads 10-11 and the shuffled top-rebuilder
conditions. SAM (APE): the 01:02 rerun overrides the first run where both have a condition,
and the 04:06 rerun adds the matched-random draws. Conditions that ended in an error are skipped.
Each merged file records its sources in `merged_from` and `merge_note`.
"""
import json, os
from pathlib import Path

OUT = Path(os.environ.get("W7_OUT", "w7_output"))


def merge(files, out, note):
    base = json.load(open(OUT / files[0]))
    conds = dict(base.get("conditions", {}))
    for f in files[1:]:
        for k, v in json.load(open(OUT / f))["conditions"].items():
            if "error" not in v:
                conds[k] = v
    base["conditions"] = conds
    base["merged_from"] = files
    base["merge_note"] = note
    json.dump(base, open(OUT / out, "w"), indent=2)
    print(out, len(conds), "conditions")


merge(["w7rope_R12_reg1_gap_2026-09-24T22-49-53Z.json", "w7rope_R12_reg1_gap_2026-09-25T04-06-12Z.json"],
      "w7rope_R12_reg1_gap_MERGED.json",
      "Heads 10-11 and the shuffled top-rebuilder conditions come from the 04:06 rerun. "
      "Per-image correctness exists only for those 6 conditions (earlier file overwritten).")
merge(["w7ape_sam_2026-09-24T18-12-56Z.json", "w7ape_sam_2026-09-25T01-02-08Z.json",
       "w7ape_sam_2026-09-25T04-06-09Z.json"],
      "w7ape_sam_MERGED.json",
      "Conditions from the first run (18:12), overridden by the 01:02 rerun where both exist, "
      "plus the matched-random draws from the 04:06 rerun. Per-image correctness exists only "
      "for the 6 matched-random conditions.")

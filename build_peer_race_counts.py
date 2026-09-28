"""Peer MK1 race counts: unique WR POINTSR races per athlete, plus domestic
Australian races (Australian Open / Oceania / Penrith Open) missing from the WR
exports. Outputs the cumulative IQR band and race-threshold ages."""
import json
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

WR_CSV = "Athlete_Career_Progression_Comparison_WR_Releases.csv"
DOMESTIC_CSV = "domestic_results.csv"
THRESHOLDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
FIRST_NAME_ALIAS = {"Joe CLARKE": "Joseph"}

use_domestic = "--no-domestic" not in sys.argv

df = pd.read_csv(WR_CSV)
df = df[df["CLASS"] == "MK1"]
pts_cols = [c for c in df.columns if c.startswith("POINTSR")]

birth = {n: int((g["YEAR"] - g["AGE"]).astype(int).mode().iloc[0])
         for n, g in df.groupby("ATHLETE_NAME_ORIGINAL")}
max_release_age = df.groupby("ATHLETE_NAME_ORIGINAL")["AGE"].max().astype(int).to_dict()

races = defaultdict(dict)
for _, row in df.iterrows():
    ath = row["ATHLETE_NAME_ORIGINAL"]
    for c in pts_cols:
        v = row[c]
        if pd.isna(v):
            continue
        parts = str(v).split(";")
        if len(parts) < 2:
            continue
        name = parts[1].strip()
        m = re.search(r"(19|20)\d{2}", name)
        if not name or not m:
            continue
        y = int(m.group(0))
        if name not in races[ath] or y < races[ath][name]:
            races[ath][name] = y


def wr_has(wr, comp, y):
    if comp == "Australian Open":
        return f"ICF World ranking Penrith {y}" in wr or f"Australian Open Penrith {y}" in wr
    if comp == "Oceania":
        return any(n.startswith("Oceania Championships") and n.endswith(str(y)) for n in wr)
    return any(comp in n and n.endswith(str(y)) for n in wr)


def domestic_name(comp, loc, y):
    loc = "Auckland" if loc == "Werro" else loc
    if comp == "Oceania":
        return f"Oceania Championships {loc} {y}"
    return f"{comp} {loc} {y}"


added = defaultdict(list)
if use_domestic:
    dom = pd.read_csv(DOMESTIC_CSV)
    dom = dom[dom["CLASS"] == "K1M"].copy()
    dom["LAST"] = dom["LASTNAME"].str.strip().str.upper()
    dom["FIRST"] = dom["FIRSTNAME"].str.strip()
    for ath in list(races):
        first, last = ath.split(" ", 1)
        m = dom[(dom["LAST"] == last) & (dom["FIRST"] == FIRST_NAME_ALIAS.get(ath, first))]
        for (y, comp, loc), _ in m.groupby(["YEAR", "COMPETITION", "LOCATION"]):
            y = int(y)
            if wr_has(races[ath], comp, y):
                continue
            name = domestic_name(comp, loc, y)
            races[ath][name] = y
            added[ath].append(name)

series, starts, thresholds = {}, {}, {}
for ath, r in races.items():
    by_age = defaultdict(int)
    for y in r.values():
        a = y - birth[ath]
        if 10 <= a <= 50:
            by_age[a] += 1
    ages = sorted(by_age)
    starts[ath] = ages[0]
    cum, age_cum, th = 0, {}, {}
    for a in range(ages[0], ages[-1] + 1):
        cum += by_age.get(a, 0)
        age_cum[a] = cum
        for t in THRESHOLDS:
            if f"r{t}" not in th and cum >= t:
                th[f"r{t}"] = a
    for a in range(ages[-1] + 1, max_release_age[ath] + 1):
        age_cum[a] = cum
    for t in THRESHOLDS:
        th.setdefault(f"r{t}", None)
    th["total"] = cum
    series[ath], thresholds[ath] = age_cum, th


def quartile(vals, q):
    return float(np.percentile(vals, q, method="linear"))


def tukey_filter(vals):
    if len(vals) < 4:
        return vals
    q1, q3 = quartile(vals, 25), quartile(vals, 75)
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    kept = [v for v in vals if lo <= v <= hi]
    return kept if len(kept) >= 2 else vals


cohort = [a for a, s in starts.items() if s <= 18]
band, prev = [], None
for age in sorted({a for ath in cohort for a in series[ath]}):
    if age < 15 or age > 34:
        continue
    vals = [series[ath][age] for ath in cohort if age in series[ath]]
    if len(vals) < 2:
        continue
    kept = tukey_filter(vals)
    q1, q3 = quartile(kept, 25), quartile(kept, 75)
    if prev:
        q1, q3 = max(q1, prev[0]), max(q3, prev[1])
    prev = (q1, q3)
    band.append({"a": age, "q1": round(q1, 2), "q3": round(q3, 2), "n": len(kept)})

json.dump({"band": band, "thresholds": thresholds, "added": added, "cohort": sorted(cohort)},
          open("/tmp/peer_race_counts.json", "w"), indent=1)
print("cohort", len(cohort), sorted(cohort))
print(json.dumps(band, separators=(",", ":")))
for ath, names in added.items():
    print(ath, names)

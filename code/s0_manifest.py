"""Stage 0. Build the this study manifest from the raw Data folder.

Input   Data/<folder>/*.wav and the one label sheet in each folder.
Output  derived/manifest.csv            one row per wav file (published columns only)
        derived/deployments.csv         one row per deployment
        derived/s0_report.json          counts and every anomaly found
        private/location_map.json       pipe location -> anonymous code (never publish)

Rules
  screening  clipped if any |x| > 16,000 counts; too short if fewer than 40,960 samples
             (the full 10 s). An earlier screening rule used 1,024, but a 10 s window is the unit
             here and only one file is short, so it is excluded either way.
  label      taken from the sheet row whose normalised filename matches the file.
  night      the date in the file name. Slots a, b, c, d are recordings on that night.
  deployment campaign + logger + anonymous pipe-location code.
  duplicates byte-identical audio (MD5 of the sample frames). If every copy is from one
             logger, the earliest copy (date, slot) is kept and the others dropped. If copies
             sit under two or more loggers, the true owner cannot be known, so every copy is
             dropped. Rule fixed on 4 November 2026 before any model was fitted.
  dates      a year written as 27 is read as the campaign year (2026). Campaign 5 files
             named 2, 3 and 4 May are re-dated to 2, 3 and 4 June. The same seven loggers
             were recording in Campaign 4 on 2 to 4 May, Campaign 5 began on 16 May, and
             2 to 4 June hold one file each where every other night holds 14.
Nothing is changed in the Data folder.
"""
import os
import re
import glob
import json
import wave
import datetime as dt
import collections
import hashlib
import numpy as np
import pandas as pd
from openpyxl import load_workbook

import p5config as C

NAME = re.compile(r"^(?:root_)?(\d+)([a-dA-D]?)-(\d+)\s*-(\d+)-(\d+)$")


def norm_name(s):
    s = os.path.splitext(str(s).strip())[0]
    s = re.sub(r"^root_", "", s, flags=re.I)
    return re.sub(r"\s+", "", s).lower()


def norm_label(v):
    v = str(v).strip().lower()
    if v in ("leak",):
        return "leak"
    if v in ("no leak", "noleak", "no-leak"):
        return "no leak"
    return None


def read_sheet(folder):
    xs = [p for p in glob.glob(os.path.join(C.DATA, folder, "*.xlsx")) if not os.path.basename(p).startswith("~$")]
    assert len(xs) == 1, (folder, xs)
    ws = load_workbook(xs[0], read_only=True, data_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    hi = next(i for i, r in enumerate(rows) if r and "Filename" in [str(x).strip() if x else x for x in r])
    head = [str(x).strip() if x is not None else None for x in rows[hi]]
    ci = {h: j for j, h in enumerate(head) if h}
    out = {}
    dup = []
    for r in rows[hi + 1:]:
        fn = r[ci["Filename"]]
        if fn is None:
            continue
        key = norm_name(fn)
        rec = dict(sheet_file=str(fn).strip(),
                   label_raw=r[ci["Status"]] if "Status" in ci else None,
                   location=(str(r[ci["Location"]]).strip() if "Location" in ci and r[ci["Location"]] else None),
                   window=r[ci["Deployement Duration"]] if "Deployement Duration" in ci else None)
        if key in out:
            dup.append(key)
        out[key] = rec
    return out, dup, os.path.basename(xs[0])


def read_wav(path):
    with wave.open(path, "rb") as w:
        fs, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float64)
    return fs, n, ch, sw, x


def parse_date(d, m, y, fallback_year):
    y = int(y)
    if y < 100:
        y += 2000
    flag = None
    if y != fallback_year:
        flag = f"year {y} in name, campaign year {fallback_year}"
        y = fallback_year
    try:
        return dt.date(y, int(m), int(d)), flag
    except ValueError:
        return None, f"invalid date {d}-{m}-{y}"


def main():
    rows, anomalies = [], collections.defaultdict(list)
    loc_map = {}
    for folder, camp, region, start in C.CAMPAIGNS:
        sheet, dup, sheet_name = read_sheet(folder)
        if dup:
            anomalies["duplicate_sheet_rows"].append([folder, dup])
        year = int(start[:4])
        wavs = sorted(glob.glob(os.path.join(C.DATA, folder, "*.wav")))
        seen = set()
        # anonymous location codes, in order of first appearance in the sheet
        locs = []
        for k, r in sheet.items():
            if r["location"] and r["location"] not in locs:
                locs.append(r["location"])
        lcode = {l: f"L{i + 1}" for i, l in enumerate(locs)}
        loc_map[C.campaign_name(camp)] = lcode
        for p in wavs:
            fname = os.path.basename(p)
            stem = os.path.splitext(fname)[0]
            key = norm_name(fname)
            m = NAME.match(stem.replace(" ", "")) or NAME.match(stem)
            if not m:
                anomalies["unparsed_name"].append([folder, fname])
                continue
            logger, slot, d, mo, y = m.groups()
            date, dflag = parse_date(d, mo, y, year)
            if dflag:
                anomalies["date_flag"].append([folder, fname, dflag])
            if camp == 5 and date is not None and date.month == 5 and date.day in (2, 3, 4):
                anomalies["date_corrected"].append([folder, fname, date.isoformat(), f"2026-06-{date.day:02d}"])
                date = dt.date(2026, 6, date.day)
            fs, n, ch, sw, x = read_wav(p)
            with wave.open(p, "rb") as w:
                md5 = hashlib.md5(w.readframes(w.getnframes())).hexdigest()
            srec = sheet.get(key)
            if srec is None:
                anomalies["no_label_row"].append([folder, fname])
            else:
                seen.add(key)
            label = norm_label(srec["label_raw"]) if srec else None
            if srec and label is None:
                anomalies["bad_label"].append([folder, fname, srec["label_raw"]])
            loc = lcode.get(srec["location"], "L1") if srec and srec["location"] else "L1"
            clip_frac = float(np.mean(np.abs(x) > C.CLIP_THRESHOLD)) if n else 1.0
            rows.append(dict(
                folder=folder, file=fname, campaign=camp, region=region,
                logger=int(logger), slot=slot.lower(), date=date.isoformat() if date else None,
                loc_code=loc, label=label, fs=fs, channels=ch, sampwidth=sw, n_samples=n,
                peak=float(np.abs(x).max()) if n else np.nan,
                rms=float(np.sqrt(np.mean((x - x.mean()) ** 2))) if n else np.nan,
                dc=float(x.mean()) if n else np.nan,
                clip_frac=clip_frac,
                clipped=bool(clip_frac > 0), md5=md5,
                too_short=bool(n < C.N_FULL),
            ))
        for k in set(sheet) - seen:
            anomalies["label_row_without_file"].append([folder, sheet[k]["sheet_file"]])

    df = pd.DataFrame(rows)
    # duplicate audio
    df["dup_group"] = df.groupby("md5")["md5"].transform("size")
    df["dup_drop"] = False
    df["dup_reason"] = ""
    for h, g in df[df["dup_group"] > 1].groupby("md5"):
        if g["logger"].nunique() > 1 or g["campaign"].nunique() > 1:
            df.loc[g.index, "dup_drop"] = True
            df.loc[g.index, "dup_reason"] = "copies under different loggers"
            anomalies["duplicate_cross_logger"].append(g[["folder", "file", "label"]].values.tolist())
        else:
            keep = g.sort_values(["date", "slot"]).index[0]
            df.loc[g.index.drop(keep), "dup_drop"] = True
            df.loc[g.index.drop(keep), "dup_reason"] = "later copy, same logger"
    df["retained"] = (~df["clipped"] & ~df["too_short"] & ~df["dup_drop"]
                      & df["label"].notna() & (df["fs"] == C.FS))
    df["deployment"] = ("C" + df["campaign"].astype(str) + "-" + df["logger"].astype(str) + "-" + df["loc_code"])
    df["night"] = df["deployment"] + "-" + df["date"].astype(str)
    df["split"] = np.where(df["campaign"].isin(C.LOCKED_CAMPAIGNS), "locked", "dev")
    df["y"] = (df["label"] == "leak").astype(int)

    # label must be constant within a deployment
    lab = df[df["label"].notna()].groupby("deployment")["label"].nunique()
    mixed = lab[lab > 1].index.tolist()
    for dpl in mixed:
        sub = df[df["deployment"] == dpl]
        anomalies["mixed_label_deployment"].append([dpl, sub["label"].value_counts().to_dict()])

    # same logger in two campaigns at once would be impossible
    span = df.groupby(["logger", "campaign"])["date"].agg(["min", "max"]).reset_index()
    for lg, g in span.groupby("logger"):
        g = g.sort_values("min").values.tolist()
        for a, b in zip(g, g[1:]):
            if b[2] <= a[3]:
                anomalies["logger_overlap_in_time"].append([int(lg), int(a[1]), int(b[1]), a[2], a[3], b[2], b[3]])

    pub_cols = ["campaign", "region", "split", "logger", "slot", "date", "loc_code", "deployment", "night",
                "label", "y", "fs", "n_samples", "peak", "rms", "dc", "clip_frac", "clipped", "too_short",
                "dup_group", "dup_drop", "dup_reason", "md5", "retained"]
    df.sort_values(["campaign", "logger", "date", "slot"]).to_csv(os.path.join(C.DERIVED, "manifest_private.csv"), index=False)
    df.sort_values(["campaign", "logger", "date", "slot"])[pub_cols].to_csv(os.path.join(C.DERIVED, "manifest.csv"), index=False)
    C.save_private("location_map.json", loc_map)

    kept = df[df["retained"]]
    dep = kept.groupby("deployment").agg(campaign=("campaign", "first"), region=("region", "first"),
                                         split=("split", "first"), logger=("logger", "first"),
                                         label=("label", "first"), y=("y", "first"),
                                         files=("file", "size"), nights=("date", "nunique"),
                                         first=("date", "min"), last=("date", "max")).reset_index()
    dep.to_csv(os.path.join(C.DERIVED, "deployments.csv"), index=False)

    def summ(d):
        return dict(files=int(len(d)), leak=int((d["label"] == "leak").sum()), no_leak=int((d["label"] == "no leak").sum()))

    by_c = []
    for c, g in df.groupby("campaign"):
        k = g[g["retained"]]
        dd = dep[dep["campaign"] == c]
        by_c.append(dict(campaign=int(c), region=g["region"].iloc[0], split=g["split"].iloc[0],
                         files=int(len(g)), clipped=int(g["clipped"].sum()), too_short=int(g["too_short"].sum()),
                         dup_dropped=int((g["dup_drop"] & ~g["clipped"] & ~g["too_short"]).sum()),
                         retained=int(len(k)), leak=int(k["y"].sum()), no_leak=int((1 - k["y"]).sum()),
                         loggers=int(k["logger"].nunique()), deployments=int(len(dd)),
                         leaking_deployments=int(dd["y"].sum()), nights=int(k["night"].nunique()),
                         first=g["date"].min(), last=g["date"].max()))
    rep = dict(
        all_files=summ(df), fs_values=df["fs"].value_counts().to_dict(),
        clipped=int(df["clipped"].sum()), too_short=int(df["too_short"].sum()),
        duplicate_files=int((df["dup_group"] > 1).sum()), duplicate_groups=int(df.loc[df["dup_group"] > 1, "md5"].nunique()),
        dropped_as_duplicate=int(df["dup_drop"].sum()),
        dropped_as_duplicate_not_otherwise_excluded=int((df["dup_drop"] & ~df["clipped"] & ~df["too_short"]).sum()),
        unlabelled=int(df["label"].isna().sum()),
        retained=summ(kept), retained_dev=summ(kept[kept.split == "dev"]), retained_locked=summ(kept[kept.split == "locked"]),
        campaigns=int(df["campaign"].nunique()), regions=int(df["region"].nunique()),
        loggers=int(kept["logger"].nunique()), deployments=int(len(dep)), leaking_deployments=int(dep["y"].sum()),
        nights=int(kept["night"].nunique()),
        nights_per_deployment=dict(median=float(dep["nights"].median()), min=int(dep["nights"].min()), max=int(dep["nights"].max())),
        clipped_share_by_label=df.groupby("label")["clipped"].mean().round(4).to_dict(),
        by_campaign=by_c, anomalies={k: v for k, v in anomalies.items()},
    )
    with open(os.path.join(C.DERIVED, "s0_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("by_campaign", "anomalies")}, indent=1, default=str))
    print(pd.DataFrame(by_c).to_string(index=False))
    for k, v in anomalies.items():
        print(f"\nANOMALY {k}: {len(v)}")
        for e in v[:12]:
            print("   ", e)


if __name__ == "__main__":
    main()

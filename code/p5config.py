"""Central configuration. Reads the raw recordings from DATA and writes every output under ROOT.

Set two environment variables before running (or edit the defaults below):
  P5_DATA   folder that holds one sub-folder of WAV files per site, plus the label workbook
  P5_ROOT   working folder for derived/, figures/, tables/, supplementary/, private/ (created if missing)
  P5_CACHE  optional local cache for decoded audio (default: ROOT/cache)

Site folders are mapped to anonymous site numbers in sites.json (not part of this repository, because the
folder names identify locations). Copy sites_example.json to sites.json and fill in the real folder names.
Only "Site n" and "District n" are ever written to published outputs.
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("P5_DATA", os.path.join(HERE, "..", "data"))
PAPER = os.environ.get("P5_ROOT", os.path.join(HERE, ".."))
ANALYSIS = os.path.dirname(DATA)
DERIVED = os.path.join(PAPER, "derived")
FIGURES = os.path.join(PAPER, "figures")
TABLES = os.path.join(PAPER, "tables")
SUPP = os.path.join(PAPER, "supplementary")
PRIVATE = os.path.join(PAPER, "private")
CACHE = os.environ.get("P5_CACHE", os.path.join(PAPER, "cache"))
for _d in (DERIVED, FIGURES, TABLES, SUPP, PRIVATE, CACHE):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- signal
FS = 4096                 # native rate of every file, fixed by the logger
NYQUIST = FS / 2          # 2,048 Hz. Nothing above this exists in the data.
N_FULL = 40960            # 10 s at 4,096 Hz
CLIP_THRESHOLD = 16000    # screening rule, |x| > 16,000 counts
MIN_SAMPLES = 1024

# ---------------------------------------------------------------- sites
# sites.json: list of [folder, site number, district code, deployment start date (YYYY-MM-DD)].
# Sites are numbered by deployment start date. District codes (R1 to R4) are used only for the
# district hold-out check.
_SITES = os.path.join(HERE, "sites.json")
if not os.path.exists(_SITES):
    raise SystemExit("sites.json not found: copy sites_example.json to sites.json and fill in the folder names")
CAMPAIGNS = [tuple(r) for r in json.load(open(_SITES, encoding="utf-8"))]
FOLDER_TO_CAMPAIGN = {f: c for f, c, _, _ in CAMPAIGNS}
FOLDER_TO_REGION = {f: r for f, _, r, _ in CAMPAIGNS}

# Development and locked test. Sites 10 to 12 were recorded after development and are scored once, at the end,
# by the models chosen on Sites 1 to 9 only.
DEV_CAMPAIGNS = list(range(1, 10))
LOCKED_CAMPAIGNS = [10, 11, 12]

SEEDS = [11, 22, 33, 44, 55]


def campaign_name(c):
    return f"Site {int(c)}"


def save_private(name, obj):
    with open(os.path.join(PRIVATE, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)

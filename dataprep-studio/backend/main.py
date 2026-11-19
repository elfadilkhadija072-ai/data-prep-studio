"""API FastAPI de DataPrep Studio.  Lancer :  python run.py  →  http://localhost:8501"""
from __future__ import annotations
import io, json, uuid
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.common import detect_problem, infer_kind, sanitize
from core.loader import normalize, read_any
from core.eda import compute_eda, correlations, categorical_stats, numeric_stats
from core.analysis import build_recommendations
from core.pipeline import preview_feature, run_pipeline
from core.report import build_report

ROOT = Path(__file__).resolve().parent
DIST = ROOT.parent / "frontend" / "dist"
SAMPLES = ROOT / "data"
MAX_SESSIONS = 6

app = FastAPI(title="DataPrep Studio", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
SESSIONS: "OrderedDict[str, dict]" = OrderedDict()


def J(obj):
    return JSONResponse(sanitize(obj))


def get(sid: str) -> dict:
    if sid not in SESSIONS:
        raise HTTPException(404, "Session introuvable : ré-importez le dataset.")
    SESSIONS.move_to_end(sid)
    return SESSIONS[sid]


def new_session(name: str, df: pd.DataFrame) -> dict:
    df, notes = normalize(df)
    sid = uuid.uuid4().hex[:10]
    SESSIONS[sid] = {"id": sid, "name": name, "df": df, "notes": notes, "cfg": {"task": "unsupervised", "target": None, "problem": None,
                     "test_size": 0.2, "drop_duplicates": True}, "eda": None, "recs": None, "result": None}
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)
    return SESSIONS[sid]


def kinds_of(s):
    return {c: infer_kind(s["df"][c]) for c in s["df"].columns}


def summary(s):
    df = s["df"]
    return {"id": s["id"], "name": s["name"], "rows": len(df), "cols": df.shape[1], "notes": s["notes"],
            "columns": [{"name": c, "kind": infer_kind(df[c]), "missing": int(df[c].isna().sum()), "unique": int(df[c].nunique())} for c in df.columns],
            "preview": json.loads(df.head(12).to_json(orient="records", date_format="iso", default_handler=str))}


# ───────────── Import ─────────────
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = read_any(file.filename or "dataset.csv", content)
    except Exception as e:
        raise HTTPException(400, f"Lecture impossible : {e}")
    if df.shape[0] < 2 or df.shape[1] < 2:
        raise HTTPException(400, "Le dataset doit contenir au moins 2 lignes et 2 colonnes.")
    return J(summary(new_session(file.filename or "dataset", df)))


@app.get("/api/samples")
def samples():
    return J([{"file": p.name, "label": p.stem.replace("_", " ").title(), "size_kb": round(p.stat().st_size / 1024)} for p in sorted(SAMPLES.glob("*.csv"))])


@app.post("/api/samples/{fname}")
def load_sample(fname: str):
    p = SAMPLES / Path(fname).name
    if not p.exists():
        raise HTTPException(404, "Exemple inconnu")
    return J(summary(new_session(p.name, read_any(p.name, p.read_bytes()))))


# ───────────── Configuration ─────────────
class Config(BaseModel):
    task: str = "unsupervised"
    target: str | None = None
    problem: str | None = None
    test_size: float = 0.2
    drop_duplicates: bool = True


@app.post("/api/session/{sid}/config")
def configure(sid: str, c: Config):
    s = get(sid); df = s["df"]
    if c.task not in ("supervised", "unsupervised"):
        raise HTTPException(400, "task invalide")
    problem = None
    if c.task == "supervised":
        if not c.target or c.target not in df.columns:
            raise HTTPException(400, "Choisissez une variable cible valide.")
        if df[c.target].isna().all():
            raise HTTPException(400, "La cible est entièrement vide.")
        problem = c.problem if c.problem in ("classification", "regression") else detect_problem(df[c.target])
    s["cfg"] = {"task": c.task, "target": c.target if c.task == "supervised" else None, "problem": problem,
                "test_size": min(max(c.test_size, 0.05), 0.5), "drop_duplicates": c.drop_duplicates}
    s["eda"] = s["recs"] = s["result"] = None
    return J({"cfg": s["cfg"]})


def ensure_eda(s):
    if s["eda"] is None:
        c = s["cfg"]
        s["eda"] = compute_eda(s["df"], c["target"], c["problem"])
        s["recs"] = build_recommendations(s["df"], s["eda"], c["target"], c["problem"])
        s["recs"]["global_plan"]["drop_duplicates"] = bool(s["cfg"]["drop_duplicates"] and s["eda"]["overview"]["duplicates"])
    return s["eda"]


# ───────────── EDA ─────────────
@app.get("/api/session/{sid}/eda")
def eda(sid: str):
    s = get(sid)
    return J({"eda": ensure_eda(s), "cfg": s["cfg"], "notes": s["notes"]})


@app.get("/api/session/{sid}/corr")
def corr(sid: str, method: str = "pearson"):
    s = get(sid)
    if method not in ("pearson", "spearman", "kendall"):
        raise HTTPException(400, "méthode invalide")
    return J(correlations(s["df"], kinds_of(s), method))


@app.get("/api/session/{sid}/scatter")
def scatter(sid: str, x: str, y: str, hue: str | None = None, n: int = 1500):
    s = get(sid); df = s["df"]
    for c in (x, y):
        if c not in df.columns:
            raise HTTPException(404, f"colonne {c} inconnue")
    cols = [x, y] + ([hue] if hue and hue in df.columns else [])
    d = df[cols].dropna()
    d = d.sample(min(n, len(d)), random_state=1) if len(d) else d
    return J({"points": [{"x": r[x], "y": r[y], "g": str(r[hue]) if len(cols) > 2 else None} for _, r in d.iterrows()]})


@app.get("/api/session/{sid}/groupby")
def groupby(sid: str, feature: str):
    """Distribution d'une feature par classe de la cible (supervisé / classification)."""
    s = get(sid); df, c = s["df"], s["cfg"]
    if c["problem"] != "classification" or feature not in df.columns:
        return J({"groups": []})
    k = infer_kind(df[feature])
    out = []
    for cls, g in df.groupby(df[c["target"]].astype(str)):
        if k == "numeric":
            v = g[feature].dropna()
            out.append({"group": cls, "n": len(g), "mean": v.mean(), "median": v.median(), "std": v.std()})
        else:
            vc = g[feature].astype(str).value_counts(normalize=True).head(5)
            out.append({"group": cls, "n": len(g), "top": [{"value": a, "pct": b * 100} for a, b in vc.items()]})
    return J({"groups": out[:20], "kind": k})


# ───────────── Recommandations ─────────────
@app.get("/api/session/{sid}/recommendations")
def recommendations(sid: str):
    s = get(sid); ensure_eda(s)
    return J({"recs": s["recs"], "cfg": s["cfg"]})


class PreviewReq(BaseModel):
    feature: str
    plan: dict = {}


@app.post("/api/session/{sid}/preview")
def preview(sid: str, r: PreviewReq):
    s = get(sid); ensure_eda(s)
    if r.feature not in s["df"].columns:
        raise HTTPException(404, "feature inconnue")
    before, after, info, rm = preview_feature(s["df"], kinds_of(s), s["recs"], s["cfg"], r.feature, r.plan)
    kind = infer_kind(s["df"][r.feature])

    def describe(series):
        if pd.api.types.is_numeric_dtype(series):
            return {"type": "numeric", **numeric_stats(series), "missing": int(series.isna().sum())}
        if pd.api.types.is_datetime64_any_dtype(series):
            return {"type": "datetime", "missing": int(series.isna().sum())}
        return {"type": "categorical", **categorical_stats(series), "missing": int(series.isna().sum())}

    res = {"info": info, "kind": kind, "before": describe(before)}
    if after is None:
        res["after"] = {"dropped": True}
    else:
        if rm is not None:
            after = after.loc[~rm.reindex(after.index, fill_value=False)]
        if after.shape[1] == 1:
            res["after"] = {"columns": [after.columns[0]], **describe(after.iloc[:, 0])}
        else:
            res["after"] = {"type": "multi", "columns": list(after.columns),
                            "shares": [{"column": c, "pct": float(after[c].mean() * 100)} for c in after.columns][:40]}
    return J(res)


# ───────────── Exécution ─────────────
class ApplyReq(BaseModel):
    choices: dict = {}
    drop_duplicates: bool | None = None


@app.post("/api/session/{sid}/apply")
def apply(sid: str, r: ApplyReq):
    s = get(sid); ensure_eda(s)
    cfg = dict(s["cfg"])
    cfg["drop_duplicates"] = s["recs"]["global_plan"]["drop_duplicates"] if r.drop_duplicates is None else r.drop_duplicates
    try:
        res = run_pipeline(s["df"], kinds_of(s), s["recs"], cfg, r.choices)
    except Exception as e:  # message lisible côté UI
        raise HTTPException(500, f"Erreur pendant l'exécution : {type(e).__name__}: {e}")
    s["result"], s["run_cfg"] = res, cfg
    s["cfg_applied"] = cfg
    train = res["train"]
    return J({"rows_before": res["rows_before"], "cols_before": res["cols_before"], "rows_train": res["rows_train"],
              "rows_test": res["rows_test"], "cols_after": res["cols_after"], "removed_rows": res["removed_rows"],
              "remaining_missing": res["remaining_missing"], "global_log": res["global_log"],
              "infos": {k: {x: v[x] for x in ("steps", "new_columns", "warnings", "imputed", "clipped", "removed_rows", "dropped")}
                        for k, v in res["infos"].items()},
              "preview": json.loads(train.head(10).to_json(orient="records", default_handler=str)),
              "columns": list(train.columns)})


def _final(s, part):
    res = s.get("result")
    if not res:
        raise HTTPException(400, "Exécutez d'abord les transformations.")
    tr, te = res["train"], res["test"]
    if part == "train":
        return tr
    if part == "test":
        if te is None:
            raise HTTPException(400, "Pas de jeu de test (apprentissage non supervisé).")
        return te
    if te is None:
        return tr
    return pd.concat([tr.assign(_split="train"), te.assign(_split="test")]).sort_index()


@app.get("/api/session/{sid}/download")
def download(sid: str, part: str = "all", fmt: str = "csv"):
    s = get(sid); d = _final(s, part)
    base = Path(s["name"]).stem + f"_transformed_{part}"
    if fmt == "xlsx":
        if len(d) > 1_000_000:
            raise HTTPException(400, "Trop de lignes pour Excel : utilisez le CSV.")
        b = io.BytesIO(); d.to_excel(b, index=False, engine="xlsxwriter")
        return Response(b.getvalue(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": f'attachment; filename="{base}.xlsx"'})
    return Response(d.to_csv(index=False).encode("utf-8"), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{base}.csv"'})


@app.get("/api/session/{sid}/report")
def report(sid: str):
    s = get(sid)
    if not s.get("result"):
        raise HTTPException(400, "Exécutez d'abord les transformations.")
    s["cfg"] = {**s["cfg"], **{k: v for k, v in s["cfg_applied"].items() if k == "drop_duplicates"}}
    data = build_report(s)
    return Response(data, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{Path(s["name"]).stem}_rapport_features.xlsx"'})


# ───────────── Frontend compilé (React) ─────────────
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")

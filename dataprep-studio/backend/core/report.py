"""Rapport Excel : une ligne par feature (stats, anomalies, recommandation, transformation réellement appliquée)."""
from __future__ import annotations
import io
from datetime import datetime
import xlsxwriter
from .common import SLOT_LABELS, slots_key

SLOTS = ["clean", "impute", "outliers", "transform", "scale", "encode"]


def _label(rec, slot, value):
    for o in rec["options"].get(slot, []):
        if o["value"] == value:
            return o["label"]
    return value


def plan_text(rec, plan):
    if plan.get("drop"):
        return "Supprimer la colonne"
    parts = [f"{SLOT_LABELS[s]} : {_label(rec, s, plan[s])}" for s in rec["slots"] if plan.get(s, "none") != "none"]
    return " | ".join(parts) or "Aucune transformation"


def decision(rec, plan):
    rp = rec["plan"]
    keys = ["drop"] + rec["slots"]
    if all(plan.get(k) == rp.get(k) for k in keys):
        return "Recommandation acceptée"
    if plan.get("drop") is False and all(plan.get(k, "none") == "none" for k in rec["slots"]):
        return "Aucune transformation (choix utilisateur)"
    return "Choix personnalisé"


def build_report(sess) -> bytes:
    eda, recs, cfg, res = sess["eda"], sess["recs"], sess["cfg"], sess["result"]
    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True, "nan_inf_to_errors": True})
    F = lambda **k: wb.add_format({"font_name": "Calibri", "font_size": 10, "valign": "top", **k})
    title = F(bold=True, font_size=18, font_color="#0B3C49")
    h = F(bold=True, font_color="#FFFFFF", bg_color="#0B3C49", border=1, text_wrap=True, valign="vcenter", align="center")
    cell = F(border=1, border_color="#D0D7DE", text_wrap=True)
    num = F(border=1, border_color="#D0D7DE", num_format="#,##0.####")
    pct = F(border=1, border_color="#D0D7DE", num_format="0.00")
    key = F(bold=True, bg_color="#E6F4F1", border=1, border_color="#D0D7DE")
    ok = F(border=1, border_color="#D0D7DE", bg_color="#DFF5E3", text_wrap=True)
    warn = F(border=1, border_color="#D0D7DE", bg_color="#FFF1CC", text_wrap=True)
    crit = F(border=1, border_color="#D0D7DE", bg_color="#FFD9D6", text_wrap=True)

    # ── Résumé ──
    ws = wb.add_worksheet("Résumé")
    ws.hide_gridlines(2); ws.set_column(0, 0, 38); ws.set_column(1, 1, 70)
    ws.write(0, 0, "Rapport de Data Preprocessing & Cleaning", title)
    ws.write(1, 0, f"Généré le {datetime.now():%d/%m/%Y %H:%M} — dataset : {sess['name']}", F(italic=True))
    ov = eda["overview"]
    rows = [
        ("Type d'apprentissage", "Supervisé" if cfg["task"] == "supervised" else "Non supervisé"),
        ("Variable cible", cfg.get("target") or "—"),
        ("Problème détecté", {"classification": "Classification", "regression": "Régression"}.get(cfg.get("problem"), "—")),
        ("Lignes (avant)", ov["rows"]), ("Colonnes (avant)", ov["cols"]),
        ("Cellules manquantes (avant)", f"{ov['missing_cells']} ({ov['missing_cells_pct']:.2f} %)"),
        ("Doublons (avant)", f"{ov['duplicates']} ({ov['duplicates_pct']:.2f} %)"),
        ("Lignes train / test (après)", f"{res['rows_train']} / {res['rows_test']}"),
        ("Colonnes (après)", res["cols_after"]),
        ("Valeurs manquantes restantes", res["remaining_missing"]),
        ("Lignes outliers supprimées (train)", res["removed_rows"]),
    ]
    for i, (k, v) in enumerate(rows, start=3):
        ws.write(i, 0, k, key); ws.write(i, 1, v, cell)
    r = 3 + len(rows) + 1
    ws.write(r, 0, "Bonnes pratiques anti-leakage appliquées", F(bold=True, font_size=12, font_color="#0B3C49"))
    for j, line in enumerate(res["global_log"], start=1):
        ws.merge_range(r + j, 0, r + j, 1, "• " + line, cell)

    # ── Features ──
    ws = wb.add_worksheet("Features")
    cols = ["Feature", "Rôle", "Type", "Valeurs non nulles", "Manquants", "% manquants", "Valeurs uniques", "Moyenne", "Écart-type",
            "Min", "Q1", "Médiane", "Q3", "Max", "Asymétrie", "Outliers (IQR)", "% outliers", "Association cible",
            "Anomalies détectées", "Transformation recommandée", "Justification statistique", "Décision de l'utilisateur",
            "Transformation réellement appliquée", "Détail d'exécution", "Colonnes produites", "Valeurs imputées",
            "Valeurs écrêtées", "Lignes supprimées", "Avertissements"]
    widths = [22, 9, 12, 11, 10, 10, 10, 12, 12, 12, 12, 12, 12, 12, 10, 10, 10, 11, 48, 44, 60, 24, 44, 52, 30, 10, 10, 10, 30]
    for i, (c, w) in enumerate(zip(cols, widths)):
        ws.write(0, i, c, h); ws.set_column(i, i, w)
    ws.set_row(0, 34); ws.freeze_panes(1, 1)
    for ri, f in enumerate(eda["features"], start=1):
        n = f["name"]; rec = recs["features"][n]; info = res["infos"][n]; st = f.get("stats", {}) or {}
        plan = info["plan"]
        sev_fmt = {"crit": crit, "warn": warn}.get(rec["severity"], ok)
        vals = [n, "Cible" if rec["role"] == "target" else "Feature", f["kind"], f["count"], f["missing"], f["missing_pct"], f["unique"],
                st.get("mean"), st.get("std"), st.get("min"), st.get("q1"), st.get("median"), st.get("q3"), st.get("max"),
                st.get("skew"), st.get("outliers_iqr"), st.get("outliers_iqr_pct"), f.get("target_assoc")]
        for ci, v in enumerate(vals):
            if v is None:
                ws.write_blank(ri, ci, None, cell)
            elif isinstance(v, (int, float)):
                ws.write_number(ri, ci, v, pct if ci in (5, 14, 16, 17) else num)
            else:
                ws.write(ri, ci, v, sev_fmt if ci == 0 else cell)
        an = "\n".join(f"[{a['severity'].upper()}] {a['title']} – {a['detail']}" for a in rec["anomalies"]) or "Aucune anomalie"
        why = "\n".join(f"{SLOT_LABELS.get(k, 'Suppression')} : {v}" for k, v in rec["why"].items()) or "—"
        extra = [an, plan_text(rec, rec["plan"]), why, decision(rec, plan), plan_text(rec, plan), " ; ".join(info["steps"]) or "—",
                 ", ".join(info["new_columns"]) if info["new_columns"] else "(supprimée)" if info["dropped"] else "—",
                 info["imputed"], info["clipped"], info["removed_rows"], " ; ".join(info["warnings"]) or ""]
        for k, v in enumerate(extra):
            fmt = num if isinstance(v, int) else cell
            ws.write(ri, 18 + k, v, fmt)
        ws.set_row(ri, 15 * max(2, an.count("\n") + 1))
    ws.autofilter(0, 0, len(eda["features"]), len(cols) - 1)
    ws.conditional_format(1, 5, len(eda["features"]), 5, {"type": "data_bar", "bar_color": "#F2994A", "bar_solid": True})
    ws.conditional_format(1, 14, len(eda["features"]), 14, {"type": "3_color_scale", "min_color": "#63BE7B", "mid_color": "#FFFFFF", "max_color": "#F8696B", "mid_type": "num", "mid_value": 0})

    # ── Détail par slot ──
    wd = wb.add_worksheet("Détail transformations")
    for i, (c, w) in enumerate(zip(["Feature", "Étape", "Recommandé", "Choisi", "Appliqué ?", "Justification"], [24, 22, 34, 34, 12, 80])):
        wd.write(0, i, c, h); wd.set_column(i, i, w)
    r = 1
    for f in eda["features"]:
        n = f["name"]; rec = recs["features"][n]; plan = res["infos"][n]["plan"]
        steps = (["drop"] if rec["can_drop"] else []) + rec["slots"]
        for s in steps:
            rv, cv = rec["plan"].get(s), plan.get(s)
            if s == "drop":
                rl, cl, nm = ("Oui" if rv else "Non"), ("Oui" if cv else "Non"), "Suppression"
                applied = bool(cv)
            else:
                rl, cl, nm = _label(rec, s, rv), _label(rec, s, cv), SLOT_LABELS[s]
                applied = cv != "none" and not plan.get("drop")
            wd.write(r, 0, n, cell); wd.write(r, 1, nm, cell); wd.write(r, 2, rl, cell)
            wd.write(r, 3, cl, ok if rv == cv else warn); wd.write(r, 4, "Oui" if applied else "Non", cell)
            wd.write(r, 5, rec["why"].get(s, ""), cell); r += 1
    wd.freeze_panes(1, 1); wd.autofilter(0, 0, r - 1, 5)

    # ── Colonnes finales ──
    wa = wb.add_worksheet("Colonnes finales")
    heads = ["Colonne produite", "Feature d'origine", "Manquants", "Uniques", "Moyenne", "Écart-type", "Min", "Max", "Asymétrie"]
    for i, (c, w) in enumerate(zip(heads, [30, 24, 11, 11, 13, 13, 13, 13, 11])):
        wa.write(0, i, c, h); wa.set_column(i, i, w)
    origin = {c: n for n, i in res["infos"].items() for c in i["new_columns"]}
    r = 1
    for c, d in res["after_stats"].items():
        row = [c, origin.get(c, ""), d["missing"], d["unique"], d.get("mean"), d.get("std"), d.get("min"), d.get("max"), d.get("skew")]
        for i, v in enumerate(row):
            if v is None: wa.write_blank(r, i, None, cell)
            elif isinstance(v, str): wa.write(r, i, v, cell)
            else: wa.write_number(r, i, v, num)
        r += 1
    wa.freeze_panes(1, 1); wa.autofilter(0, 0, r - 1, len(heads) - 1)

    # ── Corrélations ──
    cr = eda["corr"]
    if cr["columns"]:
        wc = wb.add_worksheet("Corrélations")
        wc.set_column(0, 0, 24); wc.set_column(1, len(cr["columns"]), 11)
        for j, c in enumerate(cr["columns"]):
            wc.write(0, j + 1, c, h); wc.write(j + 1, 0, c, key)
            for i, v in enumerate(cr["matrix"][j]):
                wc.write_number(j + 1, i + 1, v, pct)
        n = len(cr["columns"])
        wc.conditional_format(1, 1, n, n, {"type": "3_color_scale", "min_color": "#3B82F6", "mid_color": "#FFFFFF", "max_color": "#EF4444",
                                           "min_type": "num", "min_value": -1, "mid_type": "num", "mid_value": 0, "max_type": "num", "max_value": 1})
    wb.close()
    return buf.getvalue()

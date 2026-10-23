"""SI table of the model panel (tab:llm_panel) with parameter counts, administered precision,
release date and Epoch Capabilities Index, from ../model_metadata.csv (every
value there carries its source URL; ECI downloaded 2026-10-22).

Conventions stated in the caption: release date = first public release of the
weights version administered (proprietary: API availability); parameter counts =
nominal totals as used in the analyses, with active parameters for MoE models;
precision = the precision at which each model was administered.

Writes results/tables/model_panel.{csv,tex}.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panel import MODEL_METADATA  # noqa: E402

TABLE_DIR = Path(__file__).resolve().parents[3] / "results/tables"
GROUPS = [("open", "Open-weight, 1--70B (local inference)"),
          ("large open", "Open-weight, $\\geq$300B (hosted, proprietary protocol)"),
          ("proprietary", "Proprietary (provider APIs)")]

m = MODEL_METADATA.copy()
m["sort_size"] = m["params_total_B"].fillna(0)
m = m.sort_values(["group", "family", "sort_size", "model_id"],
                  key=lambda c: c.map({g: i for i, (g, _) in enumerate(GROUPS)}) if c.name == "group" else c)

cols = ["group", "model_id", "family", "params_total_B", "params_active_B", "params_total_card_B",
        "params_total_safetensors_B", "precision_administered", "release_date",
        "release_date_basis", "eci"]
m[cols].to_csv(TABLE_DIR / "model_panel.csv", index=False)


def num(x, d=None):
    if pd.isna(x):
        return "--"
    x = float(x)
    return f"{x:g}" if d is None else f"{x:.{d}f}"


def tex_escape(s):
    return str(s).replace("_", "\\_").replace("&", "\\&")


lines = []
for g, title in GROUPS:
    lines.append(f"\\midrule\n\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
    for _, r in m[m["group"] == g].iterrows():
        moe = str(r["architecture"]).startswith("MoE")
        lines.append(" & ".join([
            # drop a parenthetical suffix from the display name (Grok's variant IDs)
            f"\\texttt{{{tex_escape(str(r['model_id']).split(' (')[0])}}}", tex_escape(r["family"]),
            num(r["params_total_B"]), num(r["params_active_B"]) if moe else "",
            tex_escape(r["precision_administered"]) if pd.notna(r["precision_administered"]) else "--",
            r["release_date"], num(r["eci"], 1),
        ]) + " \\\\")

caption = (
    f"The {len(m)} LLMs in the panel. Params (B): nominal total parameter count as used in the "
    "analyses; Active (B): active parameters per token for mixture-of-experts models "
    "(model-card and safetensors counts for every model are in the accompanying CSV). "
    "Precision: the precision at which each model was administered; each open-weight "
    "model was run at the precision its developers released it in, except GLM-5.3-Flash "
    "(fp8, although a bf16 release exists) and the two bloomz models (released in F16, "
    "run in bf16); gpt-oss-20b ships MXFP4 expert weights. "
    "Released: first public release of the weights version administered (proprietary "
    "models: API availability). ECI: Epoch Capabilities Index, downloaded 22 Oct 2026 "
    "(--: not scored). Parameter counts of proprietary models are not disclosed."
)
header = "Identifier & Family & Params (B) & Active (B) & Precision & Released & ECI \\\\"
tex = (
    # footnotesize and 3pt padding so the seven columns fit the SI text width
    "{\\footnotesize\\setlength{\\tabcolsep}{3pt}\\setlength{\\LTcapwidth}{\\textwidth}\n"
    "\\begin{longtable}{@{}llrrllr@{}}\n"
    f"\\caption{{{caption}}}\n\\label{{tab:llm_panel}} \\\\\n\\toprule\n{header}\n\\endfirsthead\n"
    f"\\multicolumn{{7}}{{c}}{{\\tablename~\\thetable{{}} (continued)}} \\\\\n\\toprule\n{header}\n\\endhead\n"
    "\\midrule \\multicolumn{7}{r}{\\textit{continued on next page}} \\\\\n\\endfoot\n"
    "\\bottomrule\n\\endlastfoot\n" + "\n".join(lines) + "\n\\end{longtable}\n}\n"
)
with open(TABLE_DIR / "model_panel.tex", "w", encoding="utf-8") as f:
    f.write(tex)
print(m.groupby("group").size().to_dict())

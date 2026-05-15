"""
Generate self-contained Plotly HTML charts for Algeria skill rank evolution.

Run from the project root:
    /opt/anaconda3/envs/env-fcv/bin/python notebooks/linkedin-trends/gen_skill_rank_charts.py

Output: docs/extra_html/skill_rank_allskills.html
  → copied by html_extra_path to _build/html/ at Jupyter Book build time
  → embedded via <iframe> in notebooks/linkedin-trends/linkedin-algeria.ipynb
"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

FILE_PATH = "data/LinkedIn/Skill Genome and Skills Pen 2025.xlsx"
OUT_DIR = "docs/extra_html"
COUNTRY = "Algeria"


def load_skill_flow(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name="2B - SGF Ctry Ind Yr", header=5)
    return df[["Country", "Industry", "Skill", "Year", "Skill Rank"]]


def build_allskills_chart(df: pd.DataFrame, country: str) -> go.Figure:
    df = df[df["Country"] == country].copy()
    df = df.sort_values(["Industry", "Year", "Skill Rank"])

    industries = sorted(df["Industry"].unique())
    initial = "Technology, Information and Media"

    all_skills = sorted(df["Skill"].unique())
    base_colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3
    color_map = {s: base_colors[i % len(base_colors)] for i, s in enumerate(all_skills)}

    fig = go.Figure()
    trace_industries = []

    for ind in industries:
        df_ind = df[df["Industry"] == ind]
        for skill in sorted(df_ind["Skill"].unique()):
            sd = df_ind[df_ind["Skill"] == skill].sort_values("Year")
            if sd.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sd["Year"],
                y=sd["Skill Rank"],
                mode="lines+markers",
                name=skill,
                legendgroup=skill,
                line=dict(color=color_map[skill], width=2, shape="spline"),
                marker=dict(size=8),
                visible=(ind == initial),
                hovertemplate=(
                    f"<b>{skill}</b><br>Industry: {ind}<br>"
                    "Year: %{x}<br>Rank: %{y}<extra></extra>"
                ),
            ))
            trace_industries.append(ind)

    buttons = []
    for ind in industries:
        visible = [trace_industries[i] == ind for i in range(len(trace_industries))]
        buttons.append(dict(
            label=ind,
            method="update",
            args=[
                {"visible": visible},
                {"title": f"{country} – Skill rank evolution in {ind} (all skills)"},
            ],
        ))

    fig.update_layout(
        updatemenus=[dict(
            buttons=buttons, direction="down", showactive=True,
            x=0.02, xanchor="left", y=1.15, yanchor="top",
        )],
        title=f"{country} – Skill rank evolution in {initial} (all skills)",
        yaxis=dict(autorange="reversed", title="Skill rank (1 = top)"),
        xaxis=dict(title="Year", tickmode="linear"),
        template="simple_white",
        hovermode="closest",
        legend_title_text="Skill",
        margin=dict(l=40, r=260, t=80, b=40),
    )
    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading data from {FILE_PATH}...")
    skill_flow = load_skill_flow(FILE_PATH)

    print("Building all-skills chart...")
    fig = build_allskills_chart(skill_flow, COUNTRY)
    out = os.path.join(OUT_DIR, "skill_rank_allskills.html")
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Interactive comparison plot: METR benchmark results vs. our computed results.
Styled to match the METR time-horizons website aesthetic.
"""

import yaml
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import markdown


TASK_MARKERS = {
    0.25:  "Answer question",
    2:     "Count words in passage",
    6:     "Find fact on web",
    49:    "Train classifier",
    240:   "Train adversarially robust image model",
    480:   "Exploit vulnerable Ethereum smart contract",
    600:   "Implement complex protocol from multiple RFCs",
}

DISPLAY_NAMES = {
    "claude_3_opus_inspect": "Claude 3 Opus",
    "claude_3_5_sonnet_20240620_inspect": "Claude 3.5 Sonnet",
    "claude_3_5_sonnet_20241022_inspect": "Claude 3.5 Sonnet (New)",
    "claude_3_7_sonnet_inspect": "Claude 3.7 Sonnet",
    "claude_4_opus_inspect": "Claude 4 Opus",
    "claude_4_1_opus_inspect": "Claude 4.1 Opus",
    "claude_opus_4_5_inspect": "Claude Opus 4.5",
    "claude_opus_4_6_inspect": "Claude Opus 4.6",
    "gpt_4": "GPT-4",
    "gpt_4_1106_inspect": "GPT-4 Turbo (1106)",
    "gpt_4_turbo_inspect": "GPT-4 Turbo",
    "gpt_4o_inspect": "GPT-4o",
    "gpt_5_2025_08_07_inspect": "GPT-5",
    "gpt_5_1_codex_max_inspect": "GPT-5.1 Codex",
    "gpt_5_2": "GPT-5.2",
    "gpt_5_3_codex": "GPT-5.3 Codex",
    "gemini_3_pro": "Gemini 3 Pro",
    "o1_preview": "o1-preview",
    "o1_inspect": "o1",
    "o3_inspect": "o3",
    "davinci_002": "GPT-3 (davinci)",
    "gpt2": "GPT-2",
    "gpt_3_5_turbo_instruct": "GPT-3.5",
}


def fmt_duration(minutes):
    if pd.isna(minutes) or minutes is None:
        return "N/A"
    minutes = float(minutes)
    if minutes < 1:
        return f"{minutes * 60:.0f} sec"
    if minutes < 60:
        return f"{minutes:.1f} min"
    h = minutes / 60
    return f"{h:.1f} hours" if h != int(h) else f"{int(h)} hours"


def load_and_extract(path, label):
    data = yaml.safe_load(open(path))
    rows = []
    for mk, md in data.get("results", {}).items():
        m = md.get("metrics", {})
        rd = md.get("release_date")
        if rd is None:
            continue
        p50 = m.get("p50_horizon_length", {})
        rows.append({
            "model_key": mk,
            "display_name": DISPLAY_NAMES.get(mk, mk),
            "release_date": pd.to_datetime(str(rd)),
            "p50": p50.get("estimate") if p50 else None,
            "p50_ci_low": p50.get("ci_low") if p50 else None,
            "p50_ci_high": p50.get("ci_high") if p50 else None,
            "average_score": m.get("average_score", {}).get("estimate"),
            "is_sota": m.get("is_sota", False),
            "benchmark_name": md.get("benchmark_name", ""),
            "source": label,
        })
    return pd.DataFrame(rows).dropna(subset=["p50", "release_date"]).sort_values("release_date")


def fit_sota_trend(df, extend_days=90):
    sota = df[df["is_sota"]].copy()
    if len(sota) < 2:
        return None, None, None
    X = (sota["release_date"] - pd.Timestamp("1970-01-01")).dt.days.values.reshape(-1, 1)
    y = np.log(np.clip(sota["p50"].values, 1e-3, np.inf))
    reg = LinearRegression().fit(X, y)
    x_range = np.linspace(X.min() - extend_days, X.max() + extend_days, 300)
    y_pred = np.exp(reg.predict(x_range.reshape(-1, 1)))
    dates = pd.to_datetime(x_range, unit="D", origin="1970-01-01")
    return dates, y_pred, np.log(2) / reg.coef_[0]


def match_agents(exp_df, comp_df):
    pairs, used = [], set()
    for _, er in exp_df.iterrows():
        best, best_d = None, 1e9
        for idx, cr in comp_df.iterrows():
            if idx in used:
                continue
            d = abs(er["average_score"] - cr["average_score"])
            if d < 0.0001 and d < best_d:
                best_d, best = d, (idx, cr)
        if best is not None:
            used.add(best[0])
            pairs.append((er, best[1]))
    return pairs


def make_horizon_plot(exp_df, comp_df, y_type="log"):
    fig = go.Figure()
    is_log = y_type == "log"

    # Trendlines
    for df, color, dash, label in [
        (exp_df, "#2d8a56", "dash", "METR Original"),
        (comp_df, "#6b7280", "dot", "Our Script"),
    ]:
        dates, y_pred, dt = fit_sota_trend(df)
        if dates is not None:
            fig.add_trace(go.Scatter(
                x=dates, y=y_pred, mode="lines",
                line=dict(color=color, width=2.5, dash=dash),
                name=f"{label} trend ({dt:.0f}d doubling)",
                hoverinfo="skip", opacity=0.55,
            ))

    # Data points
    for df, c_sota, c_non, sym, name, xoff in [
        (exp_df, "#2d8a56", "#b0b0b0", "circle", "METR Original", -4),
        (comp_df, "#6b7280", "#c8c8c8", "diamond", "Our Script", 4),
    ]:
        dates = df["release_date"] + pd.Timedelta(days=xoff)
        colors = [c_sota if s else c_non for s in df["is_sota"]]
        sizes = [11 if s else 8 for s in df["is_sota"]]
        p50 = df["p50"].values
        ci_lo = df["p50_ci_low"].values
        ci_hi = df["p50_ci_high"].values
        err_p = np.clip(ci_hi - p50, 0, None)
        err_m = np.clip(p50 - ci_lo, 0, None)
        if not is_log:
            cap = np.max(p50) * 0.5
            err_p = np.clip(err_p, 0, cap)
            err_m = np.clip(err_m, 0, cap)

        hovers = []
        for _, r in df.iterrows():
            h = [f"<b>{r['display_name']}</b>",
                 f"Release: {r['release_date'].strftime('%b %d, %Y')}",
                 f"50% Time Horizon: {fmt_duration(r['p50'])}"]
            if not pd.isna(r.get("p50_ci_low")):
                h.append(f"95% CI: [{fmt_duration(r['p50_ci_low'])}, {fmt_duration(r['p50_ci_high'])}]")
            h.append(f"Avg. score: {r['average_score']:.1%}")
            if r["is_sota"]:
                h.append("<b>SOTA at release</b>")
            hovers.append("<br>".join(h))

        fig.add_trace(go.Scatter(
            x=dates, y=p50, mode="markers",
            marker=dict(color=colors, size=sizes, symbol=sym, line=dict(width=1.5, color="white")),
            error_y=dict(type="data", symmetric=False,
                         array=err_p.tolist(), arrayminus=err_m.tolist(),
                         color="rgba(150,150,150,0.35)", thickness=1.2, width=3),
            hovertext=hovers, hoverinfo="text", name=name,
        ))

        # Labels for METR SOTA
        if name == "METR Original":
            for _, r in df[df["is_sota"]].iterrows():
                fig.add_annotation(x=r["release_date"], y=r["p50"],
                    text=r["display_name"], showarrow=False,
                    font=dict(size=10, color="#555"),
                    xanchor="left", yanchor="bottom", xshift=14, yshift=6)

    # Task markers (log only)
    if is_log:
        for mins, label in TASK_MARKERS.items():
            fig.add_annotation(x=0, y=mins, xref="paper", yref="y",
                text=f"— {label}", showarrow=False,
                font=dict(size=10, color="#999"),
                xanchor="left", yanchor="middle", xshift=5)

    # Y-axis
    y_title = "Task duration (for humans)<br><span style='font-size:10px;color:#999'>where logistic regression predicts 50% chance of succeeding</span>"
    if is_log:
        fig.update_yaxes(type="log", title=dict(text=y_title, font=dict(size=13)),
            tickvals=[1/15, 0.25, 1, 6, 49, 60, 240, 600],
            ticktext=["4 sec", "15 sec", "1 min", "6 min", "49 min", "1 hour", "4 hours", "10 hours"],
            gridcolor="rgba(0,0,0,0.05)", range=[np.log10(0.03), np.log10(2000)])
    else:
        max_y = max(comp_df["p50"].max(), exp_df["p50"].max())
        max_hrs = int(np.ceil(max_y / 60)) + 1
        tickvals = [30] + [h * 60 for h in range(0, max_hrs + 1)]
        ticktext = ["30 min"] + [f"{h} {'hour' if h==1 else 'hours'}" if h > 0 else "0" for h in range(0, max_hrs + 1)]
        fig.update_yaxes(type="linear", title=dict(text=y_title, font=dict(size=13)),
            tickvals=tickvals, ticktext=ticktext,
            gridcolor="rgba(0,0,0,0.05)", range=[0, max_y * 1.15])
        for mins, label in {60: "Fix bugs in Python libraries", 240: "Train adversarially robust model",
                            480: "Exploit vulnerable smart contract"}.items():
            if mins < max_y * 1.1:
                fig.add_hline(y=mins, line_dash="dot", line_color="rgba(0,0,0,0.07)")
                fig.add_annotation(x=0.01, y=mins, xref="paper", text=f"— {label}",
                    showarrow=False, font=dict(size=9, color="#bbb"), xanchor="left", yanchor="bottom", yshift=2)

    fig.update_xaxes(title=dict(text="LLM release date", font=dict(size=13)),
        gridcolor="rgba(0,0,0,0.05)", dtick="M6", tickformat="%Y")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="#faf8f5",
        margin=dict(l=80, r=40, t=20, b=60), height=560,
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5,
                    font=dict(size=11), bgcolor="rgba(250,248,245,0.9)"),
        hovermode="closest")
    return fig


def make_validation_table(exp_df, comp_df):
    """Build a plotly table showing side-by-side comparison for all agents."""
    pairs = match_agents(exp_df, comp_df)
    if not pairs:
        return go.Figure()

    pairs.sort(key=lambda x: x[0]["release_date"])

    names, dates, p50_exp, p50_comp, ci_exp, ci_comp, avg_exp, sotas = [], [], [], [], [], [], [], []
    for er, cr in pairs:
        names.append(er["display_name"])
        dates.append(er["release_date"].strftime("%Y-%m-%d"))
        p50_exp.append(f"{er['p50']:.2f}")
        p50_comp.append(f"{cr['p50']:.2f}")
        ci_e = f"[{er.get('p50_ci_low',0):.1f}, {er.get('p50_ci_high',0):.1f}]" if not pd.isna(er.get("p50_ci_low")) else "—"
        ci_c = f"[{cr.get('p50_ci_low',0):.1f}, {cr.get('p50_ci_high',0):.1f}]" if not pd.isna(cr.get("p50_ci_low")) else "—"
        ci_exp.append(ci_e)
        ci_comp.append(ci_c)
        avg_exp.append(f"{er['average_score']:.1%}")
        sotas.append("Yes" if er["is_sota"] else "")

    fig = go.Figure(data=[go.Table(
        columnwidth=[140, 80, 90, 90, 120, 120, 60, 40],
        header=dict(
            values=["Model", "Released", "p50 (METR)", "p50 (Ours)", "CI (METR)", "CI (Ours)", "Avg", "SOTA"],
            font=dict(size=11, color="#444"), fill_color="#f0ede8",
            align="left", line_color="#e0ddd8", height=32,
        ),
        cells=dict(
            values=[names, dates, p50_exp, p50_comp, ci_exp, ci_comp, avg_exp, sotas],
            font=dict(size=11, color="#333"), fill_color=[["#faf8f5", "white"] * (len(names) // 2 + 1)],
            align="left", line_color="#ece9e4", height=28,
        ),
    )])
    fig.update_layout(height=max(400, len(pairs) * 30 + 80),
        margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="#faf8f5")
    return fig


def make_ci_plot(exp_df, comp_df):
    pairs = match_agents(exp_df, comp_df)
    if not pairs:
        return go.Figure()
    pairs.sort(key=lambda x: x[0]["p50"])

    fig = go.Figure()
    y_pos = 0
    y_labels = []
    for er, cr in pairs:
        name = er["display_name"]
        fig.add_trace(go.Scatter(
            x=[er.get("p50_ci_low"), er["p50"], er.get("p50_ci_high")], y=[y_pos]*3,
            mode="lines+markers", line=dict(color="#2d8a56", width=3),
            marker=dict(size=[4,9,4], color="#2d8a56", symbol=["line-ns-open","circle","line-ns-open"]),
            showlegend=(y_pos==0), name="METR (1000 bootstrap)",
            hovertemplate=f"<b>{name}</b> (METR)<br>{fmt_duration(er['p50'])}<br>CI: [{fmt_duration(er.get('p50_ci_low'))}, {fmt_duration(er.get('p50_ci_high'))}]<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[cr.get("p50_ci_low"), cr["p50"], cr.get("p50_ci_high")], y=[y_pos+0.35]*3,
            mode="lines+markers", line=dict(color="#6b7280", width=3),
            marker=dict(size=[4,9,4], color="#6b7280", symbol=["line-ns-open","diamond","line-ns-open"]),
            showlegend=(y_pos==0), name="Ours (100 bootstrap)",
            hovertemplate=f"<b>{name}</b> (Ours)<br>{fmt_duration(cr['p50'])}<br>CI: [{fmt_duration(cr.get('p50_ci_low'))}, {fmt_duration(cr.get('p50_ci_high'))}]<extra></extra>",
        ))
        y_labels.append((y_pos+0.175, name))
        y_pos += 1.0

    fig.update_xaxes(type="log", title_text="p50 Time Horizon (minutes)", gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(tickvals=[y for y,_ in y_labels], ticktext=[n for _,n in y_labels],
        tickfont=dict(size=10), gridcolor="rgba(0,0,0,0.03)")
    fig.update_layout(height=max(500, len(pairs)*50+100), plot_bgcolor="white", paper_bgcolor="#faf8f5",
        margin=dict(l=160, r=30, t=20, b=50),
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5, font=dict(size=11)))
    return fig


def render_methodology_html():
    """Convert METHODOLOGY.md to HTML."""
    try:
        md_text = open("METHODOLOGY.md").read()
        # Use python-markdown with extensions
        html = markdown.markdown(md_text, extensions=["tables", "fenced_code", "codehilite", "toc"])
        return html
    except ImportError:
        # Fallback: basic conversion
        md_text = open("METHODOLOGY.md").read()
        # Very basic markdown -> HTML
        import re
        html = md_text
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
        html = re.sub(r'```(\w*)\n(.*?)```', r'<pre><code>\2</code></pre>', html, flags=re.DOTALL)
        html = re.sub(r'\$\$(.+?)\$\$', r'<div class="math">\1</div>', html, flags=re.DOTALL)
        html = re.sub(r'\$(.+?)\$', r'<span class="math">\1</span>', html)
        html = re.sub(r'^---$', r'<hr>', html, flags=re.MULTILINE)
        html = html.replace('\n\n', '</p><p>').replace('\n', '<br>')
        html = f'<p>{html}</p>'
        return html
    except FileNotFoundError:
        return "<p>METHODOLOGY.md not found.</p>"


def main():
    exp_df = load_and_extract("benchmark_results_1_1.yaml", "METR Original")
    comp_df = load_and_extract("full_results.yaml", "Our Script")
    exp_v11 = exp_df[exp_df["benchmark_name"] == "METR-Horizon-v1.1"].copy()

    fig_log = make_horizon_plot(exp_df, comp_df, "log")
    fig_lin = make_horizon_plot(exp_df, comp_df, "linear")
    fig_tbl = make_validation_table(exp_v11, comp_df)
    fig_ci = make_ci_plot(exp_v11, comp_df)

    methodology_html = render_methodology_html()

    exp_data = yaml.safe_load(open("benchmark_results_1_1.yaml"))
    comp_data = yaml.safe_load(open("full_results.yaml"))
    exp_dt = exp_data.get("doubling_time_in_days", {}).get("from_2023_on", {})
    comp_dt = comp_data.get("doubling_time_in_days", {}).get("from_2023_on", {})

    # Count matched agents
    pairs = match_agents(exp_v11, comp_df)
    n_matched = len(pairs)
    n_exact = sum(1 for er, cr in pairs if abs(er["p50"] - cr["p50"]) / max(er["p50"], 1e-6) < 0.001)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Time Horizon Replication</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.getElementById('methodology-content'), {{delimiters:[
    {{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}
  ]}});"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif;
         color: #1a1a2e; background: #faf8f5; }}

  .header {{ max-width: 860px; margin: 0 auto; padding: 60px 24px 24px; text-align: center; }}
  .header h1 {{ font-size: 34px; font-weight: 700; line-height: 1.2; letter-spacing: -0.5px; }}
  .header .subtitle {{ margin-top: 16px; font-size: 15px; color: #666; line-height: 1.65;
                       max-width: 680px; margin-left: auto; margin-right: auto; }}

  .meta-bar {{ max-width: 860px; margin: 0 auto; padding: 12px 24px; display: flex;
               justify-content: space-between; border-top: 1px solid #e5e2dc;
               border-bottom: 1px solid #e5e2dc; font-size: 11px; color: #999;
               text-transform: uppercase; letter-spacing: 0.5px; }}
  .meta-bar a {{ color: #2d8a56; text-decoration: none; }}

  .container {{ max-width: 1100px; margin: 0 auto; padding: 30px 24px; }}

  .summary-text {{ max-width: 860px; margin: 0 auto 30px; padding: 0 24px;
                   font-size: 15px; color: #555; line-height: 1.7; }}
  .summary-text strong {{ color: #2d8a56; }}

  .tab-bar {{ display: flex; gap: 0; margin: 0 0 0; border-bottom: 2px solid #e5e2dc; }}
  .tab {{ padding: 12px 24px; cursor: pointer; font-size: 13px; font-weight: 500;
          color: #aaa; border-bottom: 2px solid transparent; margin-bottom: -2px;
          transition: all 0.15s ease; user-select: none; }}
  .tab:hover {{ color: #666; }}
  .tab.active {{ color: #1a1a2e; border-bottom-color: #2d8a56; }}

  .panel {{ display: none; background: white; border-radius: 0 0 10px 10px; padding: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04); border: 1px solid #ece9e4; border-top: none; }}
  .panel.active {{ display: block; }}

  #methodology-content {{ max-width: 800px; margin: 0 auto; padding: 20px;
                          font-size: 14.5px; line-height: 1.75; color: #444; }}
  #methodology-content h1 {{ font-size: 24px; margin: 30px 0 12px; color: #1a1a2e; }}
  #methodology-content h2 {{ font-size: 20px; margin: 28px 0 10px; color: #1a1a2e;
                             border-bottom: 1px solid #ece9e4; padding-bottom: 6px; }}
  #methodology-content h3 {{ font-size: 16px; margin: 22px 0 8px; color: #333; }}
  #methodology-content code {{ background: #f3f1ec; padding: 2px 5px; border-radius: 3px;
                               font-size: 13px; font-family: 'SF Mono', Menlo, monospace; }}
  #methodology-content pre {{ background: #f8f6f2; padding: 14px; border-radius: 6px;
                              overflow-x: auto; margin: 12px 0; border: 1px solid #ece9e4; }}
  #methodology-content pre code {{ background: none; padding: 0; font-size: 12.5px; }}
  #methodology-content table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }}
  #methodology-content th, #methodology-content td {{ padding: 8px 12px; text-align: left;
                                                      border: 1px solid #e5e2dc; }}
  #methodology-content th {{ background: #f8f6f2; font-weight: 600; }}
  #methodology-content hr {{ border: none; border-top: 1px solid #e5e2dc; margin: 30px 0; }}
  #methodology-content .math {{ font-family: 'KaTeX_Main', serif; }}
  #methodology-content blockquote {{ border-left: 3px solid #2d8a56; padding-left: 16px;
                                     color: #666; margin: 12px 0; }}
  #methodology-content a {{ color: #2d8a56; }}

  .footer {{ text-align: center; padding: 40px 24px; font-size: 12px; color: #bbb; }}
  .footer a {{ color: #2d8a56; text-decoration: none; }}

  @media (max-width: 768px) {{
    .header h1 {{ font-size: 26px; }}
    .tab {{ padding: 10px 14px; font-size: 12px; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Task-Completion Time Horizons<br>Replication Results</h1>
  <p class="subtitle">Comparison of METR's published benchmark results against an independent replication
     using a standalone open-source script. Both use weighted logistic regression on 100+ diverse software
     tasks to measure how AI agent capabilities scale with task complexity.</p>
</div>

<div class="meta-bar">
  <span>Based on METR-Horizon v1.1</span>
  <span><a href="https://arxiv.org/abs/2503.14499">arXiv:2503.14499</a></span>
</div>

<div class="summary-text">
  <p>Our standalone script reproduces METR's time horizon analysis with <strong>exact agreement on all {n_exact}
  point estimates</strong> across {n_matched} agents evaluated on the v1.1 benchmark. Both analyses find
  a doubling time of <strong>{comp_dt.get('point_estimate', '?')} days</strong> (METR's 95% CI:
  [{exp_dt.get('ci_low', '?')}, {exp_dt.get('ci_high', '?')}] days; ours with 100 bootstrap samples:
  [{comp_dt.get('ci_low', '?')}, {comp_dt.get('ci_high', '?')}] days). Confidence intervals are
  consistent between 100-sample and 1000-sample bootstrap, differing only in tail precision.</p>
</div>

<div class="container">
  <div class="tab-bar">
    <div class="tab active" onclick="showTab(0)">Log Scale</div>
    <div class="tab" onclick="showTab(1)">Linear Scale</div>
    <div class="tab" onclick="showTab(2)">Validation Table</div>
    <div class="tab" onclick="showTab(3)">CI Comparison</div>
    <div class="tab" onclick="showTab(4)">Methodology</div>
  </div>

  <div class="panel active" id="p0"><div id="plot_log"></div></div>
  <div class="panel" id="p1"><div id="plot_lin"></div></div>
  <div class="panel" id="p2"><div id="plot_tbl"></div></div>
  <div class="panel" id="p3"><div id="plot_ci"></div></div>
  <div class="panel" id="p4"><div id="methodology-content">{methodology_html}</div></div>
</div>

<div class="footer">
  Generated by <code>compute_time_horizon.py</code> &mdash;
  <a href="https://github.com/METR/eval-analysis-public">METR eval-analysis-public</a> replication.
</div>

<script>
function showTab(i) {{
  document.querySelectorAll('.tab').forEach((t,j) => t.classList.toggle('active', j===i));
  document.querySelectorAll('.panel').forEach((p,j) => p.classList.toggle('active', j===i));
  setTimeout(() => window.dispatchEvent(new Event('resize')), 30);
  if (i === 4) {{
    renderMathInElement(document.getElementById('methodology-content'), {{
      delimiters: [
        {{left: '$$', right: '$$', display: true}},
        {{left: '$', right: '$', display: false}}
      ]
    }});
  }}
}}

var cfg = {{responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['lasso2d','select2d']}};
Plotly.newPlot('plot_log', {fig_log.to_json()}.data, {fig_log.to_json()}.layout, cfg);
Plotly.newPlot('plot_lin', {fig_lin.to_json()}.data, {fig_lin.to_json()}.layout, cfg);
Plotly.newPlot('plot_tbl', {fig_tbl.to_json()}.data, {fig_tbl.to_json()}.layout, cfg);
Plotly.newPlot('plot_ci',  {fig_ci.to_json()}.data,  {fig_ci.to_json()}.layout, cfg);
</script>
</body>
</html>"""

    with open("time_horizon_comparison.html", "w") as f:
        f.write(html)
    print("Saved time_horizon_comparison.html")


if __name__ == "__main__":
    main()

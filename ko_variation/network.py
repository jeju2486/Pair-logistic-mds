"""Undirected gene covariation summaries and standalone visual exports."""
from __future__ import annotations

import html
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .postprocess import _require, annotate_pairs


EDGE_COLUMNS = ["gene_a", "gene_b", "n_locus_pairs", "min_significance", "significance_column", "representative_u",
                "representative_v", "representative_raw_odds_ratio", "raw_direction", "locus_pairs"]
NODE_COLUMNS = ["gene", "label", "product", "group", "n_loci", "degree", "n_supporting_pairs"]


def build_gene_network(signals: pd.DataFrame, annotation: pd.DataFrame | None = None,
                       *, significance_column: str = "q_bh", include_self: bool = False,
                       missing_genes: str = "error") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate unique locus pairs per unordered gene pair, without pooling ORs.

    The representative is the smallest-significance pair (ties: u, v). Its OR
    remains a locus effect. min_significance is descriptive, not a gene-level test.
    """
    if missing_genes not in {"error", "drop"}:
        raise ValueError("missing_genes must be error or drop")
    data = annotate_pairs(signals, annotation) if annotation is not None else signals.copy()
    _require(data, ["u", "v", "u_gene", "v_gene", significance_column, "raw_odds_ratio", "raw_log_odds_ratio"])
    if data.duplicated(["u", "v"]).any():
        raise ValueError("Duplicate locus pairs cannot be counted as independent support")
    missing = data.u_gene.isna() | data.v_gene.isna() | data.u_gene.astype(str).str.strip().eq("") | data.v_gene.astype(str).str.strip().eq("")
    if missing.any() and missing_genes == "error":
        raise ValueError("Missing gene annotations; supply mappings or choose missing_genes='drop'")
    data = data.loc[~missing].copy()
    data["u_gene"] = data.u_gene.astype(str)
    data["v_gene"] = data.v_gene.astype(str)
    sig = pd.to_numeric(data[significance_column], errors="raise")
    if not np.isfinite(sig).all() or ((sig < 0) | (sig > 1)).any():
        raise ValueError("Network significance values must be finite and in [0, 1]")
    data[significance_column] = sig
    data["gene_a"] = [min(a, b) for a, b in zip(data.u_gene, data.v_gene)]
    data["gene_b"] = [max(a, b) for a, b in zip(data.u_gene, data.v_gene)]
    if not include_self:
        data = data.loc[data.gene_a.ne(data.gene_b)]
    edges = []
    for (a, b), block in data.groupby(["gene_a", "gene_b"], sort=True):
        block = block.sort_values([significance_column, "u", "v"], kind="stable")
        representative = block.iloc[0]
        effects = pd.to_numeric(block.raw_log_odds_ratio, errors="raise").to_numpy(float)
        direction = ("mixed" if (effects > 0).any() and (effects < 0).any() else
                     "positive" if (effects > 0).any() else "negative" if (effects < 0).any() else
                     "neutral" if (effects == 0).any() else "unknown")
        edges.append([a, b, len(block), float(representative[significance_column]), significance_column,
                      int(representative.u), int(representative.v), float(representative.raw_odds_ratio),
                      direction, json.dumps([[int(row.u), int(row.v)] for row in block.itertuples()])])
    edge_frame = pd.DataFrame(edges, columns=EDGE_COLUMNS)
    nodes = []
    for gene in sorted(set(edge_frame.gene_a) | set(edge_frame.gene_b)):
        mapped = []
        loci = set()
        for side in ("u", "v"):
            block = data.loc[data[f"{side}_gene"].eq(gene)]
            loci.update(block[side].tolist())
            mapped.append(block)
        attributes = {}
        for attribute in ("label", "product", "group"):
            values = set()
            for side, block in zip(("u", "v"), mapped):
                column = f"{side}_{attribute}"
                if column in block:
                    values.update(str(v) for v in block[column].dropna() if str(v).strip())
            attributes[attribute] = "; ".join(sorted(values)) or (gene if attribute == "label" else "")
        incident = edge_frame.loc[edge_frame.gene_a.eq(gene) | edge_frame.gene_b.eq(gene)]
        neighbours = (set(incident.gene_a) | set(incident.gene_b)) - {gene}
        nodes.append([gene, attributes["label"], attributes["product"], attributes["group"],
                      len(loci), len(neighbours), int(incident.n_locus_pairs.sum())])
    return pd.DataFrame(nodes, columns=NODE_COLUMNS), edge_frame


def export_gene_network(nodes: pd.DataFrame, edges: pd.DataFrame, prefix: str | Path,
                        *, seed: int = 42, dpi: int = 300, title: str = "Distal gene covariation") -> dict[str, Path]:
    """Export shared deterministic layout, 300-DPI PNG, offline interactive HTML and tables."""
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise ImportError("Network exports require pip install -e '.[network]'") from exc
    if dpi < 72:
        raise ValueError("dpi must be at least 72")
    prefix = Path(prefix)
    paths = {kind: Path(str(prefix) + suffix) for kind, suffix in
             [("png", ".png"), ("html", ".html"), ("nodes", ".nodes.tsv"), ("edges", ".edges.tsv")]}
    prefix.parent.mkdir(parents=True, exist_ok=True)
    graph = nx.Graph()
    graph.add_nodes_from(nodes.gene)
    graph.add_edges_from(zip(edges.gene_a, edges.gene_b))
    positions = nx.spring_layout(graph, seed=seed, weight=None)
    colors = {"positive": "#b2182b", "negative": "#2166ac", "mixed": "#762a83", "neutral": "#777777", "unknown": "#777777"}
    groups = sorted(set(nodes.group))
    palette = plt.get_cmap("tab20")
    group_colors = {group: matplotlib.colors.to_hex(palette(i % 20)) for i, group in enumerate(groups)}
    fig, ax = plt.subplots(figsize=(max(8, min(20, 6 + np.sqrt(len(nodes)))), 8))
    if len(nodes):
        nx.draw_networkx_nodes(graph, positions, ax=ax, node_size=[200 + 100 * np.sqrt(d) for d in nodes.degree],
                               node_color=[group_colors[g] for g in nodes.group], edgecolors="white")
        nx.draw_networkx_edges(graph, positions, ax=ax,
                               edgelist=list(zip(edges.gene_a, edges.gene_b)),
                               edge_color=[colors[d] for d in edges.raw_direction],
                               width=[1 + np.log2(n) for n in edges.n_locus_pairs], alpha=0.75)
        nx.draw_networkx_labels(graph, positions, labels=dict(zip(nodes.gene, nodes.label)), ax=ax, font_size=8)
        ax.margins(0.2)
    else:
        ax.text(0.5, 0.5, "No gene pairs pass selection", ha="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.axis("off")
    handles = [Line2D([0], [0], color=c, label=d) for d, c in colors.items() if d in set(edges.raw_direction)]
    handles += [Line2D([0], [0], marker="o", linestyle="", color=c, label="Group: " + (g or "unassigned")) for g, c in group_colors.items()]
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1), frameon=False, fontsize=8)
    fig.text(0.02, 0.02, "Node size: degree | Edge width: locus-pair count | Edge color: raw OR direction\nCovariation candidates; edge minima are not gene-level p-values.", fontsize=8)
    fig.savefig(paths["png"], dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    payload_nodes = []
    for row in nodes.to_dict("records"):
        x, y = positions[row["gene"]]
        payload_nodes.append(dict(row, x=float(500 + 300 * x), y=float(400 - 300 * y), color=group_colors[row["group"]]))
    payload_edges = []
    for row in edges.to_dict("records"):
        # JSON has no Infinity/NaN: display boundary ORs as strings.
        value = row["representative_raw_odds_ratio"]
        if not np.isfinite(value):
            row["representative_raw_odds_ratio"] = str(value)
        payload_edges.append(dict(row, color=colors[row["raw_direction"]]))
    payload = json.dumps({"nodes": payload_nodes, "edges": payload_edges}, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>__TITLE__</title>
<style>body{font:15px system-ui;margin:20px;background:#fafafa}svg{width:100%;height:75vh;background:white;border:1px solid #ddd;touch-action:none}#details{white-space:pre-wrap}button,input{padding:8px;margin:5px}text{font:12px system-ui}circle,line,path{cursor:pointer}</style>
<h1>__TITLE__</h1><p>Node size: degree; node color: annotation group. Edge width: supporting locus pairs. Edge color: raw OR direction (red positive, blue negative, purple mixed, grey neutral/unknown). Undirected covariation candidates; minimum significance is not a gene-level test.</p>
<label>Find gene or label <input id="search"></label><button id="reset">Reset view</button>
<svg id="canvas" viewBox="0 0 1000 800" role="img" aria-label="Interactive gene covariation network"><g id="scene"></g></svg>
<p>Scroll to zoom; drag to pan; select a node or edge for details. Hover for a summary. All supporting pairs are in the accompanying edge table.</p><pre id="details">Select a node or edge.</pre>
<script type="application/json" id="data">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),svg=document.getElementById('canvas'),scene=document.getElementById('scene'),ns='http://www.w3.org/2000/svg',byId=new Map(data.nodes.map(n=>[n.gene,n]));
function element(tag,attrs,parent=scene){const el=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))el.setAttribute(k,v);parent.appendChild(el);return el;}
function details(el,row){const info=Object.fromEntries(Object.entries(row).filter(([k])=>!['x','y','color'].includes(k)));const t=element('title',{},el);t.textContent=JSON.stringify(info,null,2);el.addEventListener('pointerdown',()=>document.getElementById('details').textContent=JSON.stringify(info,null,2));}
for(const e of data.edges){const a=byId.get(e.gene_a),b=byId.get(e.gene_b);let el;if(a===b){el=element('path',{d:`M ${a.x} ${a.y} c -45 -65 45 -65 0 0`,fill:'none',stroke:e.color,'stroke-width':1+Math.log2(e.n_locus_pairs)});}else{el=element('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:e.color,'stroke-width':1+Math.log2(e.n_locus_pairs)});}details(el,e);}
const displayed=[];for(const n of data.nodes){const g=element('g',{});element('circle',{cx:n.x,cy:n.y,r:8+3*Math.sqrt(n.degree),fill:n.color,stroke:'white'},g);const text=element('text',{x:n.x+15,y:n.y+4},g);text.textContent=n.label;details(g,n);displayed.push([g,n]);}
if(!data.nodes.length){const t=element('text',{x:350,y:400});t.textContent='No gene pairs pass selection';}
document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();for(const [g,n]of displayed)g.style.opacity=(n.gene+' '+n.label).toLowerCase().includes(q)?1:0.15;});
let view=[0,0,1000,800],drag=null;function update(){svg.setAttribute('viewBox',view.join(' '));}svg.addEventListener('wheel',e=>{e.preventDefault();const f=e.deltaY>0?1.15:1/1.15;const w=Math.min(10000,Math.max(100,view[2]*f)),h=w*0.8;view=[view[0]+(view[2]-w)/2,view[1]+(view[3]-h)/2,w,h];update();},{passive:false});
svg.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY,...view];svg.setPointerCapture(e.pointerId);});svg.addEventListener('pointermove',e=>{if(!drag)return;const r=svg.getBoundingClientRect(),scale=Math.max(view[2]/r.width,view[3]/r.height);view[0]=drag[2]-(e.clientX-drag[0])*scale;view[1]=drag[3]-(e.clientY-drag[1])*scale;update();});svg.addEventListener('pointerup',()=>drag=null);svg.addEventListener('pointercancel',()=>drag=null);
document.getElementById('reset').onclick=()=>{view=[0,0,1000,800];update();};
</script></html>'''
    paths["html"].write_text(document.replace("__TITLE__", html.escape(title)).replace("__DATA__", payload), encoding="utf-8")
    nodes.to_csv(paths["nodes"], sep="\t", index=False)
    edges.to_csv(paths["edges"], sep="\t", index=False)
    return paths

#!/usr/bin/env python3
"""Render the complete GitHub Docs source graph to a high-resolution PNG.

The graph contains every Markdown page under ``content/``.  Index pages are
identified by filename, routing edges come from ``children`` frontmatter, and
cross-link edges come from resolvable Markdown/HTML links.  NetworkX's
ForceAtlas2 implementation provides the large-graph layout.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from profile_github_docs_network import (
    HTML_LINK_RE,
    INLINE_LINK_RE,
    REFERENCE_LINK_RE,
    clean_route,
    normalize_target,
    resolve_child,
    resolve_route,
    route_for_file,
    split_frontmatter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path, help="Path to github/docs content/")
    parser.add_argument("output", type=Path, help="Destination PNG")
    parser.add_argument("--pixels", type=int, default=6000)
    parser.add_argument("--iterations", type=int, default=125)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def load_source_graph(root: Path) -> tuple[list[str], set[str], set[tuple[str, str]], set[tuple[str, str]]]:
    paths = sorted(root.rglob("*.md"))
    nodes = [route_for_file(path, root) for path in paths]
    routes = set(nodes)
    index_nodes = {
        route for path, route in zip(paths, nodes) if path.name == "index.md"
    }
    page_data: dict[str, tuple[Path, dict[str, Any], str]] = {}
    aliases: dict[str, str] = {}

    for path, route in zip(paths, nodes):
        text = path.read_text(encoding="utf-8")
        frontmatter, body = split_frontmatter(text)
        page_data[route] = (path, frontmatter, body)
        redirect_from = frontmatter.get("redirect_from", [])
        if isinstance(redirect_from, str):
            redirect_from = [redirect_from]
        if isinstance(redirect_from, list):
            for alias in redirect_from:
                if isinstance(alias, str) and "{%" not in alias:
                    aliases[clean_route(alias)] = route

    routing_edges: set[tuple[str, str]] = set()
    link_edges: set[tuple[str, str]] = set()
    for source, (path, frontmatter, body) in page_data.items():
        if path.name == "index.md":
            children = frontmatter.get("children", [])
            if isinstance(children, str):
                children = [children]
            if isinstance(children, list):
                for child in children:
                    if not isinstance(child, str):
                        continue
                    target = resolve_child(child, source, routes, aliases)
                    if target is not None and target != source:
                        routing_edges.add((source, target))

        raw_targets = INLINE_LINK_RE.findall(body)
        raw_targets.extend(REFERENCE_LINK_RE.findall(body))
        raw_targets.extend(HTML_LINK_RE.findall(body))
        for raw_target in raw_targets:
            stripped = raw_target.strip().strip("<>")
            parsed = urlsplit(stripped.split(" ", 1)[0])
            looks_internal = (
                (not parsed.scheme and not parsed.netloc)
                or parsed.netloc in {"docs.github.com", "help.github.com"}
            ) and not stripped.startswith(("#", "mailto:", "data:", "javascript:"))
            if not looks_internal or "{%" in stripped or "{{" in stripped:
                continue
            normalized = normalize_target(stripped, source)
            target = resolve_route(normalized, routes, aliases)
            if target is not None and target != source:
                link_edges.add((source, target))

    return nodes, index_nodes, routing_edges, link_edges


def normalized_positions(position: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = list(position)
    values = np.asarray([position[key] for key in keys], dtype=float)
    values -= np.median(values, axis=0)
    span = np.ptp(values, axis=0)
    scale = max(float(span.max()), 1e-9)
    values /= scale
    return {key: values[i] for i, key in enumerate(keys)}


def render(
    nodes: list[str],
    index_nodes: set[str],
    routing_edges: set[tuple[str, str]],
    link_edges: set[tuple[str, str]],
    output: Path,
    pixels: int,
    iterations: int,
    seed: int,
) -> None:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for source, target in link_edges:
        graph.add_edge(source, target, weight=1.0)
    for source, target in routing_edges:
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += 2.5
        else:
            graph.add_edge(source, target, weight=2.5)

    degrees = dict(graph.degree())
    masses = {node: 1.0 + math.log1p(degrees[node]) for node in nodes}
    positions = nx.forceatlas2_layout(
        graph,
        max_iter=iterations,
        jitter_tolerance=1.0,
        scaling_ratio=7.0,
        gravity=0.8,
        strong_gravity=True,
        node_mass=masses,
        weight="weight",
        linlog=True,
        seed=seed,
    )
    positions = normalized_positions(positions)

    routing_segments = np.asarray(
        [[positions[source], positions[target]] for source, target in routing_edges]
    )
    link_segments = np.asarray(
        [[positions[source], positions[target]] for source, target in link_edges]
    )

    dpi = 300
    side_inches = pixels / dpi
    fig, ax = plt.subplots(figsize=(side_inches, side_inches), dpi=dpi)
    background = "#0f141b"
    foreground = "#f5f7fa"
    muted = "#aab6c5"
    document_color = "#57a6d9"
    index_color = "#f2a65a"
    routing_color = "#ffb86c"
    link_color = "#8293a8"
    hub_color = "#f5d76e"
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    ax.add_collection(
        LineCollection(
            link_segments,
            colors=link_color,
            linewidths=0.16,
            alpha=0.075,
            zorder=1,
            rasterized=True,
        )
    )
    ax.add_collection(
        LineCollection(
            routing_segments,
            colors=routing_color,
            linewidths=0.30,
            alpha=0.22,
            zorder=2,
            rasterized=True,
        )
    )

    document_nodes = [node for node in nodes if node not in index_nodes]
    index_list = [node for node in nodes if node in index_nodes]
    document_xy = np.asarray([positions[node] for node in document_nodes])
    index_xy = np.asarray([positions[node] for node in index_list])
    document_sizes = np.asarray(
        [2.0 + 1.25 * math.log1p(degrees[node]) for node in document_nodes]
    )
    index_sizes = np.asarray(
        [11.0 + 2.6 * math.log1p(degrees[node]) for node in index_list]
    )
    ax.scatter(
        document_xy[:, 0],
        document_xy[:, 1],
        s=document_sizes,
        c=document_color,
        alpha=0.72,
        linewidths=0,
        zorder=3,
        rasterized=True,
    )
    ax.scatter(
        index_xy[:, 0],
        index_xy[:, 1],
        s=index_sizes,
        c=index_color,
        alpha=0.96,
        edgecolors=foreground,
        linewidths=0.12,
        zorder=4,
        rasterized=True,
    )

    top_hubs = sorted(nodes, key=lambda node: (-degrees[node], node))[:18]
    hub_xy = np.asarray([positions[node] for node in top_hubs])
    ax.scatter(
        hub_xy[:, 0],
        hub_xy[:, 1],
        s=[28 + 5 * math.log1p(degrees[node]) for node in top_hubs],
        facecolors="none",
        edgecolors=hub_color,
        linewidths=0.75,
        zorder=5,
    )
    for rank, node in enumerate(top_hubs):
        x, y = positions[node]
        dx = 7 if rank % 2 == 0 else -7
        horizontal = "left" if dx > 0 else "right"
        label = node or "/"
        if len(label) > 54:
            label = "…/" + label.rsplit("/", 1)[-1]
        ax.annotate(
            f"{label}  ·  k={degrees[node]}",
            (x, y),
            xytext=(dx, 5 + (rank % 3) * 4),
            textcoords="offset points",
            color=foreground,
            fontsize=5.8,
            fontweight="medium",
            ha=horizontal,
            va="bottom",
            bbox={"boxstyle": "round,pad=0.24", "fc": background, "ec": "none", "alpha": 0.80},
            zorder=6,
        )

    all_xy = np.asarray(list(positions.values()))
    xmin, ymin = all_xy.min(axis=0)
    xmax, ymax = all_xy.max(axis=0)
    padx = max((xmax - xmin) * 0.06, 0.02)
    pady = max((ymax - ymin) * 0.06, 0.02)
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    fig.suptitle(
        "GitHub Docs — complete source-link graph",
        color=foreground,
        fontsize=22,
        fontweight="medium",
        y=0.975,
    )
    fig.text(
        0.5,
        0.949,
        (
            f"{len(nodes):,} Markdown pages  ·  {len(index_nodes):,} index pages  ·  "
            f"{len(routing_edges):,} routing edges  ·  {len(link_edges):,} Markdown-link edges"
        ),
        ha="center",
        va="center",
        color=muted,
        fontsize=10,
    )
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=index_color, markeredgecolor=foreground, markersize=7, label="Index page"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=document_color, markersize=6, label="Article page"),
        Line2D([0], [0], color=routing_color, lw=1.5, label="ROUTES_TO (children)"),
        Line2D([0], [0], color=link_color, lw=1.2, label="LINKS_TO (Markdown)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=hub_color, markersize=8, label="Top-degree hub"),
    ]
    legend_artist = fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.024),
        ncol=5,
        frameon=False,
        fontsize=8.5,
        labelcolor=foreground,
    )
    for text in legend_artist.get_texts():
        text.set_color(foreground)
    fig.text(
        0.5,
        0.010,
        "Source graph only; chunks, extracted entities, relations, and communities require a real GraphRAG build.",
        ha="center",
        va="center",
        color=muted,
        fontsize=7.5,
    )
    fig.subplots_adjust(left=0.018, right=0.982, top=0.928, bottom=0.058)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=dpi,
        facecolor=background,
        edgecolor="none",
        metadata={
            "Title": "GitHub Docs complete source-link graph",
            "Description": (
                f"All {len(nodes)} pages, {len(routing_edges)} routing edges, "
                f"and {len(link_edges)} Markdown-link edges."
            ),
        },
    )
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = args.corpus.resolve()
    nodes, index_nodes, routing_edges, link_edges = load_source_graph(root)
    print(
        f"layout: nodes={len(nodes)} indexes={len(index_nodes)} "
        f"routing={len(routing_edges)} links={len(link_edges)}"
    )
    render(
        nodes,
        index_nodes,
        routing_edges,
        link_edges,
        args.output.resolve(),
        args.pixels,
        args.iterations,
        args.seed,
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

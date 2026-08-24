#!/usr/bin/env python3
"""Profile routing and cross-link networks in github/docs content.

The profiler deliberately keeps three graph definitions separate:

* routing: ``index.md`` -> frontmatter ``children``
* links: resolved internal Markdown/HTML links in article bodies
* combined: the union of routing and link edges

It emits JSON to stdout.  Power-law fitting is optional because the third-party
``powerlaw`` package is intentionally not a project dependency.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit

import yaml


INLINE_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
REFERENCE_LINK_RE = re.compile(r"(?m)^\s*\[[^\]]+\]:\s*(\S+)")
HTML_LINK_RE = re.compile(r"(?i)\bhref\s*=\s*['\"]([^'\"]+)['\"]")
LOCALE_RE = re.compile(r"^[a-z]{2}(?:-[A-Z]{2})?$")
VERSION_SEGMENT_RE = re.compile(
    r"^(?:free-pro-team|enterprise-cloud|enterprise-server)@[^/]+$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path, help="Path to github/docs content/")
    parser.add_argument(
        "--powerlaw-path",
        type=Path,
        help="Optional directory containing the third-party powerlaw package",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=100,
        help="Parametric bootstrap samples for the power-law KS p-value",
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Omit full degree histograms and long unresolved-target lists",
    )
    return parser.parse_args()


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, text
    closing = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            closing = i
            break
    if closing is None:
        return {}, text
    try:
        parsed = yaml.safe_load("".join(lines[1:closing])) or {}
    except yaml.YAMLError:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}, "".join(lines[closing + 1 :])


def route_for_file(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    if relative == "index.md":
        return ""
    if relative.endswith("/index.md"):
        return relative[: -len("/index.md")]
    if relative.endswith(".md"):
        return relative[:-3]
    raise ValueError(relative)


def clean_route(path: str) -> str:
    path = unquote(path).replace("\\", "/")
    path = re.sub(r"/+", "/", path).strip("/")
    if path.endswith(".md"):
        path = path[:-3]
    if path.endswith("/index"):
        path = path[:-6]
    parts = [part for part in path.split("/") if part]
    if parts and LOCALE_RE.match(parts[0]):
        parts.pop(0)
    if parts and VERSION_SEGMENT_RE.match(parts[0]):
        parts.pop(0)
    # Frontmatter children may use repository-rooted `/content/...` paths even
    # though this profiler's graph root is already the content directory.
    if parts and parts[0] == "content":
        parts.pop(0)
    return "/".join(parts).strip("/")


def normalize_target(target: str, source_route: str) -> str | None:
    target = target.strip().strip("<>")
    # Markdown permits an optional quoted title after the destination.
    if " " in target and not target.startswith("data:"):
        target = target.split(" ", 1)[0]
    if not target or target.startswith(('#', 'mailto:', 'data:', 'javascript:')):
        return None
    if "{%" in target or "{{" in target:
        return None

    parsed = urlsplit(target)
    if parsed.scheme in {"http", "https"}:
        if parsed.netloc not in {"docs.github.com", "help.github.com"}:
            return None
        raw_path = parsed.path
    elif parsed.scheme:
        return None
    else:
        raw_path = parsed.path

    if not raw_path:
        return None
    if raw_path.startswith("/"):
        return clean_route(raw_path)

    source_parent = source_route.rsplit("/", 1)[0] if "/" in source_route else ""
    stack = [p for p in source_parent.split("/") if p]
    for part in raw_path.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if stack:
                stack.pop()
        else:
            stack.append(part)
    return clean_route("/".join(stack))


def resolve_route(
    normalized: str | None,
    routes: set[str],
    aliases: dict[str, str],
) -> str | None:
    if normalized is None:
        return None
    if normalized in routes:
        return normalized
    if normalized in aliases:
        return aliases[normalized]
    return None


def resolve_child(
    child: str,
    source_route: str,
    routes: set[str],
    aliases: dict[str, str],
) -> str | None:
    child_clean = clean_route(child)
    relative_candidate = clean_route(f"{source_route}/{child_clean}")
    resolved = resolve_route(relative_candidate, routes, aliases)
    if resolved is not None:
        return resolved
    return resolve_route(child_clean, routes, aliases)


def percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def gini(values: list[int]) -> float:
    if not values or sum(values) == 0:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    weighted = sum((i + 1) * value for i, value in enumerate(ordered))
    return (2 * weighted) / (n * sum(ordered)) - (n + 1) / n


def degree_summary(values: list[int]) -> dict[str, Any]:
    ordered = sorted(values)
    n = len(values)
    mean = statistics.fmean(values) if values else 0.0
    second = statistics.fmean(value * value for value in values) if values else 0.0
    histogram = Counter(values)
    return {
        "mean": mean,
        "median": percentile(ordered, 0.5),
        "p90": percentile(ordered, 0.9),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "max": max(values, default=0),
        "gini": gini(values),
        "second_moment": second,
        "edge_arrival_expected_degree": second / mean if mean else 0.0,
        "p0": histogram.get(0, 0) / n if n else 0.0,
        "p1": histogram.get(1, 0) / n if n else 0.0,
        "p_le_2": sum(count for k, count in histogram.items() if k <= 2) / n if n else 0.0,
        "p_ge_10": sum(count for k, count in histogram.items() if k >= 10) / n if n else 0.0,
        "histogram": {str(k): histogram[k] for k in sorted(histogram)},
    }


def undirected_adjacency(nodes: Iterable[str], edges: set[tuple[str, str]]) -> dict[str, set[str]]:
    adjacency = {node: set() for node in nodes}
    for source, target in edges:
        if source == target:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)
    return adjacency


def component_sizes(adjacency: dict[str, set[str]]) -> list[int]:
    seen: set[str] = set()
    sizes: list[int] = []
    for start in adjacency:
        if start in seen:
            continue
        queue = [start]
        seen.add(start)
        size = 0
        while queue:
            node = queue.pop()
            size += 1
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def core_numbers(adjacency: dict[str, set[str]]) -> dict[str, int]:
    # Batagelj-Zaversnik style peeling implemented with a min-heap.
    import heapq

    degrees = {node: len(neighbors) for node, neighbors in adjacency.items()}
    heap = [(degree, node) for node, degree in degrees.items()]
    heapq.heapify(heap)
    removed: set[str] = set()
    cores: dict[str, int] = {}
    while heap:
        degree, node = heapq.heappop(heap)
        if node in removed or degree != degrees[node]:
            continue
        removed.add(node)
        cores[node] = degree
        for neighbor in adjacency[node]:
            # A neighbor already at this shell's degree remains in the shell;
            # only larger residual degrees are reduced.
            if neighbor not in removed and degrees[neighbor] > degree:
                degrees[neighbor] -= 1
                heapq.heappush(heap, (degrees[neighbor], neighbor))
    return cores


def clustering_stats(adjacency: dict[str, set[str]]) -> dict[str, float]:
    triples = 0
    triangle_times_three = 0
    local_sum = 0.0
    local_count = 0
    for node, neighbors in adjacency.items():
        degree = len(neighbors)
        if degree < 2:
            continue
        possible = degree * (degree - 1) // 2
        triples += possible
        connected_pairs_twice = sum(len(adjacency[n] & neighbors) for n in neighbors)
        connected_pairs = connected_pairs_twice // 2
        triangle_times_three += connected_pairs
        local_sum += connected_pairs / possible
        local_count += 1
    return {
        "transitivity": triangle_times_three / triples if triples else 0.0,
        "average_clustering_nontrivial": local_sum / local_count if local_count else 0.0,
    }


def graph_summary(nodes: list[str], edges: set[tuple[str, str]]) -> dict[str, Any]:
    out_neighbors = {node: set() for node in nodes}
    in_neighbors = {node: set() for node in nodes}
    for source, target in edges:
        if source == target:
            continue
        out_neighbors[source].add(target)
        in_neighbors[target].add(source)
    adjacency = undirected_adjacency(nodes, edges)
    out_degree = [len(out_neighbors[node]) for node in nodes]
    in_degree = [len(in_neighbors[node]) for node in nodes]
    total_degree = [len(adjacency[node]) for node in nodes]
    components = component_sizes(adjacency)
    cores = core_numbers(adjacency)
    max_core = max(cores.values(), default=0)
    top_nodes = sorted(
        (
            {
                "route": node or "/",
                "degree": len(adjacency[node]),
                "in_degree": len(in_neighbors[node]),
                "out_degree": len(out_neighbors[node]),
            }
            for node in nodes
        ),
        key=lambda item: (-item["degree"], item["route"]),
    )[:20]
    result = {
        "nodes": len(nodes),
        "directed_edges": sum(len(v) for v in out_neighbors.values()),
        "undirected_edges": sum(len(v) for v in adjacency.values()) // 2,
        "density_undirected": (
            (sum(len(v) for v in adjacency.values())) / (len(nodes) * (len(nodes) - 1))
            if len(nodes) > 1
            else 0.0
        ),
        "out_degree": degree_summary(out_degree),
        "in_degree": degree_summary(in_degree),
        "total_degree": degree_summary(total_degree),
        "components": len(components),
        "largest_component_nodes": components[0] if components else 0,
        "largest_component_fraction": components[0] / len(nodes) if components and nodes else 0.0,
        "isolates": sum(1 for degree in total_degree if degree == 0),
        "max_core": max_core,
        "fraction_core_ge_2": sum(1 for value in cores.values() if value >= 2) / len(nodes),
        "fraction_core_ge_3": sum(1 for value in cores.values() if value >= 3) / len(nodes),
        "top_nodes": top_nodes,
    }
    result.update(clustering_stats(adjacency))
    result["_degrees_for_fit"] = total_degree
    return result


def powerlaw_profile(
    values: list[int],
    powerlaw_module: Any,
    bootstrap_samples: int,
    rng: random.Random,
) -> dict[str, Any]:
    positive = [int(value) for value in values if value > 0]
    if len(positive) < 10:
        return {"status": "insufficient_data", "positive_nodes": len(positive)}

    fit = powerlaw_module.Fit(positive, discrete=True, verbose=False)
    xmin = int(fit.power_law.xmin)
    empirical_d = float(fit.power_law.D)
    tail = [value for value in positive if value >= xmin]
    body = [value for value in positive if value < xmin]

    bootstrap_ds: list[float] = []
    for _ in range(max(0, bootstrap_samples)):
        synthetic_tail_raw = fit.power_law.generate_random(len(tail))
        synthetic_tail = [max(xmin, int(round(float(value)))) for value in synthetic_tail_raw]
        synthetic_body = [rng.choice(body) for _ in body] if body else []
        synthetic = synthetic_body + synthetic_tail
        synthetic_fit = powerlaw_module.Fit(synthetic, discrete=True, verbose=False)
        bootstrap_ds.append(float(synthetic_fit.power_law.D))

    comparisons: dict[str, Any] = {}
    for alternative in ("lognormal", "exponential", "truncated_power_law"):
        ratio, p_value = fit.distribution_compare("power_law", alternative)
        comparisons[alternative] = {
            "loglikelihood_ratio": float(ratio),
            "p_value": float(p_value),
            "favored": "power_law" if ratio > 0 else alternative,
        }

    bootstrap_p = (
        sum(value >= empirical_d for value in bootstrap_ds) / len(bootstrap_ds)
        if bootstrap_ds
        else None
    )
    significantly_worse = [
        name
        for name, comparison in comparisons.items()
        if comparison["loglikelihood_ratio"] < 0 and comparison["p_value"] < 0.1
    ]
    if bootstrap_p is not None and bootstrap_p < 0.1:
        conclusion = "power_law_rejected"
    elif len(tail) < 50:
        conclusion = "tail_too_small"
    elif significantly_worse:
        conclusion = "alternative_significantly_better"
    else:
        conclusion = "power_law_tail_plausible_not_proven"

    return {
        "status": "ok",
        "conclusion": conclusion,
        "positive_nodes": len(positive),
        "alpha": float(fit.power_law.alpha),
        "xmin": xmin,
        "ks_distance": empirical_d,
        "tail_nodes": len(tail),
        "tail_fraction_all_nodes": len(tail) / len(values),
        "bootstrap_samples": len(bootstrap_ds),
        "bootstrap_p_value": bootstrap_p,
        "comparisons": comparisons,
    }


def main() -> int:
    args = parse_args()
    root = args.corpus.resolve()
    paths = sorted(root.rglob("*.md"))
    nodes = [route_for_file(path, root) for path in paths]
    routes = set(nodes)
    page_data: dict[str, tuple[Path, dict[str, Any], str]] = {}
    aliases: dict[str, str] = {}
    parse_failures = 0

    for path, route in zip(paths, nodes):
        text = path.read_text(encoding="utf-8")
        frontmatter, body = split_frontmatter(text)
        if not frontmatter and text.startswith("---"):
            parse_failures += 1
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
    routing_refs = routing_resolved = 0
    internal_link_refs = internal_link_resolved = 0
    unresolved_routing_targets: Counter[str] = Counter()
    unresolved_internal_targets: Counter[str] = Counter()

    for source, (path, frontmatter, body) in page_data.items():
        if path.name == "index.md":
            children = frontmatter.get("children", [])
            if isinstance(children, str):
                children = [children]
            if isinstance(children, list):
                for child in children:
                    if not isinstance(child, str):
                        continue
                    routing_refs += 1
                    target = resolve_child(child, source, routes, aliases)
                    if target is not None and target != source:
                        routing_resolved += 1
                        routing_edges.add((source, target))
                    else:
                        unresolved_routing_targets[f"{source or '/'} -> {child}"] += 1

        raw_targets = INLINE_LINK_RE.findall(body)
        raw_targets.extend(REFERENCE_LINK_RE.findall(body))
        raw_targets.extend(HTML_LINK_RE.findall(body))
        for raw_target in raw_targets:
            stripped = raw_target.strip().strip("<>")
            parsed = urlsplit(stripped.split(" ", 1)[0])
            looks_internal = (
                (not parsed.scheme and not parsed.netloc)
                or parsed.netloc in {"docs.github.com", "help.github.com"}
            ) and not stripped.startswith(('#', 'mailto:', 'data:', 'javascript:'))
            if not looks_internal or "{%" in stripped or "{{" in stripped:
                continue
            internal_link_refs += 1
            normalized = normalize_target(stripped, source)
            target = resolve_route(normalized, routes, aliases)
            if target is not None:
                internal_link_resolved += 1
                if target != source:
                    link_edges.add((source, target))
            elif normalized:
                unresolved_internal_targets[normalized] += 1

    graphs = {
        "routing": graph_summary(nodes, routing_edges),
        "cross_links": graph_summary(nodes, link_edges),
        "combined": graph_summary(nodes, routing_edges | link_edges),
    }

    if args.powerlaw_path:
        sys.path.insert(0, str(args.powerlaw_path.resolve()))
        import powerlaw  # type: ignore

        rng = random.Random(args.seed)
        for graph in graphs.values():
            graph["powerlaw_fit_total_degree"] = powerlaw_profile(
                graph.pop("_degrees_for_fit"),
                powerlaw,
                args.bootstrap_samples,
                rng,
            )
    else:
        for graph in graphs.values():
            graph.pop("_degrees_for_fit")

    result = {
        "corpus": str(root),
        "markdown_pages": len(nodes),
        "frontmatter_parse_failures": parse_failures,
        "routing_resolution": {
            "references": routing_refs,
            "resolved": routing_resolved,
            "rate": routing_resolved / routing_refs if routing_refs else 0.0,
            "top_unresolved": unresolved_routing_targets.most_common(20),
        },
        "internal_link_resolution": {
            "references": internal_link_refs,
            "resolved": internal_link_resolved,
            "rate": internal_link_resolved / internal_link_refs if internal_link_refs else 0.0,
            "top_unresolved": unresolved_internal_targets.most_common(20),
        },
        "graphs": graphs,
    }
    if args.compact:
        for graph in result["graphs"].values():
            for degree_kind in ("in_degree", "out_degree", "total_degree"):
                graph[degree_kind].pop("histogram", None)
            graph["top_nodes"] = graph["top_nodes"][:10]
        result["routing_resolution"]["top_unresolved"] = result[
            "routing_resolution"
        ]["top_unresolved"][:5]
        result["internal_link_resolution"]["top_unresolved"] = result[
            "internal_link_resolution"
        ]["top_unresolved"][:5]
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

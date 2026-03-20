# src/lit_review/deep_lit/clustering.py
"""117 Hypothesis Paper Clustering — service logic.

Clusters selected papers into 4-10 thematic groups using TF-IDF + k-means,
then labels clusters via LLM.

Usage::

    from src.lit_review.deep_lit.clustering import cluster_papers

    result = cluster_papers(ranked_result, llm_client=client)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.lit_review.deep_lit import (
    _MODEL, parse_json_response,
    DEFAULT_MIN_CLUSTERS, DEFAULT_MAX_CLUSTERS,
)

logger = logging.getLogger(__name__)


def cluster_papers(
    ranked_result: Dict[str, Any],
    *,
    llm_client: Any,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
) -> Dict[str, Any]:
    """Cluster selected papers and generate labels."""
    hypothesis_id = ranked_result.get("hypothesis_id", "")
    papers = [p for p in ranked_result.get("papers", []) if p.get("selected")]

    if len(papers) < min_clusters:
        logger.warning("Too few papers (%d) for clustering", len(papers))
        return _single_cluster_fallback(hypothesis_id, papers)

    # Try sklearn-based clustering
    cluster_assignments, n_clusters, silhouette, top_terms = _tfidf_kmeans(
        papers, min_clusters, max_clusters
    )

    if cluster_assignments is None:
        # Fallback to LLM-based clustering
        logger.info("sklearn unavailable; using LLM-based clustering")
        return _llm_cluster_fallback(hypothesis_id, papers, llm_client, max_clusters)

    # Build cluster groups
    clusters: List[Dict[str, Any]] = []
    for k in range(n_clusters):
        cluster_uids = [
            papers[i].get("paper_uid", "")
            for i in range(len(papers))
            if cluster_assignments[i] == k
        ]
        clusters.append({
            "cluster_id": f"c{k+1:02d}",
            "cluster_name": "",  # Filled by LLM labeling
            "description": "",
            "paper_count": len(cluster_uids),
            "top_keywords": top_terms.get(k, []),
            "paper_uids": cluster_uids,
        })

    # Label clusters via LLM
    clusters = _label_clusters(clusters, llm_client)

    # Build assignment map
    assignments = {}
    for c in clusters:
        for uid in c["paper_uids"]:
            assignments[uid] = c["cluster_id"]

    return {
        "hypothesis_id": hypothesis_id,
        "n_clusters": n_clusters,
        "clusters": clusters,
        "paper_cluster_assignments": assignments,
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "method": "tfidf_kmeans",
            "silhouette_score": round(silhouette, 3) if silhouette else None,
        },
    }


# ------------------------------------------------------------------
# TF-IDF + k-means
# ------------------------------------------------------------------

def _tfidf_kmeans(
    papers: List[Dict[str, Any]],
    min_k: int,
    max_k: int,
) -> tuple:
    """Run TF-IDF + k-means. Returns (assignments, n_clusters, silhouette, top_terms) or (None,...) on failure."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        import numpy as np
    except ImportError:
        return None, 0, None, {}

    abstracts = [p.get("abstract", "") or p.get("title", "") for p in papers]
    if not any(abstracts):
        return None, 0, None, {}

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", min_df=2)
    try:
        tfidf_matrix = vectorizer.fit_transform(abstracts)
    except ValueError:
        return None, 0, None, {}

    feature_names = vectorizer.get_feature_names_out()

    # Find optimal k via silhouette
    best_k = min_k
    best_score = -1.0
    best_labels = None

    for k in range(min_k, min(max_k + 1, len(papers))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(tfidf_matrix)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(tfidf_matrix, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    if best_labels is None:
        return None, 0, None, {}

    # Extract top terms per cluster
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    km_final.fit(tfidf_matrix)
    order_centroids = km_final.cluster_centers_.argsort()[:, ::-1]

    top_terms = {}
    for k in range(best_k):
        top_terms[k] = [str(feature_names[i]) for i in order_centroids[k, :8]]

    logger.info("Clustering: k=%d, silhouette=%.3f", best_k, best_score)
    return list(km_final.labels_), best_k, best_score, top_terms


# ------------------------------------------------------------------
# LLM cluster labeling
# ------------------------------------------------------------------

def _label_clusters(
    clusters: List[Dict[str, Any]],
    llm_client: Any,
) -> List[Dict[str, Any]]:
    """Generate human-readable names for clusters via LLM."""
    cluster_info = []
    for c in clusters:
        cluster_info.append(
            f"Cluster {c['cluster_id']}: {c['paper_count']} papers, "
            f"top keywords: {', '.join(c['top_keywords'][:6])}"
        )

    user_msg = (
        "## Paper Clusters\n\n"
        + "\n".join(cluster_info) + "\n\n"
        "## Instructions\n"
        "Generate a short, descriptive name (5-10 words) and one-sentence description "
        "for each cluster.\n"
        'Output JSON: {"labels": [{"cluster_id": "c01", "name": "...", "description": "..."}]}'
    )

    body = {
        "model": _MODEL,
        "max_tokens": 2048,
        "system": "You are an academic literature taxonomy expert. Label paper clusters concisely.",
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
        resp_text = ""
        for block in resp.get("content", []):
            if block.get("type") == "text":
                resp_text = block.get("text", "")
                break

        parsed = parse_json_response(resp_text)
        if parsed and "labels" in parsed:
            label_map = {l["cluster_id"]: l for l in parsed["labels"]}
            for c in clusters:
                label = label_map.get(c["cluster_id"], {})
                c["cluster_name"] = label.get("name", f"Cluster {c['cluster_id']}")
                c["description"] = label.get("description", "")
    except Exception as e:
        logger.warning("Cluster labeling failed: %s", e)
        for c in clusters:
            c["cluster_name"] = f"Cluster {c['cluster_id']}: {', '.join(c['top_keywords'][:3])}"

    return clusters


# ------------------------------------------------------------------
# Fallbacks
# ------------------------------------------------------------------

def _single_cluster_fallback(
    hypothesis_id: str,
    papers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Put all papers in a single cluster."""
    uids = [p.get("paper_uid", "") for p in papers]
    return {
        "hypothesis_id": hypothesis_id,
        "n_clusters": 1,
        "clusters": [{
            "cluster_id": "c01",
            "cluster_name": "All papers",
            "description": "Single cluster (too few papers for meaningful clustering)",
            "paper_count": len(uids),
            "top_keywords": [],
            "paper_uids": uids,
        }],
        "paper_cluster_assignments": {uid: "c01" for uid in uids},
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "method": "single_cluster_fallback",
            "silhouette_score": None,
        },
    }


def _llm_cluster_fallback(
    hypothesis_id: str,
    papers: List[Dict[str, Any]],
    llm_client: Any,
    max_clusters: int,
) -> Dict[str, Any]:
    """Cluster papers using LLM when sklearn is unavailable."""
    # Prepare compact paper list
    paper_lines = []
    for i, p in enumerate(papers[:80]):  # Limit to avoid token overflow
        paper_lines.append(f"[{i}] {p.get('title', '')[:100]}")

    user_msg = (
        f"## Papers ({len(paper_lines)})\n\n"
        + "\n".join(paper_lines) + "\n\n"
        f"## Instructions\n"
        f"Group these papers into 4-{max_clusters} thematic clusters.\n"
        f'Output JSON: {{"clusters": [{{"cluster_id": "c01", "name": "...", '
        f'"description": "...", "paper_indices": [0, 3, 7]}}]}}'
    )

    body = {
        "model": _MODEL,
        "max_tokens": 4096,
        "system": "You are an academic literature taxonomy expert.",
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
        resp_text = ""
        for block in resp.get("content", []):
            if block.get("type") == "text":
                resp_text = block.get("text", "")
                break
        parsed = parse_json_response(resp_text)
    except Exception:
        parsed = None

    if not parsed or "clusters" not in parsed:
        return _single_cluster_fallback(hypothesis_id, papers)

    clusters = []
    assignments = {}
    for c in parsed["clusters"]:
        cid = c.get("cluster_id", f"c{len(clusters)+1:02d}")
        indices = c.get("paper_indices", [])
        uids = [papers[i].get("paper_uid", "") for i in indices if i < len(papers)]
        clusters.append({
            "cluster_id": cid,
            "cluster_name": c.get("name", ""),
            "description": c.get("description", ""),
            "paper_count": len(uids),
            "top_keywords": [],
            "paper_uids": uids,
        })
        for uid in uids:
            assignments[uid] = cid

    # Assign unassigned papers to closest cluster
    assigned_uids = set(assignments.keys())
    if clusters:
        default_cluster = clusters[0]["cluster_id"]
        for p in papers:
            uid = p.get("paper_uid", "")
            if uid and uid not in assigned_uids:
                assignments[uid] = default_cluster
                clusters[0]["paper_uids"].append(uid)
                clusters[0]["paper_count"] += 1

    return {
        "hypothesis_id": hypothesis_id,
        "n_clusters": len(clusters),
        "clusters": clusters,
        "paper_cluster_assignments": assignments,
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "method": "llm_fallback",
            "silhouette_score": None,
        },
    }

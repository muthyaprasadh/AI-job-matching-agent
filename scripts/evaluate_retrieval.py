from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from job_agent.config import DATA, load_profile, load_settings
from job_agent.db import Store
from job_agent.matcher import score_job
from job_agent.retrieval import (
    MiniLMCrossEncoder,
    MiniLMEmbedder,
    bm25_scores,
    job_text,
)

LABELS = DATA / "retrieval_labels.csv"


def _query_for_role(role: str, settings: dict, profile: dict) -> str:
    role_terms = settings.get("roles", {}).get(role, [])
    skill_groups = profile.get("skills", {})
    group_names = {
        "backend_java": ["languages", "backend"],
        "sde": ["languages", "backend", "core_concepts"],
        "ai_ml": ["ai_ml"],
        "data_sql": ["data", "databases"],
        "web_mern": ["languages", "frontend", "backend"],
    }.get(role, [])
    skill_terms = [
        skill
        for group in group_names
        for skill in skill_groups.get(group, [])
    ]
    return " ".join([*role_terms, *skill_terms])


def _metrics(ranking: list[tuple[str, float]], labels: dict[str, int], k: int = 5):
    ordered = [labels[uid] for uid, _ in ranking]
    relevant_count = sum(labels.values())
    recall = sum(ordered[:k]) / relevant_count if relevant_count else 0.0
    reciprocal_rank = next(
        (1 / (index + 1) for index, value in enumerate(ordered) if value), 0.0
    )
    dcg = sum(
        (2**value - 1) / math.log2(index + 2)
        for index, value in enumerate(ordered[:k])
    )
    ideal = sorted(labels.values(), reverse=True)[:k]
    idcg = sum(
        (2**value - 1) / math.log2(index + 2)
        for index, value in enumerate(ideal)
    )
    return recall, reciprocal_rank, dcg / idcg if idcg else 0.0


def evaluate(mode: str) -> dict:
    with LABELS.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels_by_role: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        labels_by_role[row["role"]][row["uid"]] = int(row["relevance"])

    store = Store(DATA / "jobs.sqlite")
    try:
        jobs = store.all_jobs()
    finally:
        store.conn.close()
    jobs_by_uid = {job.uid: job for job in jobs}
    expected_uids = {uid for labels in labels_by_role.values() for uid in labels}
    missing = expected_uids - jobs_by_uid.keys()
    if missing:
        raise RuntimeError(
            "Benchmark postings are missing from data/jobs.sqlite: "
            + ", ".join(sorted(missing))
        )

    settings = load_settings()
    profile = load_profile()
    rankings: dict[str, list[tuple[str, float]]] = {}
    if mode in {"dense", "hybrid", "reranked"}:
        embedder = MiniLMEmbedder()
        all_jobs = [jobs_by_uid[uid] for uid in sorted(expected_uids)]
        vectors = embedder.encode(
            [_query_for_role(role, settings, profile) for role in labels_by_role]
            + [job_text(job) for job in all_jobs]
        )
        query_count = len(labels_by_role)
        query_vectors = vectors[:query_count]
        job_vectors = {
            job.uid: vector
            for job, vector in zip(all_jobs, vectors[query_count:])
        }
    else:
        query_vectors = []
        job_vectors = {}

    cross_encoder = MiniLMCrossEncoder() if mode == "reranked" else None
    for role_index, (role, labels) in enumerate(labels_by_role.items()):
        role_jobs = [jobs_by_uid[uid] for uid in labels]
        query = _query_for_role(role, settings, profile)
        if mode == "keyword":
            scores = {}
            for job in role_jobs:
                match = score_job(job, settings)
                scores[job.uid] = (
                    match.score if match and role in match.matched_roles else 0.0
                )
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        elif mode == "bm25":
            scores = bm25_scores(query, [job_text(job) for job in role_jobs])
            ranked = sorted(
                ((job.uid, score) for job, score in zip(role_jobs, scores)),
                key=lambda item: (-item[1], item[0]),
            )
        else:
            query_vector = query_vectors[role_index]
            dense = sorted(
                (
                    (uid, sum(a * b for a, b in zip(vector, query_vector)))
                    for uid, vector in job_vectors.items()
                    if uid in labels
                ),
                key=lambda item: (-item[1], item[0]),
            )
            if mode == "dense":
                ranked = dense
            else:
                lexical_scores = bm25_scores(
                    query, [job_text(job) for job in role_jobs]
                )
                lexical = sorted(
                    (
                        (job.uid, score)
                        for job, score in zip(role_jobs, lexical_scores)
                        if score > 0
                    ),
                    key=lambda item: (-item[1], item[0]),
                )
                lexical_rank = {uid: index for index, (uid, _) in enumerate(lexical, 1)}
                dense_rank = {uid: index for index, (uid, _) in enumerate(dense, 1)}
                fused = {
                    uid: (
                        (1 / (60 + lexical_rank[uid]) if uid in lexical_rank else 0)
                        + (1 / (60 + dense_rank[uid]) if uid in dense_rank else 0)
                    )
                    for uid in labels
                }
                ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
                if cross_encoder:
                    candidates = [jobs_by_uid[uid] for uid, _ in ranked]
                    rescored = cross_encoder.rerank(query, candidates)
                    ranked = [(job.uid, score) for job, score in rescored]
        rankings[role] = ranked

    per_role = {
        role: _metrics(ranking, labels_by_role[role])
        for role, ranking in rankings.items()
    }
    return {
        "mode": mode,
        "pairs": len(rows),
        "jobs": len(expected_uids),
        "roles": len(labels_by_role),
        "recall@5": sum(value[0] for value in per_role.values()) / len(per_role),
        "mrr": sum(value[1] for value in per_role.values()) / len(per_role),
        "ndcg@5": sum(value[2] for value in per_role.values()) / len(per_role),
        "per_role": per_role,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate job retrieval on judged pairs.")
    parser.add_argument(
        "--mode",
        choices=["keyword", "bm25", "dense", "hybrid", "reranked"],
        default="keyword",
    )
    args = parser.parse_args()
    result = evaluate(args.mode)
    print(
        f"{result['mode']}: pairs={result['pairs']} jobs={result['jobs']} "
        f"roles={result['roles']} recall@5={result['recall@5']:.3f} "
        f"MRR={result['mrr']:.3f} nDCG@5={result['ndcg@5']:.3f}"
    )
    for role, values in result["per_role"].items():
        print(
            f"  {role}: recall@5={values[0]:.3f} MRR={values[1]:.3f} "
            f"nDCG@5={values[2]:.3f}"
        )


if __name__ == "__main__":
    main()

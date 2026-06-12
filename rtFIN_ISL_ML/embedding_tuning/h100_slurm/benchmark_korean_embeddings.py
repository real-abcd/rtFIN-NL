#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def flatten_retrieval_gt(value):
    if value is None:
        return set()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return flatten_retrieval_gt(value.item())
        out = set()
        for item in value.tolist():
            out |= flatten_retrieval_gt(item)
        return out
    if isinstance(value, (list, tuple, set)):
        out = set()
        for item in value:
            out |= flatten_retrieval_gt(item)
        return out
    return {str(value)}


def ndcg_at_k(flags, k):
    dcg = 0.0
    for i, rel in enumerate(flags[:k], start=1):
        if rel:
            dcg += 1.0 / math.log2(i + 1)
    ideal = min(sum(flags), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    return dcg / idcg if idcg else 0.0




def init_wandb_if_available(args, kind):
    project = os.environ.get("WANDB_PROJECT")
    if not project:
        return None
    try:
        import wandb
    except Exception:
        return None
    return wandb.init(
        project=project,
        name=os.environ.get("WANDB_RUN_NAME", f"benchmark_{kind}_{Path(args.model_path).name}"),
        config=vars(args),
        reinit=True,
    )


def benchmark_autorag(model, qa_path, corpus_path, batch_size, output_dir, wandb_run=None):
    qa = pd.read_parquet(qa_path)
    corpus = pd.read_parquet(corpus_path)

    corpus_texts = corpus["contents"].astype(str).tolist()
    corpus_ids = corpus["doc_id"].astype(str).tolist()
    query_texts = qa["query"].astype(str).tolist()
    relevant_docs = [flatten_retrieval_gt(x) for x in qa["retrieval_gt"].tolist()]

    corpus_emb = model.encode(
        corpus_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_emb = model.encode(
        query_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    sims = query_emb @ corpus_emb.T
    k_values = [1, 3, 5, 10, 50]
    accum = {k: {"hit": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "ndcg": 0.0} for k in k_values}
    map_scores = []
    mrr_scores = []

    for qi, relset in enumerate(relevant_docs):
        order = np.argsort(-sims[qi])
        ranked_ids = [corpus_ids[idx] for idx in order]
        flags = [1 if doc_id in relset else 0 for doc_id in ranked_ids]
        rel_total = len(relset)

        first_rank = None
        hits = 0
        precisions = []
        for rank, flag in enumerate(flags, start=1):
            if flag:
                hits += 1
                if first_rank is None:
                    first_rank = rank
                precisions.append(hits / rank)

        mrr_scores.append(0.0 if first_rank is None else 1.0 / first_rank)
        map_scores.append(sum(precisions) / rel_total if rel_total else 0.0)

        for k in k_values:
            topk = flags[:k]
            tp = sum(topk)
            precision_k = tp / k
            recall_k = tp / rel_total if rel_total else 0.0
            f1_k = 2 * precision_k * recall_k / (precision_k + recall_k) if precision_k + recall_k else 0.0
            accum[k]["hit"] += 1 if any(topk) else 0
            accum[k]["precision"] += precision_k
            accum[k]["recall"] += recall_k
            accum[k]["f1"] += f1_k
            accum[k]["ndcg"] += ndcg_at_k(flags, k)

    n = len(query_texts)
    results = {
        "corpus_size": len(corpus_texts),
        "queries_evaluated": n,
        "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "map": float(np.mean(map_scores)) if map_scores else 0.0,
    }
    for k in k_values:
        results[f"hit@{k}"] = accum[k]["hit"] / n if n else 0.0
        results[f"precision@{k}"] = accum[k]["precision"] / n if n else 0.0
        results[f"recall@{k}"] = accum[k]["recall"] / n if n else 0.0
        results[f"f1@{k}"] = accum[k]["f1"] / n if n else 0.0
        results[f"ndcg@{k}"] = accum[k]["ndcg"] / n if n else 0.0

    out = Path(output_dir) / "autorag_benchmark.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    if wandb_run is not None:
        try:
            import wandb
            wandb.log({f"autorag/{k}": v for k, v in results.items()})
        except Exception:
            pass


def benchmark_miracl(model, output_dir, batch_size, wandb_run=None):
    import mteb
    from mteb.deprecated_evaluator import MTEB

    try:
        tasks = mteb.get_tasks(tasks=["MIRACLRetrieval"], languages=["kor"])
    except TypeError:
        tasks = mteb.get_tasks(tasks=["MIRACLRetrieval"])

    evaluator = MTEB(tasks=tasks)
    results = evaluator.run(
        model,
        output_folder=str(Path(output_dir) / "mteb"),
        encode_kwargs={"batch_size": batch_size, "normalize_embeddings": True},
    )

    out = Path(output_dir) / "miracl_benchmark.txt"
    out.write_text(str(results) + "\n", encoding="utf-8")
    print(results)
    if wandb_run is not None:
        try:
            import wandb
            wandb.log({"miracl/results_text": str(results)})
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["autorag", "miracl", "both"], default="both")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--qa_path", default="/mnt/cepheid/users/hsypfsv/rtFIN_ISL_ML/embedding_tuning/h100_slurm/autorag_bench/data/qa_v4.parquet")
    parser.add_argument("--corpus_path", default="/mnt/cepheid/users/hsypfsv/rtFIN_ISL_ML/embedding_tuning/h100_slurm/autorag_bench/data/ocr_corpus_v3.parquet")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    wandb_run = init_wandb_if_available(args, args.kind)

    model = SentenceTransformer(args.model_path, device=device, trust_remote_code=True)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = 512

    if args.kind in ("autorag", "both"):
        benchmark_autorag(model, args.qa_path, args.corpus_path, args.batch_size, args.output_dir, wandb_run=wandb_run)
    if args.kind in ("miracl", "both"):
        benchmark_miracl(model, args.output_dir, args.batch_size, wandb_run=wandb_run)

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()

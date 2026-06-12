#!/usr/bin/env python3
"""
Persona Embedding Fine-tuning Script
Fine-tunes BGE-m3-ko and Qwen3 embedding models on Korean persona data
"""

import os
import json
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import load_dataset, Dataset as HFDataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sentence_transformers import SentenceTransformer, losses, evaluation
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
import pandas as pd
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonaEmbeddingDataset(Dataset):
    """Dataset for persona embedding training"""

    def __init__(self, data_path: str, tokenizer=None, max_length: int = 512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load JSONL data
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        logger.info(f"Loaded {len(self.data)} training examples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # For SentenceTransformers format
        return {
            'anchor': item['query_text'],
            'positive': item['document_text'],
            'negatives': item.get('negatives', [])
        }

def load_persona_dataset(data_path: str) -> HFDataset:
    """Load persona dataset for training"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Convert to SentenceTransformers triplet format
                data.append({
                    'anchor': item['query_text'],
                    'positive': item['document_text'],
                    'negative': item.get('negatives', [''])[0] if item.get('negatives') else ''
                })

    return HFDataset.from_list(data)

def create_sentence_transformers_model(model_name: str, device: str = 'auto'):
    """Create SentenceTransformer model"""
    model = SentenceTransformer(model_name, device=device)

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, '_first_module'):
        model._first_module().gradient_checkpointing_enable()

    return model

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    # Cosine similarity between predictions and labels
    similarities = cosine_similarity(predictions, labels)
    return {
        'cosine_similarity': np.mean(np.diag(similarities))
    }


def init_wandb(args):
    if not args.use_wandb:
        return None
    if getattr(args, 'local_rank', -1) not in (-1, 0):
        return None
    if wandb is None or not hasattr(wandb, 'init'):
        logger.warning("wandb package is not installed or unavailable; continuing without W&B logging.")
        return None
    run_name = args.wandb_run_name or os.getenv('WANDB_NAME') or f"{args.model_type}-{os.path.basename(args.output_dir)}-{int(time.time())}"
    project = args.wandb_project or os.getenv('WANDB_PROJECT', 'persona-embedding')
    wandb.init(
        project=project,
        name=run_name,
        config={k: v for k, v in vars(args).items() if k not in {'local_rank', 'use_wandb', 'evaluate_only', 'benchmark'}}
    )
    return wandb


def setup_device(args):
    if torch.cuda.is_available():
        if args.local_rank is not None and args.local_rank >= 0:
            torch.cuda.set_device(args.local_rank)
            device = torch.device('cuda', args.local_rank)
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.bf16:
        args.fp16 = False
    elif args.fp16:
        args.bf16 = False
    else:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            args.bf16 = True
        else:
            args.fp16 = True

    logger.info(
        "Device setup: device=%s, cuda_available=%s, gpu_count=%s, bf16=%s, fp16=%s, local_rank=%s",
        device,
        torch.cuda.is_available(),
        torch.cuda.device_count(),
        args.bf16,
        args.fp16,
        args.local_rank,
    )
    return device


def evaluate_embedding_model_realistic(model, eval_data_path: str, model_type: str = "bge", corpus_size: int = 1000, top_k: int = 10):
    """Evaluate embedding model with realistic retrieval scenario"""
    logger.info(f"Evaluating {model_type} model with {corpus_size} documents")

    # Load all evaluation data
    eval_data = []
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                eval_data.append(json.loads(line))

    if len(eval_data) < corpus_size:
        logger.warning(f"Only {len(eval_data)} samples available, using all")
        corpus_size = len(eval_data)

    # Create corpus: mix of positive and negative documents
    corpus_texts = []
    query_positive_map = {}

    for i, item in enumerate(eval_data[:corpus_size]):
        query = item['query_text']
        positive = item['document_text']
        negatives = item.get('negatives', [])

        # Store mapping
        if query not in query_positive_map:
            query_positive_map[query] = []
        query_positive_map[query].append(positive)

        # Add to corpus
        corpus_texts.append(positive)
        for neg in negatives[:2]:  # Add some negatives to corpus
            if neg and neg not in corpus_texts:
                corpus_texts.append(neg)

    # Encode entire corpus once
    logger.info(f"Encoding {len(corpus_texts)} documents...")
    if hasattr(model, "encode"):
        corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=False)
    else:
        # For Qwen/transformers models
        tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        corpus_inputs = tokenizer(corpus_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        corpus_inputs = {k: v.to(model.device) for k, v in corpus_inputs.items()}

        with torch.no_grad():
            corpus_outputs = model(**corpus_inputs)
            corpus_embeddings = corpus_outputs.last_hidden_state[:, 0, :]  # CLS token

    results = {'corpus_size': len(corpus_texts), 'queries_evaluated': 0}

    # Evaluate each query
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    mrr_scores = []
    avg_cosine_similarities = []

    for query, positives in query_positive_map.items():
        if hasattr(model, "encode"):
            query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)[0]
        else:
            query_input = tokenizer([query], return_tensors='pt', padding=True, truncation=True, max_length=512)
            query_input = {k: v.to(model.device) for k, v in query_input.items()}

            with torch.no_grad():
                query_output = model(**query_input)
                query_emb = query_output.last_hidden_state[0, 0, :]

        # Compute similarities with entire corpus
        similarities = torch.nn.functional.cosine_similarity(
            query_emb.unsqueeze(0), corpus_embeddings, dim=1
        )

        # Get top-k most similar documents
        top_k_values, top_k_indices = torch.topk(similarities, min(top_k, len(corpus_texts)), largest=True)

        # Check if any positive document is in top-k
        retrieved_docs = [corpus_texts[idx] for idx in top_k_indices.tolist()]
        positive_found = any(pos in retrieved_docs for pos in positives)

        # Find rank of first positive document
        positive_ranks = []
        for pos in positives:
            if pos in corpus_texts:
                pos_idx = corpus_texts.index(pos)
                pos_similarity = similarities[pos_idx].item()
                # Find rank (1-indexed)
                rank = (similarities >= pos_similarity).sum().item()
                positive_ranks.append(rank)

        if positive_ranks:
            best_rank = min(positive_ranks)
            avg_cosine_similarities.append(similarities[corpus_texts.index(positives[0])].item())

            if best_rank == 1:
                top1_correct += 1
            if best_rank <= 5:
                top5_correct += 1
            if best_rank <= 10:
                top10_correct += 1

            mrr_scores.append(1.0 / best_rank)
            results['queries_evaluated'] += 1

    # Calculate final metrics
    if results['queries_evaluated'] > 0:
        results['top1_accuracy'] = top1_correct / results['queries_evaluated']
        results['top5_accuracy'] = top5_correct / results['queries_evaluated']
        results['top10_accuracy'] = top10_correct / results['queries_evaluated']
        results['mean_reciprocal_rank'] = np.mean(mrr_scores)
        results['avg_positive_similarity'] = np.mean(avg_cosine_similarities)
    else:
        results.update({'top1_accuracy': 0.0, 'top5_accuracy': 0.0, 'top10_accuracy': 0.0,
                       'mean_reciprocal_rank': 0.0, 'avg_positive_similarity': 0.0})

    logger.info(f"Realistic evaluation results: {results}")
    return results

def save_evaluation_results(results: Dict, output_path: str):
    """Save evaluation results to JSON"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Evaluation results saved to {output_path}")

def benchmark_model(model, model_type: str, model_name: str, data_path: str, output_dir: str, max_samples: int = 1000, wandb_run=None, realistic_eval: bool = True):
    """Run comprehensive benchmarking"""
    logger.info(f"Benchmarking {model_type} model: {model_name}")

    if realistic_eval:
        # Realistic evaluation with larger corpus
        eval_results = evaluate_embedding_model_realistic(model, data_path, model_type, corpus_size=max_samples, top_k=10)
    else:
        logger.warning("Contrastive evaluator is not implemented; falling back to retrieval evaluation.")
        eval_results = evaluate_embedding_model_realistic(model, data_path, model_type, corpus_size=max_samples, top_k=10)

    results_file = os.path.join(output_dir, f"eval_results_{model_type}.json")
    save_evaluation_results(eval_results, results_file)

    if wandb_run is not None:
        wandb_run.log({f"benchmark/{model_type}/{k}": v for k, v in eval_results.items() if isinstance(v, (int, float))})

    print(f"\n{'='*60}")
    print(f"Model: {model_name} ({model_type})")
    print(f"Evaluation: {'Realistic' if realistic_eval else 'Contrastive'}")
    print(f"Corpus Size: {eval_results.get('corpus_size', 'N/A')}")
    print(f"Queries Evaluated: {eval_results.get('queries_evaluated', eval_results.get('num_samples', 0))}")
    print(f"Average Positive Similarity: {eval_results.get('avg_positive_similarity', eval_results.get('avg_cosine_similarity', 'N/A'))}")
    print(f"Top-1 Accuracy: {eval_results.get('top1_accuracy', 'N/A')}")
    print(f"Top-5 Accuracy: {eval_results.get('top5_accuracy', 'N/A')}")
    print(f"Top-10 Accuracy: {eval_results.get('top10_accuracy', 'N/A')}")
    print(f"Mean Reciprocal Rank: {eval_results.get('mean_reciprocal_rank', 'N/A')}")
    print(f"{'='*60}\n")

    return eval_results

def create_sentence_transformers_model(model_name: str, device: str = 'auto', trust_remote_code: bool = True, torch_dtype=None):
    """Create SentenceTransformer model"""
    model_kwargs = {}
    if torch_dtype is not None:
        model_kwargs['torch_dtype'] = torch_dtype

    model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=trust_remote_code,
        model_kwargs=model_kwargs,
    )

    if hasattr(model, '_first_module') and hasattr(model._first_module(), 'gradient_checkpointing_enable'):
        model._first_module().gradient_checkpointing_enable()

    return model


def enable_gradient_checkpointing(model) -> bool:
    """Enable gradient checkpointing for SentenceTransformer backbones when available."""
    modules_to_try = []

    if hasattr(model, '_first_module'):
        first_module = model._first_module()
        modules_to_try.append(first_module)
        for attr in ('auto_model', 'model'):
            child = getattr(first_module, attr, None)
            if child is not None:
                modules_to_try.append(child)

    for module in modules_to_try:
        enable = getattr(module, 'gradient_checkpointing_enable', None)
        if callable(enable):
            enable()
            return True

    return False


def train_sentence_transformer_model(args):
    """Train an embedding model with SentenceTransformer"""
    logger.info(f"Starting {args.model_type} model training with SentenceTransformer...")

    model = create_sentence_transformers_model(
        args.model_name,
        device=str(args.device),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length
        logger.info("Set SentenceTransformer max_seq_length=%s", args.max_seq_length)

    train_dataset = load_persona_dataset(args.train_data)
    # Keep args.eval_data for the post-training retrieval benchmark. The
    # trainer's built-in triplet evaluator is intentionally disabled for POC
    # runs because its metric names vary across sentence-transformers versions.
    eval_dataset = None

    train_loss = losses.MultipleNegativesRankingLoss(model)

    evaluator = None
    if eval_dataset:
        evaluator = evaluation.TripletEvaluator(
            anchors=eval_dataset['anchor'],
            positives=eval_dataset['positive'],
            negatives=eval_dataset['negative'],
            name='persona_eval',
        )

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy='no',
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=True,
        dataloader_num_workers=4,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=['wandb'] if args.use_wandb and args.local_rank in (-1, 0) else ['none'],
        run_name=args.wandb_run_name if args.use_wandb else None,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False} if args.gradient_checkpointing else None,
        optim=args.optim,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )

    trainer.train()
    model.save(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")

    return trainer

def train_bge_model(args):
    return train_sentence_transformer_model(args)


def train_qwen_model(args):
    return train_sentence_transformer_model(args)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate embedding models on persona data")
    parser.add_argument("--model_type", choices=["bge", "qwen"], required=True,
                       help="Model type to train")
    parser.add_argument("--model_name", required=True,
                       help="HuggingFace model name")
    parser.add_argument("--train_data", required=True,
                       help="Path to training data JSONL")
    parser.add_argument("--eval_data", default=None,
                       help="Path to evaluation data JSONL")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for trained model")

    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=-1,
                       help="Maximum training steps; -1 means use num_epochs")
    parser.add_argument("--optim", default="adamw_torch",
                       help="Trainer optimizer, e.g. adamw_torch or adafactor")
    parser.add_argument("--ddp_find_unused_parameters", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=None,
                       help="Override DDP find_unused_parameters")

    # Precision
    parser.add_argument("--fp16", action="store_true",
                       help="Use FP16 training")
    parser.add_argument("--bf16", action="store_true",
                       help="Use BF16 training")

    # WandB logging
    parser.add_argument("--use_wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default=None,
                       help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", default=None,
                       help="Weights & Biases run name")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv('LOCAL_RANK', '-1')),
                       help="Local rank for distributed training")

    # Memory optimization
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing for memory efficiency")
    parser.add_argument("--empty_cache_steps", type=int, default=0,
                       help="Empty CUDA cache every N steps (0 = disabled)")

    # Logging and saving
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save model every N steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                       help="Maximum number of saved checkpoints")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Log every N steps")

    # Evaluation options
    parser.add_argument("--evaluate_only", action="store_true",
                       help="Only run evaluation, skip training")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run comprehensive benchmarking after training")
    parser.add_argument("--eval_samples", type=int, default=1000,
                       help="Number of samples to use for evaluation")
    parser.add_argument("--realistic_eval", action="store_true",
                       help="Use realistic retrieval evaluation (recommended) instead of contrastive evaluation")
    parser.add_argument("--max_seq_length", type=int, default=None,
                       help="Optional SentenceTransformer max sequence length override")

    args = parser.parse_args()

    # Override local rank if torchrun sets it
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    # Set precision defaults
    if args.bf16:
        args.fp16 = False
    elif args.fp16:
        args.bf16 = False
    else:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            args.bf16 = True
        else:
            args.fp16 = True

    args.device = setup_device(args)
    wandb_run = init_wandb(args)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.evaluate_only:
        logger.info("Running evaluation only...")

        if args.model_type == "bge":
            model = SentenceTransformer(args.model_name, trust_remote_code=True, device=str(args.device))
        elif args.model_type == "qwen":
            model = SentenceTransformer(args.model_name, trust_remote_code=True, device=str(args.device))
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        if args.max_seq_length:
            model.max_seq_length = args.max_seq_length
            logger.info("Set SentenceTransformer max_seq_length=%s", args.max_seq_length)

        benchmark_data = args.eval_data or args.train_data
        benchmark_model(model, args.model_type, args.model_name, benchmark_data, args.output_dir, max_samples=args.eval_samples, wandb_run=wandb_run, realistic_eval=args.realistic_eval)
        return

    if args.model_type == "bge":
        trainer = train_bge_model(args)
        model = trainer.model
    elif args.model_type == "qwen":
        trainer = train_qwen_model(args)
        model = trainer.model
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    logger.info("Training completed successfully!")

    if args.benchmark:
        logger.info("Running post-training benchmarking...")
        benchmark_data = args.eval_data or args.train_data
        eval_results = benchmark_model(
            model,
            args.model_type,
            args.model_name,
            benchmark_data,
            args.output_dir,
            max_samples=args.eval_samples,
            wandb_run=wandb_run,
            realistic_eval=args.realistic_eval,
        )

        benchmark_file = os.path.join(args.output_dir, "benchmark_results.json")
        serializable_args = {
            key: str(value) if isinstance(value, (Path, torch.device)) else value
            for key, value in vars(args).items()
        }
        with open(benchmark_file, 'w', encoding='utf-8') as f:
            json.dump({
                "model_name": args.model_name,
                "model_type": args.model_type,
                "training_args": serializable_args,
                "evaluation_results": eval_results,
                "timestamp": str(pd.Timestamp.now())
            }, f, indent=2, ensure_ascii=False)

    if wandb_run is not None:
        wandb_run.finish()

    logger.info("All tasks completed successfully!")

if __name__ == "__main__":
    main()

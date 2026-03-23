#!/usr/bin/env python3
"""CLI runner for Welsh LLM evaluations.

Usage:
    python -m deepeval_evals.run_all --model gpt-4o
    python -m deepeval_evals.run_all --model anthropic/claude-sonnet-4-20250514 --eval welsh-lexicon
    python -m deepeval_evals.run_all --model ollama/llama3 --max-samples 50
    python -m deepeval_evals.run_all --model gpt-4o --eval welsh-legislation-translation
"""

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime

from deepeval.test_case import LLMTestCase
from tqdm import tqdm

from deepeval_evals.loaders import load_jsonl_goldens
from deepeval_evals.metrics.bleu_score import compute_corpus_bleu
from deepeval_evals.models import generate_response, resolve_hf_model_id

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "evals-cymraeg")

EVALS = {
    "welsh-lexicon": {
        "jsonl": "welsh-lexicon/data/welsh-lexicon/samples.jsonl",
        "metric": "exact_match",
    },
    "welsh-grammar": {
        "jsonl": "welsh-grammar/data/welsh-grammar/samples.jsonl",
        "metric": "exact_match",
    },
    "welsh-yes-no": {
        "jsonl": "welsh-yes-no/data/welsh-yes-no/samples.jsonl",
        "metric": "exact_match",
    },
    "welsh-obscenities": {
        "jsonl": "welsh-obscenities/data/welsh-obscenities/samples.jsonl",
        "metric": "exact_match",
    },
    "welsh-bilingual-placenames": {
        "jsonl": "welsh-bilingual-placenames/data/welsh-bilingual-placenames/samples.jsonl",
        "metric": "exact_match",
    },
    "welsh-legislation-translation": {
        "jsonl": "welsh-legislation-translation/data/welsh-legislation-translation/samples.jsonl",
        "metric": "bleu",
    },
    "welsh-registers": {
        "jsonl": "welsh-registers/data/welsh-registers/samples.jsonl",
        "metric": "exact_match",
    },
    "welsh-mmlu-lite": {
        "jsonl": "welsh-mmlu-lite/data/welsh-mmlu-lite/samples.jsonl",
        "metric": "mcq",
        "max_tokens": 10,
    },
}


def run_eval(eval_name: str, model_id: str, max_samples: int = None):
    config = EVALS[eval_name]
    jsonl_path = os.path.join(BASE_DIR, config["jsonl"])

    print(f"\n{'='*60}")
    print(f"Eval: {eval_name}")
    print(f"Model: {model_id}")
    print(f"{'='*60}")

    goldens = load_jsonl_goldens(jsonl_path, max_samples=max_samples)
    print(f"Loaded {len(goldens)} samples")

    test_cases = []
    predictions = []
    references = []

    max_tokens = config.get("max_tokens", 500)
    for i, g in enumerate(tqdm(goldens, desc="Generating responses")):
        actual = generate_response(model_id, g.system_message, g.user_message, max_tokens=max_tokens)
        predictions.append(actual)
        references.append(g.expected_output)
        test_cases.append(LLMTestCase(
            input=g.user_message,
            actual_output=actual,
            expected_output=g.expected_output,
        ))

    # Summary
    if config["metric"] in ("exact_match", "mcq"):
        if config["metric"] == "mcq":
            # Extract first A/B/C/D letter from the response
            def extract_mcq(text):
                m = re.search(r'\b([A-D])\b', text)
                return m.group(1) if m else text.strip()
            correct = sum(1 for tc in test_cases if extract_mcq(tc.actual_output) == tc.expected_output.strip())
        else:
            correct = sum(1 for tc in test_cases if tc.actual_output.strip().strip(".,!?").lower() == tc.expected_output.strip().strip(".,!?").lower())
        accuracy = correct / len(test_cases) * 100
        print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{len(test_cases)})")
        return {"eval": eval_name, "metric": "accuracy", "score": f"{accuracy:.2f}", "n": len(test_cases)}
    else:
        corpus_bleu = compute_corpus_bleu(predictions, references)
        print(f"\nCorpus BLEU: {corpus_bleu:.1f}")
        return {"eval": eval_name, "metric": "BLEU", "score": f"{corpus_bleu:.1f}", "n": len(test_cases)}


def main():
    parser = argparse.ArgumentParser(description="Run Welsh LLM evaluations")
    parser.add_argument("--model", required=True, help="LLM model ID (e.g. gpt-4o, anthropic/claude-sonnet-4-20250514, ollama/llama3)")
    parser.add_argument("--eval", choices=list(EVALS.keys()), help="Run a specific eval (default: all)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples per eval")
    args = parser.parse_args()

    model_id = args.model
    if model_id.startswith("hf/"):
        actual_model = resolve_hf_model_id()
        print(f"HF server is serving: {actual_model}")
        model_id = f"hf/{actual_model}"

    eval_names = [args.eval] if args.eval else list(EVALS.keys())

    print(f"Running {len(eval_names)} eval(s) with model: {model_id}")
    start = time.time()

    summaries = []
    for name in eval_names:
        summary = run_eval(name, model_id, args.max_samples)
        summaries.append(summary)

    elapsed = time.time() - start

    # Write summary CSV
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_id.replace("/", "_")
    csv_path = os.path.join(results_dir, f"{timestamp}_{model_slug}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["eval", "metric", "score", "n"])
        writer.writeheader()
        writer.writerows(summaries)

    # Print summary table
    print(f"\n{'='*50}")
    print(f"{'Eval':<35} {'Metric':<10} {'Score':>8} {'N':>6}")
    print(f"{'-'*50}")
    for s in summaries:
        print(f"{s['eval']:<35} {s['metric']:<10} {s['score']:>8} {s['n']:>6}")
    print(f"{'='*50}")
    print(f"Results saved to {csv_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

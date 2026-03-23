import os
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval_evals.loaders import load_jsonl_goldens
from deepeval_evals.metrics import SacreBleuMetric
from deepeval_evals.metrics.bleu_score import compute_corpus_bleu
from deepeval_evals.models import generate_response

JSONL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "evals-cymraeg",
    "welsh-legislation-translation", "data",
    "welsh-legislation-translation", "samples.jsonl"
)


def test_welsh_legislation_translation(model_id, max_samples):
    goldens = load_jsonl_goldens(JSONL_PATH, max_samples=max_samples)
    test_cases = []
    predictions = []
    references = []

    for g in goldens:
        actual = generate_response(model_id, g.system_message, g.user_message)
        predictions.append(actual)
        references.append(g.expected_output)
        test_cases.append(LLMTestCase(
            input=g.user_message,
            actual_output=actual,
            expected_output=g.expected_output,
        ))

    metric = SacreBleuMetric()
    evaluate(test_cases=test_cases, metrics=[metric])

    # Report corpus-level BLEU (the primary translation quality metric)
    corpus_bleu = compute_corpus_bleu(predictions, references)
    print(f"\nCorpus BLEU: {corpus_bleu:.1f}")

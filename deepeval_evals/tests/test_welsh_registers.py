import os
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval_evals.loaders import load_jsonl_goldens
from deepeval_evals.metrics import WelshExactMatchMetric
from deepeval_evals.models import generate_response

JSONL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "evals-cymraeg",
    "welsh-registers", "data", "welsh-registers", "samples.jsonl"
)

def test_welsh_registers(model_id, max_samples):
    goldens = load_jsonl_goldens(JSONL_PATH, max_samples=max_samples)
    test_cases = []
    for g in goldens:
        actual = generate_response(model_id, g.system_message, g.user_message)
        test_cases.append(LLMTestCase(
            input=g.user_message,
            actual_output=actual,
            expected_output=g.expected_output,
        ))

    metric = WelshExactMatchMetric()
    evaluate(test_cases=test_cases, metrics=[metric])

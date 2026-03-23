import json
from dataclasses import dataclass


@dataclass
class Golden:
    """A single eval sample loaded from JSONL."""
    system_message: str
    user_message: str
    expected_output: str


def load_jsonl_goldens(file_path: str, max_samples: int = None) -> list[Golden]:
    """Load existing OpenAI Evals JSONL files into Golden records.

    Expected JSONL format per line:
    {"input": [{"role":"system","content":"..."}, {"role":"user","content":"..."}], "ideal": "..."}
    """
    goldens = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            messages = record["input"]
            system_msg = ""
            user_msg = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                elif msg["role"] == "user":
                    user_msg = msg["content"]

            goldens.append(Golden(
                system_message=system_msg,
                user_message=user_msg,
                expected_output=record["ideal"],
            ))

            if max_samples and len(goldens) >= max_samples:
                break

    return goldens

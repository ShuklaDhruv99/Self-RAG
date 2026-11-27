from selfrag.bot.state import State
from selfrag.rag.baseline import generate  # same prompt as the naive baseline, so the Milestone 5 comparison is fair


def generate_from_context(state: State) -> dict:
    answer = generate(state["question"], state["context"])
    return {
        "answer": answer,
        "generation_count": state["generation_count"] + 1,
        "steps": ["generate_from_context"],
    }

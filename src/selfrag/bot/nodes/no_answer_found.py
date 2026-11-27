from selfrag.bot.state import State

NO_ANSWER_MESSAGE = "I could not find a reliable answer in the documents."


def no_answer_found(state: State) -> dict:
    return {"final_answer": NO_ANSWER_MESSAGE, "steps": ["no_answer_found"]}

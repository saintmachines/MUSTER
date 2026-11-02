"""An agent loop with MUSTER as its long-term memory. Replace `call_llm` with your model call."""
import muster

store = muster.Store.open("agent.muster")


def call_llm(messages):
    raise NotImplementedError("plug in your LLM client here")


def turn(user_text, *, when):
    store.add({"speaker": "user", "text": user_text, "time": when})
    ctx = store.context(user_text)
    reply = call_llm(ctx.messages(user_text, system="Answer from the memory below when it is relevant."))
    store.add({"speaker": "assistant", "text": reply, "time": when})
    return reply

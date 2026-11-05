"""Two hooks around an application's existing model call.

Import answer_with_memory in your app and pass call_model(messages) -> str.
Run `python examples/chat_app.py --demo` for a fixed-response wiring demo.
The demo uses fake retrieval models; it is not an answer-quality example.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from muster import Store


def answer_with_memory(store, user_text, call_model, *, system=None, when=None, request_id=None):
    """Retrieve context, call the app's model, then remember the completed turn.

    The caller owns the open Store and its model client. A request_id deduplicates
    the stored turn only; it does not cache or deduplicate the model call.
    A failed/non-text reply leaves this completed-turn log unchanged.
    """
    messages = store.context(user_text).messages(user_text, system=system)
    answer = call_model(messages)
    if not isinstance(answer, str):
        raise TypeError("call_model(messages) must return the assistant's text")
    stamp = when or datetime.now(timezone.utc).isoformat()
    store.add([
        {"speaker": "user", "text": user_text, "time": stamp},
        {"speaker": "assistant", "text": answer, "time": stamp},
    ], request_id=request_id)
    return answer


def demo():
    from muster.models import Models

    with TemporaryDirectory(prefix="muster-chat-demo-") as folder:
        path = Path(folder) / "demo.muster"
        with Store.open(path, models=Models.fake(32)) as store:
            store.add({"speaker": "user", "text": "The spare cable is in the blue cabinet."})

            def fixed_reader(messages):
                assert messages[0] == {"role": "system", "content": "Use relevant memory."}
                assert "blue cabinet" in messages[-1]["content"]
                assert "Where is the spare cable?" in messages[-1]["content"]
                return "The spare cable is in the blue cabinet."

            answer = answer_with_memory(
                store, "Where is the spare cable?", fixed_reader,
                system="Use relevant memory.", request_id="demo-turn-1",
            )
            assert store.version == 3
        with Store.open(path, models=Models.fake(32)) as store:
            assert store.version == 3
            assert "blue cabinet" in store.context("spare cable").text
        print("Wiring demo: fake retrieval and a fixed reader response; no AI answer was generated.")
        print(answer)
        print("Confirmed: context reached the reader, both messages were stored, and reopening kept all 3 records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="check the flow with fake retrieval and a fixed reply")
    if parser.parse_args().demo:
        demo()
    else:
        parser.print_help()

"""Which of the roster's models actually answer, and with a system role.

The lab guests are all being forced onto one model because Gemma rejects a
system role outright and the factory passes the persona in as one. That
override was applied to everybody, so three characters that the kit gives
three different voices have been speaking with the same one — and any tic we
measured off them was one voice counted three times.

Before building a mixed roster, ask each model whether it answers at all, in
Italian, with a system prompt. One short call each.

    python -m bench.model_probe [model ...]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.featherless import FeatherlessBackend, install_windows_shim  # noqa: E402

ROSTER = (
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3.5-9B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "google/gemma-4-31B-it",
)

SYSTEM = ("Sei Marco, 41 anni, skipper di Cagliari. Scrivi in italiano, "
          "una riga, minuscolo, come in chat.")
PROMPT = "ciao, come va oggi?"


def probe(model: str) -> tuple[bool, str, float]:
    start = time.monotonic()
    try:
        # Reasoning models spend the budget thinking before they write a
        # word: at 40 tokens both R1-distill and Qwen3.5 came back empty
        # with finish_reason=length, which reads like a refusal and is not.
        budget = int(os.environ.get("PROBE_TOKENS", "40"))
        backend = FeatherlessBackend(model=model, max_tokens=budget, temperature=0.8)
        answer = backend(PROMPT, system_prompt=SYSTEM)
        backend.close()
    except Exception as problem:
        return False, f"{type(problem).__name__}: {problem}"[:150], time.monotonic() - start
    text = (answer or "").strip()
    if not text:
        return False, "risposta vuota", time.monotonic() - start
    return True, text[:90], time.monotonic() - start


def main() -> int:
    install_windows_shim()
    models = sys.argv[1:] or list(ROSTER)
    ok = 0
    for model in models:
        works, detail, seconds = probe(model)
        mark = "OK  " if works else "NO  "
        ok += works
        print(f"{mark} {model:44s} {seconds:5.1f}s  {detail}")
    print(f"\n{ok}/{len(models)} modelli utilizzabili con un ruolo di sistema")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Everything `my_agent.py` does, except joining the world.

    cd Turing
    python preflight.py

Run this before a competition session. It walks the same path the real entry
takes — keys, the Windows shim, the gateway, one live model call, the node
registration — and stops at the door rather than entering the hotel, so it costs
one generation and no room.

Each check prints OK or the reason it failed. A failure here is a failure
tomorrow, found tonight instead of in front of everybody.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OK, BAD = "  OK   ", "  FAIL "
problems = []


def check(label, fn):
    print(f"{'':7}{label} ...", end=" ", flush=True)
    try:
        detail = fn()
    except Exception as e:                                  # noqa: BLE001
        print(f"\r{BAD}{label}: {type(e).__name__}: {e}")
        problems.append(label)
        return None
    print(f"\r{OK}{label}" + (f" — {detail}" if detail else ""))
    return detail


print("\nPREFLIGHT — la catena completa, senza entrare nell'albergo\n")

# 1. The two keys. NODE_KEY also lives in a cache file, so absence of the
# variable is not necessarily a problem; the absence of both is.
def keys():
    node_key = os.environ.get("NODE_KEY", "")
    cache = os.path.join(os.environ.get("APPDATA", ""), "Local", "unaiverse", "key")
    if not node_key and not os.path.exists(cache):
        raise RuntimeError("né NODE_KEY né il file di cache della chiave")
    if not os.environ.get("FEATHERLESS_API_KEY"):
        raise RuntimeError("FEATHERLESS_API_KEY non impostata")
    return "NODE_KEY " + ("da env" if node_key else "da cache") + ", FEATHERLESS_API_KEY presente"


check("chiavi", keys)

# 2. The shim. `FeatherlessAPI._ensure_server` opens with `import fcntl`, which
# does not exist on Windows, and it is called unconditionally from __init__.
def shim():
    from processors.featherless import WINDOWS_SHIM
    if sys.platform.startswith("win") and not WINDOWS_SHIM:
        raise RuntimeError("su Windows ma lo shim non si è installato")
    return "installato" if WINDOWS_SHIM else "non serve (non è Windows)"


check("shim fcntl per Windows", shim)

# 3. The agent itself, offline. Catches a broken prompt file or a syntax error
# in anything the entry imports, before any of it touches the network.
def build():
    from processors.boss import Boss
    from policies.boss_timing import BossTiming
    proc = Boss(model=os.environ.get("BOSS_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
                fallback=os.environ.get("BOSS_FALLBACK", "Qwen/Qwen2.5-32B-Instruct"),
                max_tokens=60, temperature=0.95, top_p=0.95, top_k=60,
                repetition_penalty=1.08)
    BossTiming(sense=proc.sense)
    globals()["PROC"] = proc
    return f"{len(proc.personas)} personaggi, {len(proc.entries)} aperture"


check("processore e filtro", build)

# 4. The context file has to parse into something, or the agent is a persona
# with no idea where it is.
def context():
    proc = globals().get("PROC")
    block = proc.context.block()
    if len(block) < 80:
        raise RuntimeError("il blocco di contesto è vuoto o quasi")
    return f"{len(block)} caratteri, {len(proc.context.markers())} parole-marcatore"


check("contesto della conferenza", context)

# 5. The expensive one: a real call through the shared gateway. This is the
# check that fails when the key is wrong, the model is not served any more, or
# the gateway cannot start — all of which look identical from the room.
def gateway():
    proc = globals().get("PROC")
    start = time.monotonic()
    if not proc.warm():
        raise RuntimeError("warm() ha risposto vuoto: gateway o modello non disponibili")
    return f"prima generazione in {time.monotonic() - start:.1f}s"


check("gateway Featherless (una chiamata vera)", gateway)

# 6. Node registration. New names consume a slot permanently and the account is
# at its cap, so this proves the name we intend to use is reusable.
def node():
    from unaiverse.networking.node.node import Node
    from unaiverse.agent import Agent
    proc = globals().get("PROC")
    name = os.environ.get("BOSS_NODE_NAME", "TuringBoss")
    agent = Agent(proc=proc, proc_inputs=["text"], proc_outputs=["text"])
    Node(hosted=agent, node_name=name, hidden=True, clock_delta=1. / 10.)
    return f"'{name}' registrato (nessuno slot nuovo consumato)"


check("nodo sulla rete", node)

world = os.environ.get("BOSS_WORLD", "TuringHotelItaly")
print(f"\n{'':7}mondo di destinazione: {world}")
if world == "TuringHotelItaly":
    print(f"{'':7}(nome nudo: si risolve PRIMA sui tuoi nodi — assicurati di non"
          f"\n{'':7} possedere un mondo con questo nome, o giocheresti contro te stesso)")

print()
if problems:
    print(f"  {len(problems)} problema/i: " + ", ".join(problems))
    sys.exit(1)
print("  tutto a posto: la catena regge fino alla porta dell'albergo.")

"""Combine several filters.

Each filter runs in order and the first one that says "not yet" wins, so a chain
behaves as an AND: the action goes through only when every filter agrees.

    from policies.chain import Chain
    from policies.human_delay import HumanDelay
    from policies.turn_taking import TurnTaking

    policy = Chain(TurnTaking(reply_chance=0.7), HumanDelay(median=4.0))

The whole job of this class is bookkeeping. Every filter keeps its state in the
`opts` dictionary it is given, and they all use the same key names (`ready_at`,
`quiet_until`), so sharing one dictionary would have them overwriting each
other's timers. Chain hands each filter its own private slice instead.

Two filters in a chain multiply their silences, so a chain is quieter than
either part on its own. Check with `python -m policies.simulate` before entering
a chain of three.
"""


class Chain:

    def __init__(self, *filters):
        self.filters = filters

    def __call__(self, action_id, request, all_actions, opts):
        for i, policy_filter in enumerate(self.filters):
            # One private sub-dictionary per filter, created on first use and
            # kept for the life of the agent, like `opts` itself.
            mine = opts.setdefault(f"chain_{i}", {})

            # The framework only fills these two on the top-level dictionary, so
            # copy them down or the filters lose access to the agent.
            mine["agent"] = opts.get("agent")
            mine["public"] = opts.get("public")

            action_id, request = policy_filter(action_id, request, all_actions, mine)
            if action_id < 0:
                return -1, None
        return action_id, request

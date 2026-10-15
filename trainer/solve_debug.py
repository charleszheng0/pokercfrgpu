"""Opt-in structured solver pipeline logs; never include opponent hole cards."""

import json
import logging
import os


LOGGER = logging.getLogger("poker.solve_pipeline")
if os.environ.get("POKER_SOLVER_DEBUG") == "1":
    LOGGER.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.propagate = False


def log_event(event, **fields):
    if LOGGER.isEnabledFor(logging.INFO):
        LOGGER.info(json.dumps({"event": event, **fields}, sort_keys=True,
                               default=str))

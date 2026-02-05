import math
from typing import Tuple


def ceil_to_mod(block: int, mod: int) -> int:
    if mod <= 0:
        raise ValueError("mod must be > 0")
    r = int(block) % int(mod)
    return int(block) if r == 0 else int(block) + (int(mod) - r)


def compute_sota_window(
    *,
    decision_block: int,
    alignment_mod: int,
    t2_blocks: int,
    activation_delay_intervals: int = 1,
    t2_intervals: int | None = None,
    min_t2_intervals: int = 2,
) -> Tuple[int, int, int]:
    """
    Compute a deterministic reward window aligned to `alignment_mod`.

    Returns: (start_block, end_block, effective_t2_blocks)

    Notes
    - `start_block` is aligned to the next boundary after `decision_block`, plus an
      activation delay in whole intervals. This gives validators time to observe the
      event and avoids "missed start" on tight windows.
    - If `t2_intervals` is not provided, `t2_blocks` is converted to intervals via
      ceil(t2_blocks / alignment_mod) and then clamped to `min_t2_intervals`.
    """
    mod = int(alignment_mod)
    if mod <= 0:
        raise ValueError("alignment_mod must be > 0")
    if int(decision_block) <= 0:
        raise ValueError("decision_block must be > 0")

    delay = max(0, int(activation_delay_intervals))

    if t2_intervals is None:
        blocks = int(t2_blocks)
        if blocks <= 0:
            raise ValueError("t2_blocks must be > 0 when t2_intervals is not provided")
        intervals = int(math.ceil(blocks / mod))
    else:
        intervals = int(t2_intervals)

    intervals = max(int(min_t2_intervals), intervals)

    start = ceil_to_mod(int(decision_block), mod) + delay * mod
    end = int(start) + intervals * mod
    return int(start), int(end), int(intervals * mod)


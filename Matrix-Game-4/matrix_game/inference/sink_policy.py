"""Sink RoPE policies shared by every inference runtime.

absolute  The sink keeps the position it was written at (frame zero), so its distance to the current
          block grows with the session.
fixed     Once the window is full (position > window) the sink's keys are rotated to position - window,
          a constant distance behind the current block.
future    Once the window is full the sink's keys are rotated to position + FUTURE_SINK_OFFSET, a
          constant distance ahead of the current block. Racing uses this policy: in 4001- and
          6001-frame racing rollouts it kept buildings intact where the absolute sink melted them.

Only the sink's keys move; its values and every window frame keep their absolute positions.
"""

SINK_ROPE_MODES = ("absolute", "fixed", "future")
FUTURE_SINK_OFFSET = 11


def check_sink_rope_mode(mode):
    if mode not in (None,) + SINK_ROPE_MODES:
        raise ValueError(f"sink_rope_mode must be one of {SINK_ROPE_MODES}, got {mode}")


def rotated_sink_position(mode, position, window):
    """Frame position the sink's keys are rotated to for a block at `position`, or None to keep the
    cached absolute keys (absolute mode, or the window is not full yet)."""
    check_sink_rope_mode(mode)
    if mode in (None, "absolute") or position <= window:
        return None
    if mode == "fixed":
        return position - window
    return position + FUTURE_SINK_OFFSET


def rope_positions_needed(mode, positions):
    """RoPE table rows for `positions` frame positions under `mode` (the future sink reaches past the
    last block by FUTURE_SINK_OFFSET)."""
    check_sink_rope_mode(mode)
    return positions + (FUTURE_SINK_OFFSET if mode == "future" else 0)

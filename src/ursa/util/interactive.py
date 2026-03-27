# src/ursa/util/interactive.py
from __future__ import annotations

from typing import Optional


def timed_input_with_countdown(prompt: str, timeout: int) -> Optional[str]:
    """
    Read a line with a per-second countdown. Returns:
      - the user's input (str) if provided,
      - None if timeout expires,
      - None if non-interactive or timeout<=0.
    No bracketed prefixes are printed (clean output for all prompts).
    """
    import sys
    import time

    # Non-interactive or disabled timeout → default immediately (no noisy prefix)
    try:
        is_tty = sys.stdin.isatty()
    except Exception:
        is_tty = False

    if not is_tty:
        print("(non-interactive) selecting default . . .")
        return None
    if timeout <= 0:
        print("(timeout disabled) selecting default . . .")
        return None

    # Show prompt and run a 1s polling loop
    deadline = time.time() + timeout
    print(prompt, end="", flush=True)

    try:
        import select

        while True:
            remaining = int(max(0, deadline - time.time()))
            if remaining in {30, 10, 5, 4, 3, 2, 1}:
                print(
                    f"\n{remaining} seconds left . . .  (Ctrl-C to abort)",
                    flush=True,
                )
                print(prompt, end="", flush=True)
            if remaining <= 0:
                print()
                return None

            rlist, _, _ = select.select([sys.stdin], [], [], 1.0)
            if rlist:
                line = sys.stdin.readline()
                return None if line is None else line.strip()

    except Exception:
        # Fallback if select is unavailable
        try:
            return input()
        except KeyboardInterrupt:
            raise

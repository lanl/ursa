"""ASV tracks startup readiness, excluding teardown from the measured value."""

import json
import sys

from ._startup import SCENARIOS, measure


class Startup:
    params = list(SCENARIOS)
    param_names = ["scenario"]
    unit = "seconds"
    timeout = 180

    def track_max_ready(self, scenario):
        samples = [
            measure(scenario, timeout=30, headless=sys.platform == "win32")[0]
            for _ in range(5)
        ]
        sys.stderr.write(
            json.dumps({"scenario": scenario, "samples_seconds": samples})
            + "\n"
        )
        return max(samples)

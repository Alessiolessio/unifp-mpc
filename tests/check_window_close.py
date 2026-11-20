"""GUI smoke check on a graphical Linux desktop. Close during the second episode."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Timer
from unittest.mock import patch

from view_box import Playback, main


if __name__ == "__main__":
    with TemporaryDirectory() as tmp:
        log = Path(tmp) / "episodes.jsonl"
        play = Playback.play

        def close_second(self, trace, goal, paths, planning):
            play(self, trace, goal, paths, planning)
            if self.label.startswith("Medium") and paths is None:
                # Close the real window while the controller and warmup display are active.
                Timer(.06, self.window.close).start()

        with patch.object(Playback, "play", close_second), patch.object(
            sys, "argv", ["view-box", "--episodes", "2", "--seconds", ".2", "--pause", "0",
                          "--horizon", "1.5", "--log", str(log)]
        ):
            main()
        records = [json.loads(line) for line in log.read_text().splitlines()]
        assert [r["status"] for r in records] == ["time_limit", "interrupted"]
        assert records[0]["failures"] == 1 and records[0]["prefix_max_error"] == 0
        print("Timed out, reset, and closed the window during active playback.")

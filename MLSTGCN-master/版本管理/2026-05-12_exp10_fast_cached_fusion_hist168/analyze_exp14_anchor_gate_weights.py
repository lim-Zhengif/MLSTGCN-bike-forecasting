import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("analyze_exp13_anchor_gate_weights.py")),
        run_name="__main__",
    )

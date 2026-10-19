"""Generate saved explanatory figures from the original CADRE benchmark."""

import argparse
import json
from pathlib import Path

from visualizations.common import configure_style, load_case
from visualizations.dynamics import orbit, attitude, reaction_wheel
from visualizations.environment import sun, solar, thermal
from visualizations.operations import power, battery, communication, parameters

PLOTS = {
    "orbit": orbit,
    "attitude": attitude,
    "sun": sun,
    "solar": solar,
    "thermal": thermal,
    "power": power,
    "battery": battery,
    "reaction_wheel": reaction_wheel,
    "communication": communication,
    "parameters": parameters,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem",
        nargs="+",
        choices=PLOTS,
        help="Generate only these figures; the default is all ten",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "figures"
    )
    args = parser.parse_args(argv)

    configure_style()
    case = load_case()
    args.output.mkdir(parents=True, exist_ok=True)
    measurements = {
        "source": "Original CADRE test/data1346.pkl, case 0",
        "nodes": case.n,
        "duration_hours": float(case.hours[-1]),
        "scope": "Benchmark signals and Python reference reconstructions; no new AMIGO solve",
        "comparisons": {},
    }

    for name in args.problem or PLOTS:
        print(f"Generating {name}...", flush=True)
        measurements["comparisons"][name] = PLOTS[name](case, args.output)

    (args.output / "measurements.json").write_text(
        json.dumps(measurements, indent=2) + "\n"
    )
    print(f"\nFigures saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()

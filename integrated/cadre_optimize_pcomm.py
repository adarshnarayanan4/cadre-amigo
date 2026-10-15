"""Run the integrated CADRE communication and design optimization."""

from integrated.cadre_model import build_model
from integrated.cadre_optimizer import optimize_case
from integrated.cadre_options import parse_args
from integrated.cadre_reference import prepare_case
from integrated.cadre_results import validate_and_save_results


def main(argv=None):
    """Prepare, build, optimize, and independently validate one CADRE case."""
    args = parse_args(argv)
    case = prepare_case(args)

    model = build_model(case, args)

    result = optimize_case(model, case, args)

    return validate_and_save_results(case, args, result)


if __name__ == "__main__":
    main()

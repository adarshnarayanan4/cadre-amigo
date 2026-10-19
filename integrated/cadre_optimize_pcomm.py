"""Run the integrated CADRE communication and design optimization."""

from integrated.model.builder import build_model
from integrated.optimization.driver import optimize_case
from integrated.optimization.options import parse_args
from integrated.reference.case import prepare_case
from integrated.reporting.results import validate_and_save_results


def main(argv=None):
    """Prepare, build, optimize, and independently validate one CADRE case."""
    args = parse_args(argv)
    case = prepare_case(args)

    model = build_model(case, args)

    result = optimize_case(model, case, args)

    return validate_and_save_results(case, args, result)


if __name__ == "__main__":
    main()

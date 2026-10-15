"""Consistent external-cache and objective probes for CADRE optimization."""

from amigo.algorithm.evaluator import Evaluator


class ConsistentEvaluator(Evaluator):
    def evaluate_hessian(self, state):
        if not state.hessian_current:
            self.problem.update(state.current.get_solution())

        super().evaluate_hessian(state)

    def evaluate_directional_derivative(self, state):
        if not state.residual_current:
            self.evaluate_residual(state)

        if not state.step_current:
            raise ValueError("Step not current at this point")

        point = self.problem.create_vector()
        gradient = self.problem.create_vector()

        current = state.current.get_solution()
        constraints = self.problem.get_constraint_indices()

        point.copy(current)
        point.fill_at(constraints, 0.0)

        self.problem.update(point)

        try:
            self.problem.gradient(state.obj_scale, point, gradient)

            # Even at lambda=0, dL/dlambda=c(x). Objective line search must
            # exclude c(x)^T dlambda from its directional derivative.
            gradient.fill_at(constraints, 0.0)

            derivative = self.problem.dot(state.step.get_solution(), gradient)
        finally:
            self.problem.update(current)

        return derivative + self.optimizer.compute_log_barrier_derivative(
            state.mu, state.current, state.step
        )

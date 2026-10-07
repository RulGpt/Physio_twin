from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.statespace.mlemodel import MLEModel

from .state_space import OBS_DIM, STATE_DIM, StateSpaceConfig


@dataclass(frozen=True)
class ParameterRecoveryResult:
    """Fitted matrices and optimization diagnostics."""

    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    param_names: tuple[str, ...]
    unconstrained_params: np.ndarray
    constrained_params: np.ndarray
    converged: bool
    llf: float
    optimizer_message: str
    optimizer_iterations: int | None
    optimizer_function_calls: int | None

def _log_positive(x: float) -> float:
    return float(np.log(max(x, 1e-12)))


class PopulationCoreMLE(MLEModel):
    """14-parameter observable-core PHYSIO-TWIN model.

    Free parameters:
      - A00, A01, A10, A11
      - B00..B12 for fatigue/recovery rows
      - Q00, Q11
      - R00, R11

    The latent-load row of A/B and Q22 remain fixed at the supplied
    population/test configuration for this first recovery experiment.

    A stability-oriented parameter transform is used for the observable
    A block, with an additional likelihood-level stability safeguard.
    Variances are transformed to remain strictly positive.
    """

    def __init__(
        self,
        observations: np.ndarray,
        inputs: np.ndarray,
        fixed_config: StateSpaceConfig,
    ):
        observations = np.asarray(observations, dtype=float)
        inputs = np.asarray(inputs, dtype=float)

        if observations.ndim != 2 or observations.shape[1] != OBS_DIM:
            raise ValueError("observations must have shape (n, 2)")
        if inputs.ndim != 2 or inputs.shape[1] != 3:
            raise ValueError("inputs must have shape (n, 3)")
        if len(observations) != len(inputs):
            raise ValueError("observations and inputs must have equal length")

        self.inputs = inputs
        self.fixed_config = fixed_config

        # Time-varying deterministic input contribution is represented through
        # a time-varying design of the transition update in update().
        super().__init__(
            observations,
            k_states=STATE_DIM,
            initialization="approximate_diffuse",
        )

        self.ssm.initialize_known(
            fixed_config.initial_state,
            fixed_config.initial_covariance,
        )

        self["design"] = fixed_config.C
        self["transition"] = fixed_config.A
        self["selection"] = np.eye(STATE_DIM)
        self["state_cov"] = fixed_config.Q
        self["obs_cov"] = fixed_config.R

        self._param_names = (
            "A_00", "A_01", "A_10", "A_11",
            "B_00", "B_01", "B_02",
            "B_10", "B_11", "B_12",
            "log_Q_00", "log_Q_11",
            "log_R_00", "log_R_11",
        )

    @property
    def start_params(self) -> np.ndarray:
        A = self.fixed_config.A
        B = self.fixed_config.B
        Q = self.fixed_config.Q
        R = self.fixed_config.R

        # Statsmodels expects start_params in the constrained parameter
        # space by default. It will call untransform_params() internally
        # before optimization.
        return np.array([
            A[0, 0], A[0, 1], A[1, 0], A[1, 1],
            B[0, 0], B[0, 1], B[0, 2],
            B[1, 0], B[1, 1], B[1, 2],
            Q[0, 0], Q[1, 1],
            R[0, 0], R[1, 1],
        ])
    
    @property
    def param_names(self):
        return list(self._param_names)

    def transform_params(self, unconstrained):
        params = np.asarray(unconstrained).copy()
        # Constrain the observable A block to a safe numerical range.
        # This is an optimization constraint, not a physiological claim.
        params[:4] = 0.9 * np.tanh(params[:4])
        # Variances must remain strictly positive.
        params[10:] = np.exp(params[10:]) + 1e-6
        return params

    def untransform_params(self, constrained):
        params = np.asarray(constrained).copy()
        # Inverse of 0.9 * tanh(x)
        scaled = np.clip(params[:4] / 0.9, -0.999999, 0.999999)
        params[:4] = np.arctanh(scaled)
        params[10:] = np.log(np.maximum(params[10:] - 1e-6, 1e-12))
        return params

    def _matrices(self, constrained):
        p = np.asarray(constrained)
        dtype = np.result_type(constrained, float)
        A = self.fixed_config.A.astype(dtype, copy=True)
        B = self.fixed_config.B.astype(dtype, copy=True)
        Q = self.fixed_config.Q.astype(dtype, copy=True)
        R = self.fixed_config.R.astype(dtype, copy=True)


        A[0, 0], A[0, 1], A[1, 0], A[1, 1] = p[:4]
        B[0, :] = p[4:7]
        B[1, :] = p[7:10]
        Q[0, 0], Q[1, 1] = p[10:12]
        R[0, 0], R[1, 1] = p[12:14]

        return A, B, Q, R

    def update(self, params, transformed=True, **kwargs):
        params = super().update(
            params,
            transformed=transformed,
            **kwargs,
        )
        constrained = (
            np.asarray(params)
            if transformed
            else self.transform_params(params)
        )
        A, B, Q, R = self._matrices(constrained)

        self["transition"] = A
        self["state_cov"] = Q
        self["obs_cov"] = R

        # The synthetic model uses:
        #
        #     X_t = A X_{t-1} + B U_{t-1} + w_{t-1}
        #
        # Therefore U[t-1] drives X[t].
        # At t=0 there is no previous input, so the intercept is zero.

        intercept = np.zeros(
            (STATE_DIM, len(self.inputs)),
            dtype=np.result_type(B, float),
        )

        intercept[:, 1:] = B @ self.inputs[:-1].T

        self["state_intercept"] = intercept      
        self["state_intercept"] = intercept

        self._candidate_stable = np.max(np.abs(np.linalg.eigvals(A))) < 0.995
        self._candidate_positive = (
            np.all(np.linalg.eigvalsh(Q) >= -1e-10)
            and np.all(np.linalg.eigvalsh(R) > 0)
        )

    def loglike(self, params, *args, **kwargs):
        # Statsmodels passes transformed=False during optimization.
        # Direct calls default to transformed=True unless explicitly specified.
        transformed = kwargs.pop("transformed", None)

        if transformed is None and args:
            first = args[0]
            if isinstance(first, dict):
                transformed = first.get("transformed", True)
            else:
                transformed = True

        if transformed is None:
            transformed = True

        if transformed:
            constrained = np.asarray(params)
        else:
            constrained = self.transform_params(params)

        self.update(
            constrained,
            transformed=True,
        )

        if not getattr(self, "_candidate_stable", False):
            return -1e12

        if not getattr(self, "_candidate_positive", False):
            return -1e12

        return super().loglike(
            constrained,
            transformed=True,
        )

    def fit_model(
        self,
        maxiter: int = 150,
        method: str = "lbfgs",
        disp: bool = False,
    ):
        return self.fit(
            start_params=self.start_params,
            method=method,
            maxiter=maxiter,
            disp=disp,
        )


def fit_population_core(
    observations: np.ndarray,
    inputs: np.ndarray,
    initial_config: StateSpaceConfig,
    maxiter: int = 150,
) -> ParameterRecoveryResult:
    model = PopulationCoreMLE(
        observations,
        inputs,
        fixed_config=initial_config,
    )

    result = model.fit_model(maxiter=maxiter)

    # Statsmodels returns the fitted parameters in the constrained
    # parameter space for this model.
    constrained = np.asarray(result.params, dtype=float)

    # Convert the constrained fitted parameters back to the optimizer's
    # unconstrained representation only for bookkeeping/reporting.
    unconstrained = np.asarray(
        model.untransform_params(constrained),
        dtype=float,
    )

    A, B, Q, R = model._matrices(constrained)
    return ParameterRecoveryResult(
    A=A,
    B=B,
    Q=Q,
    R=R,
    param_names=tuple(model.param_names),
    unconstrained_params=unconstrained,
    constrained_params=constrained,
    converged=bool(
        result.mle_retvals.get("converged", False)
    ),
    llf=float(result.llf),
    optimizer_message=str(
        result.mle_retvals.get("message", "")
    ),
    optimizer_iterations=result.mle_retvals.get("iterations"),
    optimizer_function_calls=result.mle_retvals.get("fcalls"),
)
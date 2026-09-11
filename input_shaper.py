from collections import deque

import numpy as np


class InputShaper:
    def __init__(
        self,
        sampling_time,
        natural_frequency,
        damping_ratio=0.0,
        shaper_type="ZVD",
        residual_vibration=0.05
    ):
        if sampling_time <= 0:
            raise ValueError("sampling_time must be positive")

        if natural_frequency <= 0:
            raise ValueError("natural_frequency must be positive")

        if not 0.0 <= damping_ratio < 1.0:
            raise ValueError(
                "damping_ratio must satisfy 0 <= zeta < 1" )

        self.sampling_time = float(sampling_time)
        self.natural_frequency = float(natural_frequency)
        self.damping_ratio = float(damping_ratio)
        self.shaper_type = shaper_type.upper()
        self.residual_vibration = float(residual_vibration)

        if self.shaper_type not in ("ZV", "ZVD", "ZVDD", "EI"):
            raise ValueError(
                "shaper_type must be 'ZV', 'ZVD', 'ZVDD' or 'EI'")

        if self.shaper_type == "EI":
            if not 0.0 < self.residual_vibration < 1.0:
                raise ValueError(
                    "residual_vibration must be between 0 and 1"
                )

        self.weights, impulse_times = self._compute_parameters()

        self.delay_samples = np.rint(
            impulse_times / self.sampling_time
        ).astype(int)

        self.actual_impulse_times = (
            self.delay_samples * self.sampling_time
        )

        self.max_delay = int(np.max(self.delay_samples))
        self.buffer = None
        self.reference_shape = None

    def _compute_parameters(self):
        zeta = self.damping_ratio

        root = np.sqrt(1.0 - zeta**2)
        damped_frequency = self.natural_frequency * root
        k = np.exp(-zeta * np.pi / root)

        if self.shaper_type == "ZV":
            weights = np.array([
                1.0,
                k
            ]) / (1.0 + k)

            impulse_times = np.array([
                0.0,
                np.pi / damped_frequency
            ])

        elif self.shaper_type == "ZVD":
            weights = np.array([
                1.0,
                2.0 * k,
                k**2
            ]) / (1.0 + k)**2

            impulse_times = np.array([
                0.0,
                np.pi / damped_frequency,
                2.0 * np.pi / damped_frequency
            ])

        elif self.shaper_type == "ZVDD":
            weights = np.array([
                1.0,
                3.0 * k,
                3.0 * k**2,
                k**3
            ]) / (1.0 + k)**3

            impulse_times = np.array([
                0.0,
                np.pi / damped_frequency,
                2.0 * np.pi / damped_frequency,
                3.0 * np.pi / damped_frequency
            ])

        elif self.shaper_type == "EI":
            v_max = self.residual_vibration

            weights = np.array([
                1.0 + v_max,
                2.0 * k * (1.0 - v_max),
                k**2 * (1.0 + v_max)
            ]) / (1.0 + k)**2

            impulse_times = np.array([
                0.0,
                np.pi / damped_frequency,
                2.0 * np.pi / damped_frequency
            ])

        return weights, impulse_times

    def reset(self):
        self.buffer = None
        self.reference_shape = None

    def shape(self, reference):
        reference = np.asarray(reference, dtype=float)

        if self.buffer is None:
            self.reference_shape = reference.shape

            self.buffer = deque(
                (
                    reference.copy()
                    for _ in range(self.max_delay + 1)
                ),
                maxlen=self.max_delay + 1
            )

        elif reference.shape != self.reference_shape:
            raise ValueError(
                f"Expected reference shape {self.reference_shape}, "
                f"received {reference.shape}"
            )

        else:
            self.buffer.append(reference.copy())

        shaped_reference = np.zeros_like(reference)

        for weight, delay in zip(
            self.weights,
            self.delay_samples
        ):
            shaped_reference += (
                weight
                * self.buffer[-1 - int(delay)]
            )

        return shaped_reference

"""Unitree Go2W wheel-legged quadruped."""

from src.utils.mjlab_compat import enable_continuous_velocity_actuators

# Runs in each worker when the asset or its pickled spec_fn is imported.
enable_continuous_velocity_actuators()

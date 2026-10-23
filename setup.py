"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "mujoco==3.5.0",
    "mujoco-warp==3.5.0",
    "warp-lang==1.12.0",
    "scipy>=1.14.0",
    # RSL-RL 5.0.1 passes start_method, removed in W&B 0.29.
    "wandb>=0.22.3,<0.29",
]

# Installation operation
setup(
    name="unitree_rl_mjlab",
    packages=["src"],
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)

from setuptools import find_packages, setup
from typing import List


def get_requirements() -> List[str]:
    """Read package dependencies from requirements.txt."""
    try:
        with open("requirements.txt", "r") as file:
            requirements = file.read().splitlines()

        requirements = [
            requirement.strip()
            for requirement in requirements
            if requirement.strip()
            and requirement.strip() != "-e ."
        ]

        return requirements

    except FileNotFoundError:
        return []


setup(
    name="wake_uncertainty",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=get_requirements(),
    description="Uncertainty-aware surrogate modelling for offshore wind farm wake losses",
    author="Varun Kokkiligadda",
    author_email="varun.kokkiligadda25@imperial.ac.uk",
    python_requires=">=3.10",
)
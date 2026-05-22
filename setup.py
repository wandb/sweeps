from setuptools import setup

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read().splitlines()

with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="sweeps",
    version="0.2.1.dev1",
    description="Weights and Biases Hyperparameter Sweeps Engine.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Weights & Biases",
    author_email="support@wandb.com",
    url="https://github.com/wandb/sweeps",
    packages=["sweeps", "sweeps.config"],
    package_dir={"sweeps": "src/sweeps"},
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)

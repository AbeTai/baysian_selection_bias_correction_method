from setuptools import setup, find_packages

setup(
    name="correction_bias_based_on_statistical_decision_theory",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy==2.1.3",
        "pandas==2.2.3",
        "scipy==1.14.1",
        "scikit-learn==1.5.2",
        "matplotlib==3.9.2",
        "tqdm==4.67.0",
        "polyagamma==2.0.1",
        "setuptools",
    ],
)
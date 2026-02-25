from setuptools import setup, find_packages

setup(
    name="confusionrag",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "rank_bm25>=0.2.2",
        "transformers>=4.28.0",
        "torch>=1.13.0",
        "tabulate>=0.9.0",
    ],
    extras_require={
        "dense": ["sentence-transformers>=2.2.0"],
        "test": ["pytest>=7.0", "pytest-cov"],
    },
)

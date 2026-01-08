from setuptools import setup, find_packages

setup(
    name="pytorch_to_c",
    version="0.1.0",
    description="PyTorch to C Compiler for Microcontrollers",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
        ],
    },
    python_requires=">=3.8",
)


from setuptools import find_packages, setup
from pathlib import Path

base_path = Path(__file__).resolve().parent
with open(base_path / "readme.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open(base_path / "requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="MetaHarmonizerTest",
    version="0.2.4",
    description="Metadata harmonizer for cBioPortal clinical metadata using LM and LLM",
    package_dir={"": "."},
    packages=find_packages(where="."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sehyun Oh/Abhilash Dhal",
    author_email="adhalbiophysics@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=required,
    python_requires=">3.10",
)

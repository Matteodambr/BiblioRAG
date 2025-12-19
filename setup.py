"""Setup script for BiblioRAG."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bibliorag",
    version="0.1.0",
    author="BiblioRAG Contributors",
    description="RAG wrapper for Mendeley library using Paper-QA2 and Gemini",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Matteodambr/BiblioRAG",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "bibliorag=bibliorag.cli:main",
        ],
    },
)

# setup.py
from setuptools import setup, find_packages

setup(
    name="rag_app",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain-openai",
        "langchain-chroma",
        "langchain-core",
        "rank-bm25",
        "pydantic",
    ],
)
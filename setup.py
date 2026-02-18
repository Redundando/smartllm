from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="smartllm",
    version="0.1.0",
    author="Arved Klöhn",
    author_email="arved.kloehn@gmail.com",
    description="A unified async Python wrapper for multiple LLM providers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Redundando/smartllm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "bedrock": ["aioboto3>=12.0.0"],
        "all": ["openai>=1.0.0", "aioboto3>=12.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.10.0",
        ],
    },
    keywords="llm openai bedrock claude gpt async ai ml",
    project_urls={
        "Bug Reports": "https://github.com/Redundando/smartllm/issues",
        "Source": "https://github.com/Redundando/smartllm",
    },
)

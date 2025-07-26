from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-challenge-hcm-chatbot",
    version="1.0.0",
    author="AI Challenge HCM Team",
    author_email="team@aichallengehcm.com",
    description="Multimodal Virtual Assistant for AI Challenge HCM 2025",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ai-challenge-hcm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "black>=23.11.0",
            "flake8>=6.1.0",
        ],
        "gpu": [
            "torch>=2.1.1",
            "faiss-gpu>=1.7.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-challenge-hcm=src.api:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
    keywords="ai machine-learning multimodal search vietnamese fastapi",
    project_urls={
        "Bug Reports": "https://github.com/your-username/ai-challenge-hcm/issues",
        "Source": "https://github.com/your-username/ai-challenge-hcm",
        "Documentation": "https://github.com/your-username/ai-challenge-hcm#readme",
    },
) 
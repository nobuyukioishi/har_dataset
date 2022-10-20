import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="har_dataset",
    version="0.0.1",
    author="Nobuyuki Oishi",
    author_email="n.oishi@sussex.ac.uk",
    description="A Python package for sensor-based human activity recognition datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NobuyukiOISHI/hampel_filter",
    py_modules=["har_dataset"],
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "pandas",
        "requests",
        "tqdm"
    ],
    extras_requires={
        "dev": [
            "pytest>=6.0.2"
        ]
    },
    python_requires='>=3.8',
)
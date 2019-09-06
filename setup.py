import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gpu-lib",
    version="0.0.1",
    author="brynhayder",
    description="Tools for NVIDIA GPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brynhayder/gpu-lib",
    packages=setuptools.find_packages(),
    install_requires=[
        'py3nvml',
        'xmltodict'
        ],
    entry_points={
        'console_scripts': [
            'exec_when_free = gpulib.exec_when_free:main',
            ]
        },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        # The following may be untrue and needs to be checked!
        "Operating System :: OS Independent",
    ],
)

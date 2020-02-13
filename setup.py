from setuptools import setup, find_packages

Name = "Colorful_Image"
with open("README.md", "r") as fh:
    long_description = fh.read()
INSTALL_REQUIRES = [
    "tensorflow"
]

DEV_REQUIRES = [
    "pytest",
    "pycodestyle",
    "setuptools",
    "pylint",
    "pytest-html",
    "pytest-xdist",
    "pytest-forked",
    "pytest-cov"
]

setup(
    name=Name,
    version="0.1.0",
    author='Rafik_G & Sid_A',
    package=find_packages(),
    description=long_description,
    install_requires=INSTALL_REQUIRES,
    extras_requires={
        'dev': DEV_REQUIRES
    }
)

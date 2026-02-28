from setuptools import setup, find_packages

requirements = []

__version__ = "0.0.1"
setup(
    name="crowd_nav_rl",
    version=__version__,
    description="Crowd Navigation RL",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.11",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages(include=["new_rl"]),
    # install_requires=requirements,
)

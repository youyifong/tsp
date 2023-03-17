import setuptools
from setuptools import setup

install_deps = []

    
setup(
    name="tsp",
    license="BSD",
    author="Sunwoo Han and Youyi Fong",
    author_email="youyifong@gmail.com",
    description="Utility Functions from Sunwoo Han and Youyi Fong",
    url="https://github.com/youyifong/tsp",
    packages=setuptools.find_packages(),
    use_scm_version=True,
    install_requires = install_deps,
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ),
)

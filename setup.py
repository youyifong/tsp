import setuptools
from setuptools import setup

install_deps = ['numpy>=1.20.0'
                ]

    
setup(
    name="syotil",
    license="BSD",
    author="Sunwoo Han and Youyi Fong",
    author_email="sunwooya@gmail.com",
    description="utility functions",
    url="https://github.com/youyifong/syotil",
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

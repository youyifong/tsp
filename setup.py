import setuptools
from setuptools import setup

install_deps = ['numpy>=1.20.0'
                ]

    
setup(
    name="cellmask",
    license="BSD",
    author="Sunwoo Han and Youyi Fong",
    author_email="youyifong@gmail.com",
    description="Cell segmentation using Mask R-CNN",
    url="https://github.com/youyifong/cellmask",
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

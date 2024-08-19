from setuptools import Extension, find_packages, setup
from pathlib import Path

import numpy
from Cython.Build import cythonize

_DIR = Path(__file__).parent

exts = [
    Extension(
        name="core",
        sources=["core.pyx"],
    )
]

setup(
    name="monotonic_align",
    version="1.0",
    description="",
    include_dirs=[numpy.get_include()],
    modules=["core",],
    ext_modules=cythonize(exts, language_level=3),
    python_requires=">=3.9.0",
)

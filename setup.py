import setuptools
import cmake_build_extension

import inspect
import os
import sys
from pathlib import Path

init_py = inspect.cleandoc(
    """
    import cmake_build_extension
    with cmake_build_extension.build_extension_env():
        from .lib import RLPack
    """
)

CIBW_CMAKE_OPTIONS = []
if "CIBUILDWHEEL" in os.environ and os.environ["CIBUILDWHEEL"] == "1":
    if sys.platform == "linux":
        CIBW_CMAKE_OPTIONS += ["-DCMAKE_INSTALL_LIBDIR=lib"]

setuptools.setup(
    name="RLPack",
    version="0.0.1",
    author="Kartik Rajeshwaran",
    author_email="kartik.rajeshwaran@gmail.com",
    description="Implementation of RL Algorithms",
    long_description="Implementation of RL Algorithms with C++ backend and made available to Gym Frontend "
                     "with Python Bindings",
    long_description_content_type="text/markdown",
    url="",
    project_urls={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": ["RLPack_entry = RLPack.bin.__main__:main"],
    },
    ext_modules=[
        cmake_build_extension.CMakeExtension(
            name="RLPack",
            install_prefix="rlpack",
            cmake_depends_on=["pybind11"],
            expose_binaries=["bin/RLPack"],
            write_top_level_init=init_py,
            source_dir=str(Path(__file__).parent.absolute()),
            cmake_configure_options=[
                                        "-DBUILD_SHARED_LIBS:BOOL=OFF",
                                    ] + CIBW_CMAKE_OPTIONS,
        ),
    ],
    cmdclass={
        "build_ext": cmake_build_extension.BuildExtension,
    },
)

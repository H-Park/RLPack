import os
import pathlib
import sys
import cmake_build_extension
import pathlib
with cmake_build_extension.build_extension_env():
    from .lib import RLPack

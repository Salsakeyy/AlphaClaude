import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        import subprocess
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        build_args = ["--config", "Release", "-j4"]

        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)

setup(
    name="alphaclaude",
    version="0.1.0",
    description="AlphaZero Chess Clone",
    ext_modules=[CMakeExtension("alphaclaude_cpp")],
    cmdclass={"build_ext": CMakeBuild},
    packages=["py_src"],
    python_requires=">=3.8",
)

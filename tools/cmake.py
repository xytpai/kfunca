"Manages CMake."

import os
import sys
import pybind11
from packaging.version import Version
from pathlib import Path
from subprocess import CalledProcessError, check_call, check_output
from typing import Any, cast


BUILD_DIR = 'build'


def _mkdir_p(d: str) -> None:
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"Failed to create folder {os.path.abspath(d)}: {e.strerror}"
        ) from e


def which(thefile: str) -> str | None:
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for d in path:
        fname = os.path.join(d, thefile)
        if os.access(fname, os.F_OK | os.X_OK) and not os.path.isdir(fname):
            return fname
    return None


class CMake:
    "Manages cmake."

    def __init__(self, parallel_build=None, build_dir=BUILD_DIR) -> None:
        self._cmake_command = CMake._get_cmake_command()
        self.build_dir = build_dir
        self.parallel_build = parallel_build

    @property
    def _cmake_cache_file(self) -> str:
        r"""Returns the path to CMakeCache.txt.

        Returns:
          string: The path to CMakeCache.txt.
        """
        return os.path.join(self.build_dir, "CMakeCache.txt")

    @staticmethod
    def _get_cmake_command() -> str:
        "Returns cmake command."
        cmake_command = which("cmake")
        cmake_version = CMake._get_version(cmake_command)
        _cmake_min_version = Version("3.18.0")
        if all(
            ver is None or ver < _cmake_min_version
            for ver in [cmake_version]
        ):
            raise RuntimeError("no cmake with version >= 3.18.0 found")
        return cmake_command

    @staticmethod
    def _get_version(cmd: str | None) -> Any:
        "Returns cmake version."
        if cmd is None:
            return None
        for line in check_output([cmd, "--version"]).decode("utf-8").split("\n"):
            if "version" in line:
                return Version(line.strip().split(" ")[2])
        raise RuntimeError("no version found")

    @staticmethod
    def defines(args, **kwargs) -> None:
        "Adds definitions to a cmake argument list."
        for key, value in sorted(kwargs.items()):
            if value is not None:
                args.append(f"-D{key}={value}")

    def run(self, args: list[str], env: dict[str, str]) -> None:
        "Executes cmake with arguments and an environment."
        command = [self._cmake_command] + args
        print(" ".join(command))
        try:
            check_call(command, cwd=self.build_dir, env=env)
        except (CalledProcessError, KeyboardInterrupt):
            # This error indicates that there was a problem with cmake, the
            # Python backtrace adds no signal here so skip over it by catching
            # the error and exiting manually
            sys.exit(1)
    
    def generate(self, rerun: bool, output_dir: str):
        if rerun and os.path.isfile(self._cmake_cache_file):
            os.remove(self._cmake_cache_file)
        args = ['..']
        _mkdir_p(self.build_dir)
        pybind11_dir = pybind11.__path__[0]
        CMake.defines(args, 
                      pybind11_DIR=os.path.join(pybind11_dir, 'share/cmake/pybind11'),
                      CMAKE_LIBRARY_OUTPUT_DIRECTORY=output_dir)
        self.run(args, os.environ)
    
    def build(self):
        if not self.parallel_build:
            self.run(['--build', '.'], os.environ)
        elif isinstance(self.parallel_build, int):
            nworkers = str(self.parallel_build)
            self.run(['--build', '.', '-j', nworkers], os.environ)
        else:
            self.run(['--build', '.', '--parallel'], os.environ)

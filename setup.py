import os
import shutil
import pathlib
from setuptools import Extension, find_packages, setup, Command
from setuptools.command.build_ext import build_ext
from tools.cmake import CMake


package_name = 'kfunca'
version = '0.1.1'
base_dir = os.path.dirname(os.path.abspath(__file__))
default_parallel_build = 4  # Default number of parallel jobs for building


class BuildExt(build_ext):
    def run(self):
        for ext in self.extensions:
            if isinstance(ext, Extension):
                self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cmake = CMake(default_parallel_build)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name)).parent.absolute()
        cmake.generate(rerun=True, output_dir=extdir)
        cmake.build()


class CleanCmd(Command):
    user_options = []

    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass

    def run(self):
        import glob
        import re

        with open(".gitignore") as f:
            ignores = f.read()
            pat = re.compile(r"^#( BEGIN NOT-CLEAN-FILES )?")
            for wildcard in filter(None, ignores.split("\n")):
                match = pat.match(wildcard)
                if match:
                    if match.group(1):
                        # Marker is found and stop reading .gitignore.
                        break
                    # Ignore lines which begin with '#'.
                else:
                    # Don't remove absolute paths from the system
                    wildcard = wildcard.lstrip("./")

                    for filename in glob.glob(wildcard):
                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)


def main():
    setup(
        name=package_name,
        version=version,
        description=("NA"),
        ext_modules=[Extension('kfunca', sources=[])],
        cmdclass={
            "build_ext": BuildExt,
            "clean": CleanCmd,
        },
    )


if __name__ == "__main__":
    main()

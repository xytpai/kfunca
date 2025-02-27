import os
import pathlib
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from tools.cmake import CMake


package_name = 'kfunca'
version = '0.1.1'
base_dir = os.path.dirname(os.path.abspath(__file__))


class BuildExt(build_ext):
    def run(self):
        for ext in self.extensions:
            if isinstance(ext, Extension):
                self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cmake = CMake()
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name)).parent.absolute()
        cmake.generate(rerun=True, output_dir=extdir)
        cmake.build()


def main():
    setup(
        name=package_name,
        version=version,
        description=("NA"),
        ext_modules=[Extension('kfunca', sources=[])],
        cmdclass={"build_ext": BuildExt},
    )


if __name__ == "__main__":
    main()

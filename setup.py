import os
from setuptools import Extension, find_packages, setup
from tools.cmake import CMake


package_name = 'kfunca'
version = '0.1.1'
base_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    cmake = CMake()
    cmake.generate(rerun=True)


if __name__ == "__main__":
    main()

cmake_minimum_required(VERSION 3.12)
project(kfunca)

message(pybind11_DIR)
find_package(pybind11 REQUIRED)

pybind11_add_module(kfunca src/register.cpp)

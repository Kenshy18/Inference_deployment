cmake_minimum_required(VERSION 3.20)
project(humanlike_temporal_polygon LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(Python 3.10 EXACT COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

pybind11_add_module(native_temporal_polygon MODULE native_temporal_polygon.cpp)
target_compile_features(native_temporal_polygon PRIVATE cxx_std_17)
target_compile_options(native_temporal_polygon PRIVATE -O3 -DNDEBUG -Wall -Wextra -Wpedantic)
target_link_libraries(native_temporal_polygon PRIVATE OpenMP::OpenMP_CXX)

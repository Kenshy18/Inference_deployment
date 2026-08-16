cmake_minimum_required(VERSION 3.20)

project(mask_pipeline_native_interval LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(Python 3.10 EXACT COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenCV 4.8 REQUIRED COMPONENTS core imgproc)
find_package(OpenMP REQUIRED)

pybind11_add_module(native_interval_metrics MODULE native_interval_metrics.cpp)
target_compile_features(native_interval_metrics PRIVATE cxx_std_17)
target_compile_options(
  native_interval_metrics
  PRIVATE
    -O3
    -DNDEBUG
    -Wall
    -Wextra
    -Wpedantic
    -ffp-contract=off
)
target_link_libraries(native_interval_metrics PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

# Keep the private build environment's OpenCV runtime discoverable when the
# module is imported by the deployed Production Python 3.10 interpreter.
if(DEFINED ENV{CONDA_PREFIX})
  set_target_properties(
    native_interval_metrics
    PROPERTIES
      BUILD_RPATH "$ENV{CONDA_PREFIX}/lib"
      INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib"
  )
endif()

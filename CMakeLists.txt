cmake_minimum_required(VERSION 3.20)
project(unifp_mpc LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)
set(mujoco_dir "${Python_SITEARCH}/mujoco")
find_file(mujoco_lib libmujoco.so.3.6.0 PATHS "${mujoco_dir}" NO_DEFAULT_PATH REQUIRED)

pybind11_add_module(native_probe tests/native_probe.cpp)
target_compile_features(native_probe PRIVATE cxx_std_17)
target_include_directories(native_probe PRIVATE "${mujoco_dir}/include")
target_link_libraries(native_probe PRIVATE onnxruntime::onnxruntime "${mujoco_lib}")

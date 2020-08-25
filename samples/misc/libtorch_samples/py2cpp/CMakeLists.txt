cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(py2cpp)

find_package(Torch REQUIRED)

add_executable(py2cpp py2cpp.cpp)
target_link_libraries(py2cpp "${TORCH_LIBRARIES}")
set_property(TARGET py2cpp PROPERTY CXX_STANDARD 11)

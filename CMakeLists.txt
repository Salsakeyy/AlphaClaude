cmake_minimum_required(VERSION 3.15)
project(alphaclaude LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 REQUIRED)
find_package(OpenMP QUIET)

pybind11_add_module(alphaclaude_cpp
    cpp_src/bitboard.cpp
    cpp_src/magic.cpp
    cpp_src/position.cpp
    cpp_src/movegen.cpp
    cpp_src/game.cpp
    cpp_src/mcts.cpp
    cpp_src/bindings.cpp
)

target_include_directories(alphaclaude_cpp PRIVATE cpp_src)
target_compile_options(alphaclaude_cpp PRIVATE -O3 -march=native)

if(OpenMP_CXX_FOUND)
    target_link_libraries(alphaclaude_cpp PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(alphaclaude_cpp PRIVATE HAS_OPENMP)
    message(STATUS "OpenMP found - parallel MCTS enabled")
else()
    message(STATUS "OpenMP not found - falling back to single-threaded MCTS")
endif()

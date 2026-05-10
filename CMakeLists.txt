cmake_minimum_required(VERSION 3.15)
project(AlphaTaflEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include FetchContent
include(FetchContent)

# ------------------------------------------------------------------------------
# PyTorch C++ (libtorch from pip installation)
# ------------------------------------------------------------------------------
# Use the PyTorch pip package's C++ headers and libraries.
# This avoids downloading a separate libtorch package.
set(PYTHON_SITE_PACKAGES "${Python_SITELIB}")
if(NOT PYTHON_SITE_PACKAGES)
    execute_process(
        COMMAND python -c "import site; print(site.getsitepackages()[-1])"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

set(TORCH_ROOT "${PYTHON_SITE_PACKAGES}/torch")
set(TORCH_INCLUDE "${TORCH_ROOT}/include")
set(TORCH_LIB "${TORCH_ROOT}/lib")

message(STATUS "TORCH_ROOT: ${TORCH_ROOT}")
message(STATUS "TORCH_INCLUDE: ${TORCH_INCLUDE}")
message(STATUS "TORCH_LIB: ${TORCH_LIB}")

# Collect torch libraries
file(GLOB TORCH_LIBS "${TORCH_LIB}/*.lib")

# ------------------------------------------------------------------------------
# GoogleTest
# ------------------------------------------------------------------------------
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ------------------------------------------------------------------------------
# Pybind11
# ------------------------------------------------------------------------------
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# ------------------------------------------------------------------------------
# Engine Core Library
# ------------------------------------------------------------------------------
# We compile the core logic as a static library so both tests and Python bindings can use it.
add_library(core_engine STATIC
    src/engine/game_state.cpp
    src/engine/mcts.cpp
    src/inference/inference_engine.cpp
)
target_include_directories(core_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/engine
    ${CMAKE_CURRENT_SOURCE_DIR}/src/inference
    ${TORCH_INCLUDE}
    ${TORCH_INCLUDE}/torch/csrc/api/include
)
target_link_libraries(core_engine PUBLIC ${TORCH_LIBS})

# ------------------------------------------------------------------------------
# C++ Unit Tests
# ------------------------------------------------------------------------------
enable_testing()

add_executable(engine_tests
    tests/cpp/test_game_state.cpp
)
target_link_libraries(engine_tests PRIVATE core_engine GTest::gtest_main)

add_executable(regression_tests
    tests/cpp/test_regression.cpp
)
target_link_libraries(regression_tests PRIVATE core_engine GTest::gtest_main)

add_executable(profile_tests
    tests/cpp/test_profile.cpp
)
target_link_libraries(profile_tests PRIVATE core_engine GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(engine_tests)
gtest_discover_tests(regression_tests)
gtest_discover_tests(profile_tests)

# ------------------------------------------------------------------------------
# Python Bindings
# ------------------------------------------------------------------------------
pybind11_add_module(alphatafl_engine src/bindings/bind.cpp)

target_include_directories(alphatafl_engine PRIVATE
    ${TORCH_INCLUDE}
    ${TORCH_INCLUDE}/torch/csrc/api/include
)

target_link_libraries(alphatafl_engine PRIVATE
    core_engine
    ${TORCH_LIBS}
)

# On Windows, copy required DLLs next to the pyd so they are found at runtime
if(WIN32)
    file(GLOB TORCH_DLLS "${TORCH_LIB}/*.dll")
    foreach(dll ${TORCH_DLLS})
        add_custom_command(TARGET alphatafl_engine POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${dll}" $<TARGET_FILE_DIR:alphatafl_engine>
        )
    endforeach()
endif()

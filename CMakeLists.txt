cmake_minimum_required(VERSION 3.15)
project(AlphaTaflEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include FetchContent
include(FetchContent)

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
)
target_include_directories(core_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/engine)

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
target_link_libraries(alphatafl_engine PRIVATE core_engine)

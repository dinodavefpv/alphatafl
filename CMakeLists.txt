cmake_minimum_required(VERSION 3.15)
project(AlphaTaflEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include FetchContent
include(FetchContent)

# ------------------------------------------------------------------------------
# ONNX Runtime stand-alone C++
# ------------------------------------------------------------------------------
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime")
set(ONNXRUNTIME_INCLUDE "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

message(STATUS "ONNXRUNTIME_ROOT: ${ONNXRUNTIME_ROOT}")
message(STATUS "ONNXRUNTIME_INCLUDE: ${ONNXRUNTIME_INCLUDE}")
message(STATUS "ONNXRUNTIME_LIB_DIR: ${ONNXRUNTIME_LIB_DIR}")

# Find the main onnxruntime library
find_library(ONNXRUNTIME_LIB onnxruntime
    PATHS ${ONNXRUNTIME_LIB_DIR}
    NO_DEFAULT_PATH
)

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
    ${ONNXRUNTIME_INCLUDE}
)
target_link_libraries(core_engine PUBLIC ${ONNXRUNTIME_LIB})

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
    ${ONNXRUNTIME_INCLUDE}
)

target_link_libraries(alphatafl_engine PRIVATE
    core_engine
    ${ONNXRUNTIME_LIB}
)

# On Windows, copy required DLLs next to the pyd so they are found at runtime
if(WIN32)
    file(GLOB ONNXRUNTIME_DLLS "${ONNXRUNTIME_LIB_DIR}/*.dll")
    foreach(dll ${ONNXRUNTIME_DLLS})
        add_custom_command(TARGET alphatafl_engine POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${dll}" $<TARGET_FILE_DIR:alphatafl_engine>
        )
    endforeach()
endif()

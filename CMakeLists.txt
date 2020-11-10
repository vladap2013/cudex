cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(cudex LANGUAGES CXX CUDA)

enable_language(CUDA)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_STANDARD 17)
set(CUDA_STANDARD_REQUIRED ON)

# ----- glog

if (NOT TARGET glog::glog)
	message("Downloading google log")
	include(FetchContent)

	FetchContent_Declare(
	  glog
	  URL https://github.com/google/glog/archive/v0.4.0.tar.gz
	  URL_HASH SHA256=f28359aeba12f30d73d9e4711ef356dc842886968112162bc73002645139c39c
	)

	FetchContent_MakeAvailable(glog)
endif()


# ----- Main library

add_library(cudex_lib INTERFACE)
target_include_directories(cudex_lib INTERFACE include)
target_link_libraries(cudex_lib INTERFACE glog::glog)


# ----- Setup tests

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
	message("Top level project, setting project variables")

	set(CMAKE_CUDA_ARCHITECTURES 61 75)

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror all-warnings")
endif()

if (NOT TARGET gtest_main)
	message("Downloading google test")
	include(FetchContent)

	FetchContent_Declare(
	  googletest
	  URL https://github.com/google/googletest/archive/release-1.10.0.tar.gz
	  URL_HASH SHA256=9dc9157a9a1551ec7a7e43daea9a694a0bb5fb8bec81235d8a1e6ef64c716dcb
	)

	FetchContent_MakeAvailable(googletest)
endif()

add_executable(cudex_tests
	tests/test_memory.cpp.cu
	tests/test_launcher.cpp.cu
	tests/test_uarray.cpp.cu
)
target_link_libraries(cudex_tests cudex_lib gtest_main)

add_test(cudex_tests cudex_tests)

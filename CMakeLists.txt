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
	find_package(glog REQUIRED)
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
	  GIT_REPOSITORY https://github.com/google/googletest.git
	  GIT_TAG master
	)

	FetchContent_MakeAvailable(googletest)
endif()

add_executable(cudex_tests
	tests/test_memory.cpp.cu
	tests/test_launcher.cpp.cu
)
target_link_libraries(cudex_tests cudex_lib gtest_main)

add_test(cudex_tests cudex_tests)

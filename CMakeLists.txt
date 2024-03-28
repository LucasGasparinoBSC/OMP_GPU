cmake_minimum_required(VERSION 3.20)

project(testOMP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# If no build type is specified, default to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-acc -cuda -mp=gpu -Minfo=all -lnvToolsExt -fopenmp -gpu=cc90,lineinfo,pinned,,ptxinfo")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -G -Minform=inform")
set(CMAKE_CXX_FLAGS_RELEASE "-fast")

file(GLOB_RECURSE SOURCES "src/*.cpp")
set(HEADER_DIR "src")
file(GLOB_RECURSE HEADERS "src/*.h")

include_directories(${HEADER_DIR})
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(ICP)

find_package(PCL 1.10 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})
add_executable(ICP src/main.cpp)

target_link_libraries(ICP ${PCL_LIBRARIES})
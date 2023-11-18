cmake_minimum_required(VERSION 3.5.0)
project(ICP++ VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

cmake_policy(SET CMP0074 NEW)
find_package(PCL)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(ICP++ src/main.cpp)

target_link_libraries(ICP++ ${PCL_LIBRARIES})

message("PCL_LIBRARIES - ${PCL_LIBRARIES}")
message("INCLUDE DIRS - ${PCL_INCLUDE_DIRS}")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

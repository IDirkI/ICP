cmake_minimum_required(VERSION 3.0.0)
project(ICP++ VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_package(PCL)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(ICP++ src/main.cpp)

target_link_libraries(ICP++ ${PCL_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

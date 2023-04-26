cmake_minimum_required(VERSION 3.24)
project(multiAgentGovernment)

set(CMAKE_CXX_STANDARD 20)

link_libraries(tbb)

add_executable(multiAgentGovernment src/main.cpp src/Experiment1.cpp)
# add_executable(multiAgentGovernment src/main.cpp)

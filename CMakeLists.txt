cmake_minimum_required(VERSION 3.24)
project(multiAgentGovernment)

set(CMAKE_CXX_STANDARD 20)

include_directories(include/mlpack-4.2.0)
# link_libraries(tbb armadillo)
link_libraries(armadillo)

# add_executable(multiAgentGovernment src/main.cpp src/tests/randomEncounterSocietyTest.cpp src/tests/mlpacktests.cpp)
add_executable(multiAgentGovernment src/main.cpp)

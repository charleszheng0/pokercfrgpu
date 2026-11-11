cmake_minimum_required(VERSION 3.16)
project(pokercfr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
find_package(Threads REQUIRED)

add_library(pokercore
  src/cards.cpp
  src/range.cpp
  src/game.cpp
  src/thread_pool.cpp)
target_include_directories(pokercore PUBLIC src)
target_link_libraries(pokercore PUBLIC Threads::Threads)
target_compile_options(pokercore PRIVATE -Wall -Wextra)

find_package(SQLite3 REQUIRED)
# CMake before 4.1 names the target SQLite::SQLite3 (e.g. on the Oracle VM).
if(NOT TARGET SQLite3::SQLite3 AND TARGET SQLite::SQLite3)
  add_library(SQLite3::SQLite3 ALIAS SQLite::SQLite3)
endif()
find_package(ZLIB REQUIRED)
add_library(pokerstore src/store.cpp)
target_include_directories(pokerstore PUBLIC src)
target_link_libraries(pokerstore PUBLIC SQLite3::SQLite3 ZLIB::ZLIB)
target_compile_options(pokerstore PRIVATE -Wall -Wextra)

# Solver version stored with every solution: the git commit, if available.
execute_process(COMMAND git rev-parse --short HEAD WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE POKER_GIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT POKER_GIT)
  set(POKER_GIT "dev")
endif()

add_executable(pokersolve src/cli.cpp)
target_link_libraries(pokersolve pokercore)

add_executable(pokerbatch src/batch.cpp)
target_link_libraries(pokerbatch pokercore pokerstore)
target_compile_definitions(pokerbatch PRIVATE POKER_VERSION="${POKER_GIT}")

add_executable(pokerlab src/server.cpp)
target_link_libraries(pokerlab pokercore pokerstore)
target_compile_definitions(pokerlab PRIVATE WEB_DIR="${CMAKE_SOURCE_DIR}/web" DATA_DIR="${CMAKE_SOURCE_DIR}/data")

add_executable(tests tests/tests.cpp src/value_net.cpp)
target_link_libraries(tests pokerflat_cpu)

enable_testing()
add_test(NAME tests COMMAND tests)

add_executable(pokerscreen src/screen.cpp)
target_link_libraries(pokerscreen pokercore)

# Depth-limited flop solving with a leaf value network.
add_executable(pokervnet src/vnet.cpp src/value_net.cpp)
target_link_libraries(pokervnet pokerflat_cpu)
target_compile_options(pokervnet PRIVATE -Wall -Wextra)

# Level-by-level solver: CPU-thread back end (always) and CUDA back end (-DPOKER_CUDA=ON).
add_library(pokerflat_cpu gpu/flat_solver_cpu.cpp)
target_include_directories(pokerflat_cpu PUBLIC gpu)
target_link_libraries(pokerflat_cpu PUBLIC pokercore)
target_compile_options(pokerflat_cpu PRIVATE -Wall -Wextra)
add_executable(pokerflat gpu/bench.cpp)
target_link_libraries(pokerflat pokerflat_cpu)

option(POKER_CUDA "Build the CUDA solver (needs nvcc)" OFF)
if(POKER_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)  # must be set before enable_language
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  add_library(pokerflat_cuda gpu/flat_solver.cu)
  target_include_directories(pokerflat_cuda PUBLIC gpu)
  target_link_libraries(pokerflat_cuda PUBLIC pokercore)
  target_compile_options(pokerflat_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -O3>)
  set_target_properties(pokerflat_cuda PROPERTIES CUDA_STANDARD 17)
  add_executable(pokergpu gpu/bench.cpp)
  target_link_libraries(pokergpu pokerflat_cuda)
  add_executable(pokervnet_gpu src/vnet.cpp src/value_net.cpp)
  target_link_libraries(pokervnet_gpu pokerflat_cuda)
endif()

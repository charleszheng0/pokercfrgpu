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
add_library(pokerstore src/store.cpp)
target_include_directories(pokerstore PUBLIC src)
target_link_libraries(pokerstore PUBLIC SQLite3::SQLite3)
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

add_executable(tests tests/tests.cpp)
target_link_libraries(tests pokercore)

enable_testing()
add_test(NAME tests COMMAND tests)

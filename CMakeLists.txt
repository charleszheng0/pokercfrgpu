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

add_executable(pokersolve src/cli.cpp)
target_link_libraries(pokersolve pokercore)

add_executable(pokerlab src/server.cpp)
target_link_libraries(pokerlab pokercore)
target_compile_definitions(pokerlab PRIVATE WEB_DIR="${CMAKE_SOURCE_DIR}/web")

add_executable(tests tests/tests.cpp)
target_link_libraries(tests pokercore)

enable_testing()
add_test(NAME tests COMMAND tests)

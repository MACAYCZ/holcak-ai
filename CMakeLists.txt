cmake_minimum_required(VERSION 3.28)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

include_directories(source/)

project(holcak-ai LANGUAGES CUDA)
file(GLOB_RECURSE source source/*.cu)
add_library(holcak-ai STATIC ${source})

project(example LANGUAGES CUDA)
add_executable(example main.cu)
target_link_libraries(example PUBLIC holcak-ai)

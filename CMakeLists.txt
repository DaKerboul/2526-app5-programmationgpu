cmake_minimum_required(VERSION 3.18)
project(cuda_course CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

include(FetchContent)

set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)

if(CMAKE_CUDA_COMPILER)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "" FORCE)
    set(Kokkos_ARCH_AMPERE86 ON CACHE BOOL "" FORCE)
endif()

FetchContent_Declare(
    kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG 5.0.2
)
FetchContent_MakeAvailable(kokkos)

file(GLOB_RECURSE CUDA_SOURCES
    "exemples/*.cu"
    "tp/**/*.cu"
)

foreach(SRC ${CUDA_SOURCES})
    get_filename_component(TARGET_NAME ${SRC} NAME_WE)
    get_filename_component(TARGET_DIR ${SRC} DIRECTORY)
    string(REPLACE "${CMAKE_SOURCE_DIR}/" "" REL_DIR ${TARGET_DIR})
    string(REPLACE "/" "_" TARGET_PREFIX ${REL_DIR})
    add_executable(${TARGET_PREFIX}_${TARGET_NAME} ${SRC})
    set_target_properties(${TARGET_PREFIX}_${TARGET_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endforeach()

file(GLOB_RECURSE KOKKOS_SOURCES
    "tp/tp1_kokkos/*.cpp"
)

foreach(SRC ${KOKKOS_SOURCES})
    get_filename_component(TARGET_NAME ${SRC} NAME_WE)
    get_filename_component(TARGET_DIR ${SRC} DIRECTORY)
    string(REPLACE "${CMAKE_SOURCE_DIR}/" "" REL_DIR ${TARGET_DIR})
    string(REPLACE "/" "_" TARGET_PREFIX ${REL_DIR})
    add_executable(${TARGET_PREFIX}_${TARGET_NAME} ${SRC})
    target_link_libraries(${TARGET_PREFIX}_${TARGET_NAME} PRIVATE Kokkos::kokkos)
endforeach()

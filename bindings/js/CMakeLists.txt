cmake_minimum_required(VERSION 3.10)
project(boat-napi LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Boat source tree root
set(BOAT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# Boat build output (pre-built via main CMake)
set(BOAT_BUILD_DIR "${BOAT_ROOT}/build")

# Node.js N-API include/lib provided by cmake-js
list(APPEND CMAKE_JS_INC "${BOAT_ROOT}/include")

add_library(${PROJECT_NAME} SHARED
    src/boat_bridge.c
    src/addon.cpp
    src/tensor_wrapper.cpp
    src/model_wrapper.cpp
    ${CMAKE_JS_SRC}
)

# Output as .node (Node.js native addon extension)
set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".node"
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${BOAT_ROOT}/include
    ${CMAKE_JS_INC}
)

target_link_directories(${PROJECT_NAME} PRIVATE
    ${BOAT_BUILD_DIR}/Release
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    boat
    ${CMAKE_JS_LIB}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    BOAT_STATIC_BUILD
)

# On Windows, generate node.lib from the .def file provided by cmake-js
if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
    add_custom_command(
        OUTPUT ${CMAKE_JS_NODELIB_TARGET}
        COMMAND lib.exe /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} /machine:x64
        DEPENDS ${CMAKE_JS_NODELIB_DEF}
        COMMENT "Generating node.lib from .def file"
    )
    add_custom_target(generate_node_lib DEPENDS ${CMAKE_JS_NODELIB_TARGET})
    add_dependencies(${PROJECT_NAME} generate_node_lib)
endif()

# MSVC-specific: silence C4819 (codepage warning), match boat release flags
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /wd4819)
endif()

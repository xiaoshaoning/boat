cmake_minimum_required(VERSION 3.10)
project(minimind_example C)

if(TARGET boat)
    set(MINIMIND_SOURCES
        engine.c
        minimind_cli.c
        model.c
        sampling.c
        tokenizer.c
        weights.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/json.c
    )

    add_executable(minimind ${MINIMIND_SOURCES})
    target_include_directories(minimind PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
    )
    target_link_libraries(minimind PRIVATE boat)
    target_compile_definitions(minimind PRIVATE BOAT_STATIC_BUILD
        _CRT_NONSTDC_NO_DEPRECATE _CRT_SECURE_NO_WARNINGS)

    if(UNIX)
        target_link_libraries(minimind PRIVATE m)
    endif()

    set_target_properties(minimind PROPERTIES C_STANDARD 11)
    return()
endif()

# Fallback standalone build (no boat library dependency)
add_executable(minimind
    engine.c
    minimind_cli.c
    model.c
    sampling.c
    tokenizer.c
    weights.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/json.c
)
target_include_directories(minimind PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)
target_compile_definitions(minimind PRIVATE
    _CRT_NONSTDC_NO_DEPRECATE _CRT_SECURE_NO_WARNINGS)

if(MSVC)
    target_compile_options(minimind PRIVATE /O2 /fp:fast)
else()
    target_compile_options(minimind PRIVATE -O3 -ffast-math)
    target_link_libraries(minimind PRIVATE m)
endif()

set_target_properties(minimind PROPERTIES C_STANDARD 11)

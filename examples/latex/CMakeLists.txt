# Nougat-LaTeX OCR inference example
# Requires boat library built with -DBOAT_WITH_CUDA=ON for GPU acceleration

cmake_minimum_required(VERSION 3.10)
project(latex_example)

if(TARGET boat AND BOAT_WITH_CUDA)
    # Common sources
    set(LATEX_COMMON_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/safetensors.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/json.c
    )
    set(LATEX_COMMON_INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
    )

    add_executable(latex_ocr
        latex.c
        image.c
        nougat_model.c
        nougat_decoder.c
        ${LATEX_COMMON_SOURCES}
    )
    target_include_directories(latex_ocr PRIVATE ${LATEX_COMMON_INCLUDES})
    target_link_libraries(latex_ocr PRIVATE boat)
    target_compile_definitions(latex_ocr PRIVATE
        BOAT_STATIC_BUILD BOAT_WITH_CUDA
        _CRT_NONSTDC_NO_DEPRECATE _CRT_SECURE_NO_WARNINGS)
    if(UNIX)
        target_link_libraries(latex_ocr PRIVATE m)
    endif()
    set_target_properties(latex_ocr PROPERTIES C_STANDARD 11)

    return()
endif()

message(WARNING "Nougat-LaTeX example requires boat library built with BOAT_WITH_CUDA=ON. Skipping.")

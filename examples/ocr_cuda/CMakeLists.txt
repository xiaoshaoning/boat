# CUDA-accelerated GLM-OCR example
# Requires boat library built with -DBOAT_WITH_CUDA=ON -DBOAT_WITH_CUDNN=ON

cmake_minimum_required(VERSION 3.10)
project(ocr_cuda_example)

if(TARGET boat AND BOAT_WITH_CUDA)
    # Shared sources from ../ocr/ and ../common/
    set(OCR_COMMON_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../ocr/tokenizer.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../ocr/image.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../ocr/sampling.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/json.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/safetensors.c
    )
    set(OCR_COMMON_INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}/../ocr
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
    )

    # ---- CUDA OCR inference executable ----
    add_executable(ocr_cuda
        ocr_cuda.c
        ocr_cuda_kernels.cu
        cogvit_cuda.cu
        glm_cuda.cu
        ${OCR_COMMON_SOURCES}
    )
    target_include_directories(ocr_cuda PRIVATE ${OCR_COMMON_INCLUDES})
    target_link_libraries(ocr_cuda PRIVATE boat)
    target_compile_definitions(ocr_cuda PRIVATE
        BOAT_STATIC_BUILD BOAT_WITH_CUDA BOAT_WITH_CUDNN
        _CRT_NONSTDC_NO_DEPRECATE _CRT_SECURE_NO_WARNINGS)
    if(WIN32)
        target_link_libraries(ocr_cuda PRIVATE windowscodecs.lib ole32.lib)
    endif()
    if(UNIX)
        target_link_libraries(ocr_cuda m)
    endif()
    set_target_properties(ocr_cuda PROPERTIES C_STANDARD 11)
    set_target_properties(ocr_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # ---- CUDA OCR test executable ----
    add_executable(test_ocr_cuda
        test_ocr_cuda.cu
        ocr_cuda_kernels.cu
        cogvit_cuda.cu
        glm_cuda.cu
        ${OCR_COMMON_SOURCES}
    )
    target_include_directories(test_ocr_cuda PRIVATE ${OCR_COMMON_INCLUDES})
    target_link_libraries(test_ocr_cuda PRIVATE boat)
    target_compile_definitions(test_ocr_cuda PRIVATE
        BOAT_STATIC_BUILD BOAT_WITH_CUDA BOAT_WITH_CUDNN
        _CRT_NONSTDC_NO_DEPRECATE _CRT_SECURE_NO_WARNINGS)
    if(WIN32)
        target_link_libraries(test_ocr_cuda PRIVATE windowscodecs.lib ole32.lib)
    endif()
    if(UNIX)
        target_link_libraries(test_ocr_cuda m)
    endif()
    set_target_properties(test_ocr_cuda PROPERTIES C_STANDARD 11)
    set_target_properties(test_ocr_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    return()
endif()

# Standalone build warning
message(WARNING "CUDA OCR example requires boat library built with BOAT_WITH_CUDA=ON. Skipping.")

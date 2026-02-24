# CMakeLists.txt for Boat framework tests

# Simple test executable
add_executable(test_basic test_basic.c)
target_link_libraries(test_basic boat)

# Disable encoding warning on Windows
if(MSVC)
    target_compile_options(test_basic PRIVATE /wd4819)
endif()

# Copy DLL to test binary directory on Windows
if(WIN32)
    add_custom_command(TARGET test_basic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:boat>
            $<TARGET_FILE_DIR:test_basic>
        COMMENT "Copying boat DLL to test directory"
    )
endif()

# Enable testing if CTest is available
if(BUILD_TESTING)
    add_test(NAME test_basic COMMAND test_basic)
endif()
cmake_minimum_required(VERSION 3.24)
project(screenshot_test CXX)
# Screenshot files: plugin/src/imagefile.cpp and screenshot.h's gamma conversion, written and decoded through WIC.
# x86 like the plugin (the same code and system codecs the game process uses).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
set(PLUGIN_SRC ${CMAKE_CURRENT_LIST_DIR}/../../plugin/src)
add_executable(screenshot_test screenshot_test.cpp ${PLUGIN_SRC}/imagefile.cpp)
target_include_directories(screenshot_test PRIVATE ${PLUGIN_SRC})
target_compile_definitions(screenshot_test PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE)
target_compile_options(screenshot_test PRIVATE /EHsc /utf-8 /W4 /permissive-)
target_link_libraries(screenshot_test PRIVATE windowscodecs ole32 shell32)

cmake_minimum_required(VERSION 3.24)
project(dxtparity CXX)
# Native DXT codec vs the game's d3dx9_38.dll (x86 only: the reference is the 32-bit DLL).
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 4)
  message(FATAL_ERROR "dxtparity must be built for x86")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BACKEND ${CMAKE_CURRENT_LIST_DIR}/../../backend)
add_executable(dxtparity dxtparity.cpp ${BACKEND}/d3dxtex/dxt.cpp)
target_include_directories(dxtparity PRIVATE ${BACKEND})
target_compile_options(dxtparity PRIVATE /EHsc /W3 /arch:SSE2)
# The reproduced x87 arithmetic depends on exact IEEE operations and MXCSR rounding spans.
set_source_files_properties(${BACKEND}/d3dxtex/dxt.cpp PROPERTIES COMPILE_OPTIONS "/fp:strict")

cmake_minimum_required(VERSION 3.24)
project(sdp2spv CXX)

# Offline host tool (x64). Translates the game's D3D9 shader packages to SPIR-V.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/dxbc_spirv.cmake)

add_executable(sdp2spv sdp2spv.cpp)
target_link_libraries(sdp2spv PRIVATE dxbc_spv)
if(MSVC)
  target_compile_options(sdp2spv PRIVATE /EHsc /utf-8 /permissive- /W3 /UNDEBUG)
endif()

cmake_minimum_required(VERSION 3.24)
project(shaderdiff CXX)

# Differential test: run each original D3D9 shader on real D3D9 and its SPIR-V translation on Vulkan with identical
# inputs, then compare the results.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_program(GLSLANG glslangValidator HINTS $ENV{VULKAN_SDK}/Bin REQUIRED)

set(VS_SPV_HDR ${CMAKE_CURRENT_BINARY_DIR}/passthrough_vert.h)
add_custom_command(
  OUTPUT ${VS_SPV_HDR}
  COMMAND ${GLSLANG} -V --target-env vulkan1.3 --vn kPassthroughVertSpv -o ${VS_SPV_HDR}
          ${CMAKE_CURRENT_LIST_DIR}/passthrough.vert
  DEPENDS ${CMAKE_CURRENT_LIST_DIR}/passthrough.vert)

add_executable(shaderdiff shaderdiff.cpp ${VS_SPV_HDR})
target_include_directories(shaderdiff PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(shaderdiff PRIVATE Vulkan::Vulkan d3d9 d3dcompiler)
target_compile_definitions(shaderdiff PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
if(MSVC)
  target_compile_options(shaderdiff PRIVATE /EHsc /utf-8 /W3 /permissive-)
endif()

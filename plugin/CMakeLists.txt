cmake_minimum_required(VERSION 3.24)
project(fnv_vk C CXX)

# xNVSE plugin, 32-bit (the game is a 32-bit process). Build with an x86 MSVC environment.
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 4)
  message(FATAL_ERROR "fnv_vk must be built for x86 (32-bit)")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")  # no VC++ redist dependency

set(MINHOOK_DIR ${CMAKE_CURRENT_LIST_DIR}/../third_party/minhook)
add_library(minhook STATIC
  ${MINHOOK_DIR}/src/buffer.c ${MINHOOK_DIR}/src/hook.c ${MINHOOK_DIR}/src/trampoline.c ${MINHOOK_DIR}/src/hde/hde32.c)
target_include_directories(minhook PUBLIC ${MINHOOK_DIR}/include)

# Vulkan backend (headers only; vulkan-1.dll is loaded at runtime -- the SDK has no 32-bit import library).
include(${CMAKE_CURRENT_LIST_DIR}/../backend/CMakeLists.txt)

add_library(fnv_vk SHARED
  src/cputime.cpp
  src/crashlog.cpp
  src/display.cpp
  src/effects.cpp
  src/envprobe.cpp
  src/gbbuffers.cpp
  src/gbdevice.cpp
  src/gbobjects.cpp
  src/gbquery.cpp
  src/gbshaders.cpp
  src/gbstate.cpp
  src/gbsurfaces.cpp
  src/imagefile.cpp
  src/main.cpp
  src/movie.cpp
  src/observe.cpp
  src/overlay.cpp
  src/present.cpp
  src/replacements.cpp
  src/screenshot.cpp
  src/shadow.cpp
  src/sunshadows.cpp
  src/textures.cpp
  src/upscale.cpp
  src/viewvector.cpp
  src/wetworld.cpp)
target_link_libraries(fnv_vk PRIVATE minhook fnvvk_backend dbghelp dxguid dinput8 comctl32 windowscodecs ole32 shell32)  # dinput8.lib: c_dfDIMouse2 (data only)
target_compile_definitions(fnv_vk PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(fnv_vk PRIVATE /EHsc /utf-8 /W4 /permissive-)
# FSR 3.1 (upscaling): the FidelityFX API built as fnv_vk_fidelityfx.dll (backend/CMakeLists.txt,
# cmake/fidelityfx.cmake), loaded at run time.
add_dependencies(fnv_vk fnv_vk_fidelityfx)

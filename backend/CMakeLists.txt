# fnv_vk Vulkan backend: game-independent, built for x86 (plugin) and x64 (offline harness).
find_path(FNVVK_VULKAN_INCLUDE vulkan/vulkan.h HINTS $ENV{VULKAN_SDK}/Include REQUIRED)
set(FNVVK_BACKEND_DIR ${CMAKE_CURRENT_LIST_DIR})
set(FNVVK_VMA_INCLUDE ${CMAKE_CURRENT_LIST_DIR}/../third_party/VulkanMemoryAllocator/include)
add_library(fnvvk_backend STATIC
  ${FNVVK_BACKEND_DIR}/backend.cpp
  ${FNVVK_BACKEND_DIR}/vkfn.cpp
  ${FNVVK_BACKEND_DIR}/shaderpack.cpp
  ${FNVVK_BACKEND_DIR}/vma_impl.cpp
  ${FNVVK_BACKEND_DIR}/d3dxtex/dxt.cpp
  ${FNVVK_BACKEND_DIR}/d3dxtex/texture.cpp)
target_include_directories(fnvvk_backend PUBLIC ${FNVVK_BACKEND_DIR} ${FNVVK_VULKAN_INCLUDE} ${FNVVK_VMA_INCLUDE})
target_compile_features(fnvvk_backend PUBLIC cxx_std_20)
target_compile_definitions(fnvvk_backend PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
# Dear ImGui (third_party/imgui, upstream v1.92.6) with its Vulkan renderer: the in-game menu, drawn by the backend at
# present. Vulkan functions come from the backend's loader (no prototypes, no import library).
set(FNVVK_IMGUI_DIR ${CMAKE_CURRENT_LIST_DIR}/../third_party/imgui)
add_library(fnvvk_imgui STATIC
  ${FNVVK_IMGUI_DIR}/imgui.cpp ${FNVVK_IMGUI_DIR}/imgui_draw.cpp ${FNVVK_IMGUI_DIR}/imgui_tables.cpp
  ${FNVVK_IMGUI_DIR}/imgui_widgets.cpp ${FNVVK_IMGUI_DIR}/backends/imgui_impl_vulkan.cpp)
target_include_directories(fnvvk_imgui PUBLIC ${FNVVK_IMGUI_DIR} ${FNVVK_IMGUI_DIR}/backends ${FNVVK_VULKAN_INCLUDE})
target_compile_definitions(fnvvk_imgui PUBLIC IMGUI_IMPL_VULKAN_NO_PROTOTYPES VK_NO_PROTOTYPES IMGUI_DISABLE_OBSOLETE_FUNCTIONS)
target_compile_features(fnvvk_imgui PUBLIC cxx_std_20)
if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 4)
  target_compile_options(fnvvk_imgui PRIVATE /arch:SSE2)
endif()
# glslang (third_party/glslang, 16.4.0): effect shaders are GLSL compiled at load (backend/effectlib.cpp). GLSL only:
# no HLSL front end, no optimizer (no SPIRV-Tools dependency), no tools or tests.
set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/glslang ${CMAKE_BINARY_DIR}/glslang EXCLUDE_FROM_ALL)
target_sources(fnvvk_backend PRIVATE ${FNVVK_BACKEND_DIR}/effectlib.cpp)
target_link_libraries(fnvvk_backend PUBLIC bcrypt fnvvk_imgui glslang glslang-default-resource-limits)
if(MSVC)
  target_compile_options(fnvvk_backend PRIVATE /EHsc /utf-8 /permissive-)
  set_source_files_properties(${FNVVK_BACKEND_DIR}/backend.cpp ${FNVVK_BACKEND_DIR}/vkfn.cpp ${FNVVK_BACKEND_DIR}/shaderpack.cpp PROPERTIES COMPILE_OPTIONS "/W4")
  # The D3DX reimplementation reproduces exact IEEE operation sequences and MXCSR rounding spans.
  set_source_files_properties(${FNVVK_BACKEND_DIR}/d3dxtex/dxt.cpp ${FNVVK_BACKEND_DIR}/d3dxtex/texture.cpp
                              PROPERTIES COMPILE_OPTIONS "/W4;/fp:strict")
  if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    target_compile_options(fnvvk_backend PRIVATE /arch:SSE2)
  endif()
endif()
# FSR 3.1 (backend_ffx.inl): the FidelityFX API's headers from the patched SDK copy (cmake/fidelityfx.cmake; their
# handle types depend on the patches). The API itself is fnv_vk_fidelityfx.dll, built there for x86 and loaded at run time.
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/fidelityfx.cmake)
target_include_directories(fnvvk_backend PRIVATE ${FFX_SDK_DIR}/ffx-api/include)

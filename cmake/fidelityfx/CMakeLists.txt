# fnv_vk_fidelityfx.dll: the FidelityFX API (ffx-api, Vulkan) with FSR 3.1 upscaling, frame generation, optical flow and
# the frame generation swapchain, built for the game's 32-bit process. Copied by cmake/fidelityfx.cmake over the SDK's
# own sdk/src/backends/vk/CMakeLists.txt in the patched copy: this directory is the SDK's Vulkan backend directory (the
# SDK's shader scripts glob their sources relative to it).

get_filename_component(FFX_SDK ${CMAKE_CURRENT_SOURCE_DIR}/../../.. ABSOLUTE)  # <copy>/sdk (normalised: the shader
                                                                              # compiler's include paths stay short)
get_filename_component(FFX_API ${FFX_SDK}/../ffx-api ABSOLUTE)
set(FFX_GPU_PATH ${FFX_SDK}/include/FidelityFX/gpu)
set(FFX_PASS_SHADER_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${FFX_PASS_SHADER_OUTPUT_PATH})
set(FNVVK_FFX_SUPPORT ${FNVVK_ROOT}/cmake/fidelityfx)

# --- shaders: the SDK's permutation scripts (FidelityFX_SC.exe, glslang -> SPIR-V) -------------------------------------
if(CMAKE_GENERATOR STREQUAL "Ninja")
  set(USE_DEPFILE TRUE)
endif()
include(${FFX_GPU_PATH}/CMakeCompileShaders.txt)
set(FFX_SC_PERMUTATION_OUTPUTS "")
include(${CMAKE_CURRENT_SOURCE_DIR}/CMakeShadersFSR3Upscaler.txt)
include(${CMAKE_CURRENT_SOURCE_DIR}/CMakeShadersFrameinterpolation.txt)
include(${CMAKE_CURRENT_SOURCE_DIR}/CMakeShadersOpticalflow.txt)
# (compile_shaders_with_depfile returns its output variable's own name as the list's first element: only headers stay)
list(FILTER FFX_SC_PERMUTATION_OUTPUTS INCLUDE REGEX "_permutations\.h$")
add_custom_target(fnv_vk_fidelityfx_shaders DEPENDS ${FFX_SC_PERMUTATION_OUTPUTS})

# --- the DLL -----------------------------------------------------------------------------------------------------------
file(GLOB FFX_SWAPCHAIN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/FrameInterpolationSwapchain/*.cpp)
set(FFX_DLL_SOURCES
  # SDK: shared host code, the Vulkan backend (with the frame generation swapchain), the three components
  ${FFX_SDK}/src/shared/ffx_assert.cpp
  ${FFX_SDK}/src/shared/ffx_message.cpp
  ${FFX_SDK}/src/shared/ffx_object_management.cpp
  ${FFX_SDK}/src/shared/ffx_breadcrumbs_list.cpp
  ${FFX_SDK}/src/backends/shared/ffx_shader_blobs.cpp
  ${FFX_SDK}/src/backends/shared/blob_accessors/ffx_fsr3upscaler_shaderblobs.cpp
  ${FFX_SDK}/src/backends/shared/blob_accessors/ffx_frameinterpolation_shaderblobs.cpp
  ${FFX_SDK}/src/backends/shared/blob_accessors/ffx_opticalflow_shaderblobs.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ffx_vk.cpp
  ${FFX_SWAPCHAIN_SOURCES}
  ${FFX_SDK}/src/components/fsr3upscaler/ffx_fsr3upscaler.cpp
  ${FFX_SDK}/src/components/frameinterpolation/ffx_frameinterpolation.cpp
  ${FFX_SDK}/src/components/opticalflow/ffx_opticalflow.cpp
  # FidelityFX API: the upscale and frame generation providers and the Vulkan frame generation swapchain provider
  # (FSR 2's provider is left out, patches/fidelityfx)
  ${FFX_API}/src/ffx_api.cpp
  ${FFX_API}/src/ffx_provider.cpp
  ${FFX_API}/src/ffx_provider_fsr3upscale.cpp
  ${FFX_API}/src/ffx_provider_framegeneration.cpp
  ${FFX_API}/src/backends.cpp
  ${FFX_API}/src/validation.cpp
  ${FFX_API}/src/vk/ffx_provider_framegenerationswapchain_vk.cpp
  ${FFX_API}/src/resource/ffx_api_dll.rc
  # Vulkan entry points from the loaded vulkan-1.dll (there is no 32-bit import library)
  ${FNVVK_FFX_SUPPORT}/vk_imports.cpp)

add_library(fnv_vk_fidelityfx SHARED ${FFX_DLL_SOURCES})
add_dependencies(fnv_vk_fidelityfx fnv_vk_fidelityfx_shaders)
target_include_directories(fnv_vk_fidelityfx PRIVATE
  ${FFX_SDK}/include ${FFX_SDK}/src/shared ${FFX_SDK}/src/backends/shared ${FFX_SDK}/src/components
  ${FFX_PASS_SHADER_OUTPUT_PATH} ${FFX_API}/include ${FFX_API}/src ${FNVVK_FFX_SUPPORT} ${FNVVK_VULKAN_INCLUDE})
target_compile_definitions(fnv_vk_fidelityfx PRIVATE
  FFX_FSR3UPSCALER FFX_FI FFX_OF FFX_BACKEND_VK FFX_NO_FSR2_PROVIDER
  FFXDLL_VERSION_MAJOR=1 FFXDLL_VERSION_MINOR=0 FFXDLL_VERSION_PATCH=1 FFXDLL_VERSION_API=VK CI_PIPELINE_ID=0
  _UNICODE UNICODE _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR NOMINMAX)
# C++17 with MSVC's default conformance mode, as the SDK is written for (the plugin's C++20 implies /permissive-).
set_target_properties(fnv_vk_fidelityfx PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
if(MSVC)
  # The SDK's recommended switches (sdk/toolchain.cmake); every Vulkan entry point comes from vk_imports.h.
  target_compile_options(fnv_vk_fidelityfx PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/permissive /W3 /GR- /fp:fast /GS /Gy /EHsc /utf-8 /bigobj /arch:SSE2 /FI${FNVVK_FFX_SUPPORT}/vk_imports.h>)
endif()
target_link_libraries(fnv_vk_fidelityfx PRIVATE dxgi)
set_target_properties(fnv_vk_fidelityfx PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

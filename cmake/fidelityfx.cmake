# Builds AMD FidelityFX FSR 3.1 (upscaler, frame generation, optical flow and the frame generation swapchain) for
# Vulkan as fnv_vk_fidelityfx.dll: the FidelityFX API (ffx-api) with its Vulkan backend, from the pinned SDK submodule
# (third_party/FidelityFX-SDK, v1.1.4, the last release with Vulkan source) plus the local patches in
# patches/fidelityfx/*.patch. AMD ships the API only as signed 64-bit DLLs; the game is a 32-bit process, so it is
# built here for x86. The submodule stays an untouched upstream checkout: the parts used are copied into the build tree
# and patched there (as cmake/dxbc_spirv.cmake does), and re-copied whenever a patch changes.
#
# The shaders are compiled by the SDK's own driver (sdk/tools/binary_store/FidelityFX_SC.exe, glslang to SPIR-V) with
# the SDK's own permutation scripts (CMakeCompile*Shaders.txt); the generated permutation headers are compiled into the
# DLL. The plugin loads the DLL at run time (backend/backend_ffx.inl) only when upscaling is enabled; fnv_vk uses
# only the upscaler, but the API's provider list links the other components.

set(FNVVK_ROOT ${CMAKE_CURRENT_LIST_DIR}/..)
set(FFX_SDK_SRC_DIR ${FNVVK_ROOT}/third_party/FidelityFX-SDK)
set(FFX_SDK_DIR ${CMAKE_BINARY_DIR}/fidelityfx-patched)
# Writing a patch: point FNVVK_FFX_WORKTREE at a git checkout of the copied parts (sdk/, ffx-api/) with the patches
# applied; it is built as it is (only the DLL's CMakeLists.txt is copied in), and `git diff` there is the next patch.
set(FNVVK_FFX_WORKTREE "" CACHE PATH "FidelityFX SDK work tree used instead of the patched copy (patch authoring)")

file(GLOB FFX_PATCHES CONFIGURE_DEPENDS ${FNVVK_ROOT}/patches/fidelityfx/*.patch)
list(SORT FFX_PATCHES)
set(FFX_BUILD_LISTS ${CMAKE_CURRENT_LIST_DIR}/fidelityfx/CMakeLists.txt)

set(_stamp_content "")
foreach(p ${FFX_PATCHES} ${FFX_BUILD_LISTS})
  file(SHA256 ${p} _h)
  string(APPEND _stamp_content "${p}:${_h}\n")
endforeach()
execute_process(COMMAND git -C ${FFX_SDK_SRC_DIR} rev-parse HEAD OUTPUT_VARIABLE _rev OUTPUT_STRIP_TRAILING_WHITESPACE)
string(APPEND _stamp_content "rev:${_rev}\n")

set(_stamp ${FFX_SDK_DIR}/.fnvvk-patch-stamp)
set(_old "")
if(EXISTS ${_stamp})
  file(READ ${_stamp} _old)
endif()
if(FNVVK_FFX_WORKTREE)
  set(FFX_SDK_DIR ${FNVVK_FFX_WORKTREE})
  file(COPY ${FFX_BUILD_LISTS} DESTINATION ${FFX_SDK_DIR}/sdk/src/backends/vk)
elseif(NOT _old STREQUAL _stamp_content)
  message(STATUS "FidelityFX SDK: copying ${_rev} and applying ${FFX_PATCHES}")
  file(REMOVE_RECURSE ${FFX_SDK_DIR})
  # Only what FSR 3.1 uses: the GPU and host headers, the shared and Vulkan backend sources, the three components and
  # the FidelityFX API layer.
  file(COPY ${FFX_SDK_SRC_DIR}/sdk/include DESTINATION ${FFX_SDK_DIR}/sdk)
  file(COPY ${FFX_SDK_SRC_DIR}/sdk/src/shared DESTINATION ${FFX_SDK_DIR}/sdk/src)
  file(COPY ${FFX_SDK_SRC_DIR}/sdk/src/backends/shared ${FFX_SDK_SRC_DIR}/sdk/src/backends/vk
       DESTINATION ${FFX_SDK_DIR}/sdk/src/backends)
  foreach(c fsr3upscaler frameinterpolation opticalflow)
    file(COPY ${FFX_SDK_SRC_DIR}/sdk/src/components/${c} DESTINATION ${FFX_SDK_DIR}/sdk/src/components)
  endforeach()
  file(COPY ${FFX_SDK_SRC_DIR}/ffx-api DESTINATION ${FFX_SDK_DIR})
  foreach(p ${FFX_PATCHES})
    # The copy lives inside the fnv_vk work tree: GIT_CEILING_DIRECTORIES keeps `git apply` from resolving paths against
    # that repository (see cmake/dxbc_spirv.cmake). The reverse check proves the patch landed.
    execute_process(COMMAND ${CMAKE_COMMAND} -E env GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}
                            git apply --whitespace=nowarn ${p}
                    WORKING_DIRECTORY ${FFX_SDK_DIR} RESULT_VARIABLE _r ERROR_VARIABLE _e)
    execute_process(COMMAND ${CMAKE_COMMAND} -E env GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}
                            git apply --reverse --check ${p}
                    WORKING_DIRECTORY ${FFX_SDK_DIR} RESULT_VARIABLE _v ERROR_VARIABLE _ve)
    if(NOT _r EQUAL 0 OR NOT _v EQUAL 0)
      message(FATAL_ERROR "FidelityFX SDK: patch ${p} did not apply:\n${_e}${_ve}")
    endif()
  endforeach()
  # The build of the DLL replaces the Vulkan backend's own CMakeLists.txt (the SDK's targets Visual Studio x64 only);
  # it lives in the backend directory because the SDK's shader scripts glob their sources relative to it.
  file(COPY ${FFX_BUILD_LISTS} DESTINATION ${FFX_SDK_DIR}/sdk/src/backends/vk)
  file(WRITE ${_stamp} "${_stamp_content}")
endif()

# The DLL is for the game's 32-bit process (and the 32-bit harness); 64-bit builds only use the headers.
if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(FFX_SC_EXECUTABLE ${FFX_SDK_SRC_DIR}/sdk/tools/binary_store/FidelityFX_SC.exe)
  add_subdirectory(${FFX_SDK_DIR}/sdk/src/backends/vk ${CMAKE_BINARY_DIR}/fidelityfx)
endif()

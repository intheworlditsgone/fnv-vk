# Builds dxbc-spirv (sm3 + spirv + ir + util) as a static library from the pinned submodule plus the local
# patches in patches/dxbc-spirv/*.patch. The submodule stays an untouched upstream checkout: the sources are
# copied into the build tree and patched there, and re-copied whenever a patch changes.

set(FNVVK_ROOT ${CMAKE_CURRENT_LIST_DIR}/..)
set(DXBC_SPV_SRC_DIR ${FNVVK_ROOT}/third_party/dxbc-spirv)
set(DXBC_SPV_DIR ${CMAKE_BINARY_DIR}/dxbc-spirv-patched)

file(GLOB DXBC_SPV_PATCHES CONFIGURE_DEPENDS ${FNVVK_ROOT}/patches/dxbc-spirv/*.patch)
list(SORT DXBC_SPV_PATCHES)

set(_stamp_content "")
foreach(p ${DXBC_SPV_PATCHES})
  file(SHA256 ${p} _h)
  string(APPEND _stamp_content "${p}:${_h}\n")
endforeach()
execute_process(COMMAND git -C ${DXBC_SPV_SRC_DIR} rev-parse HEAD OUTPUT_VARIABLE _rev OUTPUT_STRIP_TRAILING_WHITESPACE)
string(APPEND _stamp_content "rev:${_rev}\n")

set(_stamp ${DXBC_SPV_DIR}/.fnvvk-patch-stamp)
set(_old "")
if(EXISTS ${_stamp})
  file(READ ${_stamp} _old)
endif()
if(NOT _old STREQUAL _stamp_content)
  message(STATUS "dxbc-spirv: copying ${_rev} and applying ${DXBC_SPV_PATCHES}")
  file(REMOVE_RECURSE ${DXBC_SPV_DIR})
  foreach(d sm3 spirv ir util submodules)
    file(COPY ${DXBC_SPV_SRC_DIR}/${d} DESTINATION ${DXBC_SPV_DIR})
  endforeach()
  foreach(p ${DXBC_SPV_PATCHES})
    # The copy lives inside the fnv_vk work tree; inside a repository `git apply` resolves paths against the
    # repository root and silently skips the rest. GIT_CEILING_DIRECTORIES stops it from finding that repo,
    # so it applies relative to the copy like plain `patch`. The reverse check proves the patch landed.
    execute_process(COMMAND ${CMAKE_COMMAND} -E env GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}
                            git apply --whitespace=nowarn ${p}
                    WORKING_DIRECTORY ${DXBC_SPV_DIR} RESULT_VARIABLE _r ERROR_VARIABLE _e)
    execute_process(COMMAND ${CMAKE_COMMAND} -E env GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}
                            git apply --reverse --check ${p}
                    WORKING_DIRECTORY ${DXBC_SPV_DIR} RESULT_VARIABLE _v ERROR_VARIABLE _ve)
    if(NOT _r EQUAL 0 OR NOT _v EQUAL 0)
      message(FATAL_ERROR "dxbc-spirv: patch ${p} did not apply:\n${_e}${_ve}")
    endif()
  endforeach()
  file(WRITE ${_stamp} "${_stamp_content}")
endif()

file(GLOB DXBC_SPV_SOURCES
  ${DXBC_SPV_DIR}/sm3/*.cpp
  ${DXBC_SPV_DIR}/spirv/*.cpp
  ${DXBC_SPV_DIR}/ir/*.cpp
  ${DXBC_SPV_DIR}/ir/passes/*.cpp
  ${DXBC_SPV_DIR}/util/*.cpp)

add_library(dxbc_spv STATIC ${DXBC_SPV_SOURCES})
target_include_directories(dxbc_spv PUBLIC ${DXBC_SPV_DIR})
target_compile_definitions(dxbc_spv PUBLIC DXBC_SPV_ENABLE_SM3 DXBC_SPV_ENABLE_SPIRV)
if(MSVC)
  # Keep the library's internal asserts active in every config.
  target_compile_options(dxbc_spv PRIVATE /EHsc /utf-8 /bigobj /permissive- /W3 /UNDEBUG)
endif()

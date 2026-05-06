cmake_minimum_required(VERSION 3.14)
project(tts-cpp LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)
include(CheckIncludeFileCXX)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(TTS_CPP_STANDALONE_DEFAULT ON)
else()
    set(TTS_CPP_STANDALONE_DEFAULT OFF)
endif()

option(TTS_CPP_BUILD_LIBRARY     "tts-cpp: build the static tts-cpp library"                 ON)
option(TTS_CPP_BUILD_EXECUTABLES "tts-cpp: build the tts-cli executable + mel2wav demo"     ${TTS_CPP_STANDALONE_DEFAULT})
option(TTS_CPP_BUILD_TESTS       "tts-cpp: build per-stage numerical validation harnesses"   ${TTS_CPP_STANDALONE_DEFAULT})
option(TTS_CPP_INSTALL           "tts-cpp: generate install rules"                           ON)
option(TTS_CPP_USE_SYSTEM_GGML   "tts-cpp: use system-installed GGML library"                OFF)
option(TTS_CPP_GGML_LIB_PREFIX   "tts-cpp: rename bundled ggml libraries to libtts-ggml-* (filename only; CMake target names and C symbols are unchanged). Prevents shared-library filename collisions when multiple consumers that vendor different ggml versions are loaded into the same host process. Has no effect when TTS_CPP_USE_SYSTEM_GGML=ON." ON)

if (CMAKE_CROSSCOMPILING)
    if (NOT DEFINED GGML_NATIVE OR GGML_NATIVE)
        message(STATUS "tts-cpp: cross-compiling to ${CMAKE_SYSTEM_NAME} -- "
                       "forcing GGML_NATIVE=OFF (ggml's ARM feature probes use try_run() "
                       "which CMake refuses in cross-compile mode).")
        set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    endif()
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "Android")
    if (NOT DEFINED GGML_BLAS OR GGML_BLAS)
        set(GGML_BLAS OFF CACHE BOOL "" FORCE)
    endif()
    if (NOT DEFINED GGML_ACCELERATE OR GGML_ACCELERATE)
        set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
    endif()
endif()

# Two related workarounds for clang-cl / MSVC builds on Windows.  Both
# come from msys2 sneaking GCC-flavoured libraries onto CMake's search
# paths and being mismatched against MSVC-compiled translation units.
# Both are no-ops on MinGW (excluded) and on non-Windows.
#
# (1) libm. ggml/src/CMakeLists.txt unconditionally links libm on Windows
#     unless ONEAPI_ROOT is set, which fails when find_library picks up a
#     stray libm.dll.a from msys2 (lld-link errors with `could not open
#     'm.lib'`). Set ONEAPI_ROOT in the configure-time environment so
#     ggml's existing escape hatch fires.  set(ENV{...}) only affects
#     the running CMake configure process — the same scope ggml's check
#     runs in.
#
# (2) OpenMP. CMake's FindOpenMP picks `-fopenmp=libomp` (LLVM-style) for
#     MSVC, but resolves OpenMP_*_LIBRARIES to msys2's libgomp.dll.a if
#     msys2 is on PATH.  The resulting object files reference LLVM
#     `__kmpc_*` symbols that GOMP doesn't provide and lld-link errors
#     out with `undefined symbol: __kmpc_global_thread_num`.  Disable
#     OpenMP for this toolchain; the perf hit on the CPU encoder is
#     bounded and only affects this specific build environment.
if (WIN32 AND NOT MINGW)
    if (NOT DEFINED ENV{ONEAPI_ROOT})
        set(ENV{ONEAPI_ROOT} "1")
    endif()
    if (NOT DEFINED GGML_OPENMP AND NOT DEFINED CACHE{GGML_OPENMP})
        # Normal (non-cache) variable on purpose: scoped to this
        # listfile + its add_subdirectory(ggml) below, doesn't poison
        # parent-project CMakeCache when tts-cpp is consumed via
        # add_subdirectory.  ggml's `option(GGML_OPENMP ... ON)` respects
        # an existing normal variable under CMP0077 (CMake 3.13+).
        set(GGML_OPENMP OFF)
    endif()
endif()

# Snapshot the user's BUILD_SHARED_LIBS intent BEFORE add_subdirectory(ggml).
# ggml's own CMake declares its own option(BUILD_SHARED_LIBS ...) with
# Windows-non-MinGW default = ON, so any later `if (DEFINED BUILD_SHARED_LIBS)`
# check sees the variable as defined regardless of what the user passed.
# We capture the user's explicit pre-ggml setting here and use that to drive
# the tts-cpp library linkage default.
set(_tts_cpp_user_set_shared FALSE)
if (DEFINED BUILD_SHARED_LIBS OR DEFINED CACHE{BUILD_SHARED_LIBS})
    set(_tts_cpp_user_set_shared TRUE)
endif()

if (NOT TARGET ggml)
    if (TTS_CPP_USE_SYSTEM_GGML)
        find_package(ggml CONFIG REQUIRED)
        if (NOT ggml_FOUND)
            message(FATAL_ERROR "System-installed GGML library not found.")
        endif()
        add_library(ggml ALIAS ggml::ggml)
    else()
        add_subdirectory(ggml)
    endif()
endif()

# Rename the bundled ggml shared/static library filenames so the speech
# stack (whisper, parakeet, chatterbox, supertonic) can vendor a single
# ggml file set without colliding with other addons that ship their own
# (fabric/qvac-ggml-*, diffusion-ggml-*).  CMake target names and C
# symbols are unchanged; only the on-disk filename gains the prefix.
function(tts_cpp_apply_ggml_prefix target)
    if (NOT TARGET ${target})
        return()
    endif()
    get_target_property(_tcgp_type ${target} TYPE)
    if (_tcgp_type STREQUAL "INTERFACE_LIBRARY" OR _tcgp_type STREQUAL "OBJECT_LIBRARY")
        return()
    endif()
    get_target_property(_tcgp_old ${target} OUTPUT_NAME)
    if (NOT _tcgp_old OR _tcgp_old STREQUAL "_tcgp_old-NOTFOUND")
        set(_tcgp_old ${target})
    endif()
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME "tts-${_tcgp_old}"
    )
endfunction()

if (TTS_CPP_GGML_LIB_PREFIX AND NOT TTS_CPP_USE_SYSTEM_GGML)
    foreach (_tcgp_target ggml ggml-base)
        tts_cpp_apply_ggml_prefix(${_tcgp_target})
    endforeach()
    if (DEFINED GGML_AVAILABLE_BACKENDS)
        foreach (_tcgp_target ${GGML_AVAILABLE_BACKENDS})
            tts_cpp_apply_ggml_prefix(${_tcgp_target})
        endforeach()
    endif()
    # Renaming the .so/.dll files alone is not enough: ggml's runtime
    # backend loader (`ggml_backend_load_best`) hard-codes the
    # `libggml-` / `ggml-` filename prefix when scanning under
    # GGML_BACKEND_DL=ON.  qvac-ext-ggml/speech absorbs the patch that
    # adds GGML_BACKEND_DL_PROJECT_PREFIX; defining it here teaches the
    # loader to look for our prefixed filenames at runtime.  Without
    # it the renamed .so/.dll files exist on disk but Vulkan / OpenCL
    # / CUDA backends silently fail to load on Android.
    if (TARGET ggml)
        target_compile_definitions(ggml PRIVATE
            GGML_BACKEND_DL_PROJECT_PREFIX="tts-"
        )
    endif()
    message(STATUS "tts-cpp: bundled ggml libraries will be emitted as libtts-ggml-* (set TTS_CPP_GGML_LIB_PREFIX=OFF to keep upstream filenames)")
endif()

find_package(OpenMP)

if (MSVC)
    add_compile_definitions(_USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
endif()

set(TTS_CPP_LIB_SOURCES
    src/main.cpp
    src/chatterbox_cli.cpp
    src/gpt2_bpe.cpp
    src/chatterbox_tts.cpp
    src/chatterbox_engine.cpp
    src/voice_features.cpp
    src/mel_extract_stft.cpp
    src/voice_encoder.cpp
    src/campplus.cpp
    src/s3tokenizer.cpp
    src/t3_mtl.cpp
    src/supertonic_gguf.cpp
    src/supertonic_preprocess.cpp
    src/supertonic_vocoder.cpp
    src/supertonic_duration.cpp
    src/supertonic_text_encoder.cpp
    src/supertonic_vector_estimator.cpp
    src/supertonic_engine.cpp
)

add_library(mtl_tokenizer STATIC src/mtl_tokenizer.cpp)
# $<BUILD_INTERFACE:...> so the source-tree path stays out of any
# INTERFACE_INCLUDE_DIRECTORIES that would otherwise leak into the
# install(EXPORT) and trip CMake's source-prefix check.
target_include_directories(mtl_tokenizer PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
set_target_properties(mtl_tokenizer PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (TTS_CPP_BUILD_LIBRARY)
    # Honour BUILD_SHARED_LIBS only when the *user* set it; otherwise
    # default to STATIC even though ggml's own CMake declares an
    # option(BUILD_SHARED_LIBS ...) that flips the variable to ON on
    # Windows non-MinGW.  Static is the right default here because the
    # supertonic test harnesses link against tts-cpp directly and use
    # detail-namespaced symbols (declared in src/supertonic_internal.h)
    # that aren't part of the TTS_CPP_API public surface; a SHARED
    # library would hide them and the tests would fail to link.
    # Consumers that want a SHARED tts-cpp pass -DBUILD_SHARED_LIBS=ON
    # explicitly and accept the export-surface restriction.
    if (_tts_cpp_user_set_shared)
        add_library(tts-cpp ${TTS_CPP_LIB_SOURCES})
    else()
        add_library(tts-cpp STATIC ${TTS_CPP_LIB_SOURCES})
    endif()
    add_library(tts-cpp::tts-cpp ALIAS tts-cpp)

    target_include_directories(tts-cpp
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
            ${CMAKE_CURRENT_SOURCE_DIR}/ggml/src
    )
    target_compile_features(tts-cpp PUBLIC cxx_std_17)
    target_link_libraries(tts-cpp PUBLIC ggml)
    target_link_libraries(tts-cpp PRIVATE mtl_tokenizer)

    # Symbol-export contract for shared builds.  Standard llama-style
    # PAIR: PUBLIC define so consumers see the import side of TTS_CPP_API
    # (declspec(dllimport) on Windows / visibility("default") elsewhere),
    # PRIVATE define inside the library's TUs to flip Windows from
    # dllimport to dllexport.  Both no-ops in static builds (the
    # default), so static consumers see exactly the same surface.
    get_target_property(_tts_cpp_type tts-cpp TYPE)
    if (_tts_cpp_type STREQUAL "SHARED_LIBRARY")
        target_compile_definitions(tts-cpp PUBLIC  TTS_CPP_SHARED)
        target_compile_definitions(tts-cpp PRIVATE TTS_CPP_BUILD)
    endif()

    if (OpenMP_CXX_FOUND)
        target_link_libraries(tts-cpp PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if (GGML_CUDA)
        target_compile_definitions(tts-cpp PRIVATE GGML_USE_CUDA)
    endif()
    if (GGML_METAL)
        target_compile_definitions(tts-cpp PRIVATE GGML_USE_METAL)
    endif()
    if (GGML_VULKAN)
        target_compile_definitions(tts-cpp PRIVATE GGML_USE_VULKAN)
    endif()
    if (GGML_OPENCL)
        target_compile_definitions(tts-cpp PRIVATE GGML_USE_OPENCL)
    endif()

    set_target_properties(tts-cpp PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME tts-cpp
    )
endif()

if (TTS_CPP_BUILD_EXECUTABLES)
    if (NOT TTS_CPP_BUILD_LIBRARY)
        message(FATAL_ERROR "TTS_CPP_BUILD_EXECUTABLES requires TTS_CPP_BUILD_LIBRARY=ON")
    endif()

    add_executable(tts-cli src/cli_main.cpp)
    target_link_libraries(tts-cli PRIVATE tts-cpp)

    add_executable(chatterbox src/cli_main.cpp)
    target_link_libraries(chatterbox PRIVATE tts-cpp)

    add_executable(mel2wav src/mel2wav.cpp)
    target_link_libraries(mel2wav PRIVATE ggml)
    target_include_directories(mel2wav PRIVATE ggml/include src)

    add_executable(supertonic-cli src/supertonic_cli.cpp)
    target_link_libraries(supertonic-cli PRIVATE tts-cpp)
endif()

if (TTS_CPP_BUILD_TESTS)
    add_executable(test-s3gen test/test_s3gen.cpp)
    target_link_libraries(test-s3gen PRIVATE ggml)
    target_include_directories(test-s3gen PRIVATE ggml/include src)

    add_executable(test-voice-features
        test/test_voice_features.cpp
        src/voice_features.cpp
        src/mel_extract_stft.cpp)
    target_link_libraries(test-voice-features PRIVATE ggml)
    target_include_directories(test-voice-features PRIVATE ggml/include src)

    add_executable(test-resample
        test/test_resample.cpp
        src/voice_features.cpp
        src/mel_extract_stft.cpp)
    target_link_libraries(test-resample PRIVATE ggml)
    target_include_directories(test-resample PRIVATE src)

    add_executable(test-voice-encoder
        test/test_voice_encoder.cpp
        src/voice_encoder.cpp
        src/voice_features.cpp
        src/mel_extract_stft.cpp)
    target_link_libraries(test-voice-encoder PRIVATE ggml)
    target_include_directories(test-voice-encoder PRIVATE ggml/include src)

    add_executable(test-campplus
        test/test_campplus.cpp
        src/campplus.cpp)
    target_link_libraries(test-campplus PRIVATE ggml)
    target_include_directories(test-campplus PRIVATE ggml/include src)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(test-campplus PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_executable(test-fbank
        test/test_fbank.cpp
        src/voice_features.cpp
        src/mel_extract_stft.cpp)
    target_link_libraries(test-fbank PRIVATE ggml)
    target_include_directories(test-fbank PRIVATE ggml/include src)

    add_executable(test-voice-embedding
        test/test_voice_embedding.cpp
        src/campplus.cpp
        src/voice_features.cpp
        src/mel_extract_stft.cpp)
    target_link_libraries(test-voice-embedding PRIVATE ggml)
    target_include_directories(test-voice-embedding PRIVATE ggml/include src)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(test-voice-embedding PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_executable(test-s3tokenizer
        test/test_s3tokenizer.cpp
        src/s3tokenizer.cpp)
    target_link_libraries(test-s3tokenizer PRIVATE ggml)
    target_include_directories(test-s3tokenizer PRIVATE ggml/include src)

    add_executable(test-mtl-tokenizer test/test_mtl_tokenizer.cpp)
    target_link_libraries(test-mtl-tokenizer PRIVATE mtl_tokenizer)
    target_include_directories(test-mtl-tokenizer PRIVATE src)

    add_executable(test-t3-mtl test/test_t3_mtl.cpp)
    target_link_libraries(test-t3-mtl PRIVATE tts-cpp ggml)
    target_include_directories(test-t3-mtl PRIVATE ggml/include src include)
    if (GGML_METAL)
        target_compile_definitions(test-t3-mtl PRIVATE GGML_USE_METAL)
    endif()

    add_executable(test-t3-mtl-stages test/test_t3_mtl_stages.cpp)
    target_link_libraries(test-t3-mtl-stages PRIVATE tts-cpp ggml)
    target_include_directories(test-t3-mtl-stages PRIVATE ggml/include src include)
    if (GGML_METAL)
        target_compile_definitions(test-t3-mtl-stages PRIVATE GGML_USE_METAL)
    endif()

    add_executable(test-streaming
        test/test_streaming.cpp
        src/chatterbox_tts.cpp)
    target_link_libraries(test-streaming PRIVATE ggml)
    target_include_directories(test-streaming PRIVATE ggml/include src include)

    add_executable(test-cpu-caches
        test/test_cpu_caches.cpp
        src/chatterbox_tts.cpp)
    target_link_libraries(test-cpu-caches PRIVATE ggml)
    target_include_directories(test-cpu-caches PRIVATE ggml/include src include)

    add_executable(test-t3-caches test/test_t3_caches.cpp)
    target_link_libraries(test-t3-caches PRIVATE tts-cpp ggml)
    target_include_directories(test-t3-caches PRIVATE ggml/include src include)

    add_executable(test-metal-ops test/test_metal_ops.cpp)
    target_link_libraries(test-metal-ops PRIVATE ggml)
    target_include_directories(test-metal-ops PRIVATE ggml/include src)
    if (GGML_METAL)
        target_compile_definitions(test-metal-ops PRIVATE GGML_USE_METAL)
    endif()

    if (NOT TTS_CPP_BUILD_LIBRARY)
        message(FATAL_ERROR "Supertonic tests require TTS_CPP_BUILD_LIBRARY=ON")
    endif()

    function(add_supertonic_harness target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE tts-cpp)
        target_include_directories(${target} PRIVATE ggml/include src include)
    endfunction()

    add_supertonic_harness(test-supertonic-preprocess test/test_supertonic_preprocess.cpp)
    add_supertonic_harness(test-supertonic-vocoder test/test_supertonic_vocoder.cpp)
    add_supertonic_harness(test-supertonic-vocoder-trace test/test_supertonic_vocoder_trace.cpp)
    add_supertonic_harness(test-supertonic-vocoder-pointwise test/test_supertonic_vocoder_pointwise.cpp)
    add_supertonic_harness(test-supertonic-duration test/test_supertonic_duration.cpp)
    add_supertonic_harness(test-supertonic-duration-trace test/test_supertonic_duration_trace.cpp)
    add_supertonic_harness(test-supertonic-text-encoder test/test_supertonic_text_encoder.cpp)
    add_supertonic_harness(test-supertonic-text-encoder-trace test/test_supertonic_text_encoder_trace.cpp)
    add_supertonic_harness(test-supertonic-vector test/test_supertonic_vector.cpp)
    add_supertonic_harness(test-supertonic-vector-trace test/test_supertonic_vector_trace.cpp)
    add_supertonic_harness(test-supertonic-pipeline test/test_supertonic_pipeline.cpp)
    add_supertonic_harness(supertonic-bench src/supertonic_bench.cpp)
endif()

set(TTS_CPP_POINTWISE_ACCEL_TARGETS
    tts-cpp
    test-supertonic-vector
    test-supertonic-vector-trace
    test-supertonic-vocoder
    test-supertonic-vocoder-trace
    test-supertonic-vocoder-pointwise
    test-supertonic-pipeline
    supertonic-bench)

if (APPLE)
    find_library(TTS_CPP_ACCELERATE_FRAMEWORK Accelerate)
    if (TTS_CPP_ACCELERATE_FRAMEWORK)
        foreach(_tts_accel_target ${TTS_CPP_POINTWISE_ACCEL_TARGETS})
            if (TARGET ${_tts_accel_target})
                target_link_libraries(${_tts_accel_target} PRIVATE ${TTS_CPP_ACCELERATE_FRAMEWORK})
                target_compile_definitions(${_tts_accel_target} PRIVATE TTS_CPP_USE_ACCELERATE)
            endif()
        endforeach()
    endif()
else()
    check_include_file_cxx(cblas.h TTS_CPP_HAS_CBLAS_H)
    find_package(BLAS)
    if (TTS_CPP_HAS_CBLAS_H AND BLAS_FOUND)
        foreach(_tts_cblas_target ${TTS_CPP_POINTWISE_ACCEL_TARGETS})
            if (TARGET ${_tts_cblas_target})
                target_link_libraries(${_tts_cblas_target} PRIVATE ${BLAS_LIBRARIES})
                target_compile_definitions(${_tts_cblas_target} PRIVATE TTS_CPP_USE_CBLAS)
            endif()
        endforeach()
    endif()
endif()

if (TTS_CPP_INSTALL AND TTS_CPP_BUILD_LIBRARY)
    include(CMakePackageConfigHelpers)

    # Re-install the bundled ggml targets into a `ggml-targets` export
    # set so install(EXPORT tts-cppTargets) can resolve the
    # `tts-cpp PUBLIC ggml` link as the namespaced `ggml::ggml`.  Without
    # this, CMake refuses to generate tts-cppTargets.cmake because the
    # in-tree `ggml` target is not part of any export set.  Replicates
    # the standard install(TARGETS ggml ggml-base EXPORT ggml-targets)
    # pattern that downstream ggml-based projects ship in their own
    # CMake (e.g. llama.cpp); we add it here at the tts-cpp level
    # because the upstream ggml at the pinned commit installs ggml
    # without an EXPORT clause.  The bundled ggml's hand-written
    # ggml-config.cmake (find_library + UNKNOWN IMPORTED) still ships
    # and is what's loaded via find_dependency(ggml) — we don't replace
    # it; the new ggml-targets.cmake just carries the export so
    # CMake's cross-package reference check passes here.
    if (NOT TTS_CPP_USE_SYSTEM_GGML)
        set(_tts_cpp_ggml_export_targets ggml ggml-base)
        if (DEFINED GGML_AVAILABLE_BACKENDS)
            list(APPEND _tts_cpp_ggml_export_targets ${GGML_AVAILABLE_BACKENDS})
        endif()
        # No PUBLIC_HEADER DESTINATION here: the bundled ggml's own
        # install(TARGETS ggml LIBRARY PUBLIC_HEADER) already copies the
        # ggml/include/*.h headers into the install tree.  Duplicating
        # the rule here re-resolves the same relative paths against the
        # tts-cpp source dir and `cmake --install` fails with
        # "file INSTALL cannot find <root>/include/ggml.h: File exists".
        # Keep this rule minimal: EXPORT + library destinations only,
        # so tts-cppTargets can resolve the namespaced ggml::ggml link
        # without rebuilding the headers install.
        install(TARGETS ${_tts_cpp_ggml_export_targets}
            EXPORT ggml-targets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
        install(EXPORT ggml-targets
            FILE ggml-targets.cmake
            NAMESPACE ggml::
            DESTINATION ${CMAKE_INSTALL_DATADIR}/ggml
        )
        unset(_tts_cpp_ggml_export_targets)
    endif()

    # Export the tts-cpp target via an install(EXPORT) set so the
    # generated tts-cppTargets.cmake carries the correct shape for
    # whatever linkage the build produced (STATIC vs SHARED, with
    # IMPORTED_IMPLIB on Windows shared builds, etc.).
    #
    # mtl_tokenizer is a private STATIC dep of tts-cpp; for static
    # archives, the PRIVATE link bleeds into the export interface
    # (static archives don't carry transitive PRIVATE deps the way
    # shared libs do), so the exporter would otherwise refuse with
    # "requires target mtl_tokenizer that is not in any export set".
    # Add it to the same export so consumers get a self-contained set.
    install(TARGETS tts-cpp mtl_tokenizer
        EXPORT tts-cppTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(DIRECTORY include/tts-cpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h"
    )
    install(EXPORT tts-cppTargets
        FILE tts-cppTargets.cmake
        NAMESPACE tts-cpp::
        DESTINATION ${CMAKE_INSTALL_DATADIR}/tts-cpp
    )

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tts-cppConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/tts-cppConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_DATADIR}/tts-cpp
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/tts-cppConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMinorVersion
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/tts-cppConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/tts-cppConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_DATADIR}/tts-cpp
    )
endif()

cmake_minimum_required(VERSION 3.14)
project(qvac-tts-cpp LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(QVAC_TTS_STANDALONE_DEFAULT ON)
else()
    set(QVAC_TTS_STANDALONE_DEFAULT OFF)
endif()

option(QVAC_TTS_BUILD_LIBRARY     "qvac-tts: build the static qvac-tts library"                 ON)
option(QVAC_TTS_BUILD_EXECUTABLES "qvac-tts: build the qvac-tts CLI + mel2wav demo"             ${QVAC_TTS_STANDALONE_DEFAULT})
option(QVAC_TTS_BUILD_TESTS       "qvac-tts: build per-stage numerical validation harnesses"    ${QVAC_TTS_STANDALONE_DEFAULT})
option(QVAC_TTS_INSTALL           "qvac-tts: generate install rules"                            ON)

add_subdirectory(ggml)

find_package(OpenMP)

set(QVAC_TTS_LIB_SOURCES
    src/main.cpp
    src/gpt2_bpe.cpp
    src/chatterbox_tts.cpp
    src/chatterbox_engine.cpp
    src/voice_features.cpp
    src/mel_extract_stft.cpp
    src/voice_encoder.cpp
    src/campplus.cpp
    src/s3tokenizer.cpp
)

if (QVAC_TTS_BUILD_LIBRARY)
    add_library(qvac-tts STATIC ${QVAC_TTS_LIB_SOURCES})
    add_library(qvac-tts::qvac-tts ALIAS qvac-tts)

    target_include_directories(qvac-tts
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    )
    target_link_libraries(qvac-tts PUBLIC ggml)

    if (OpenMP_CXX_FOUND)
        target_link_libraries(qvac-tts PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if (GGML_CUDA)
        target_compile_definitions(qvac-tts PRIVATE GGML_USE_CUDA)
    endif()
    if (GGML_METAL)
        target_compile_definitions(qvac-tts PRIVATE GGML_USE_METAL)
    endif()
    if (GGML_VULKAN)
        target_compile_definitions(qvac-tts PRIVATE GGML_USE_VULKAN)
    endif()

    set_target_properties(qvac-tts PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME qvac-tts
    )
endif()

if (QVAC_TTS_BUILD_EXECUTABLES)
    if (NOT QVAC_TTS_BUILD_LIBRARY)
        message(FATAL_ERROR "QVAC_TTS_BUILD_EXECUTABLES requires QVAC_TTS_BUILD_LIBRARY=ON")
    endif()

    add_executable(qvac-tts-cli src/cli_main.cpp)
    target_link_libraries(qvac-tts-cli PRIVATE qvac-tts)
    set_target_properties(qvac-tts-cli PROPERTIES OUTPUT_NAME qvac-tts)

    add_executable(mel2wav src/mel2wav.cpp)
    target_link_libraries(mel2wav PRIVATE ggml)
    target_include_directories(mel2wav PRIVATE ggml/include src)
endif()

if (QVAC_TTS_BUILD_TESTS)
    add_executable(test-s3gen src/test_s3gen.cpp)
    target_link_libraries(test-s3gen PRIVATE ggml)
    target_include_directories(test-s3gen PRIVATE ggml/include src)

    add_executable(test-voice-features
        src/test_voice_features.cpp
        src/voice_features.cpp
        src/mel_extract_stft.cpp)
    target_link_libraries(test-voice-features PRIVATE ggml)
    target_include_directories(test-voice-features PRIVATE ggml/include src)

    add_executable(test-resample
        src/test_resample.cpp
        src/voice_features.cpp
        src/mel_extract_stft.cpp)
    target_link_libraries(test-resample PRIVATE ggml)
    target_include_directories(test-resample PRIVATE src)

    add_executable(test-voice-encoder
        src/test_voice_encoder.cpp
        src/voice_encoder.cpp
        src/voice_features.cpp
        src/mel_extract_stft.cpp)
    target_link_libraries(test-voice-encoder PRIVATE ggml)
    target_include_directories(test-voice-encoder PRIVATE ggml/include src)

    add_executable(test-campplus
        src/test_campplus.cpp
        src/campplus.cpp)
    target_link_libraries(test-campplus PRIVATE ggml)
    target_include_directories(test-campplus PRIVATE ggml/include src)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(test-campplus PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_executable(test-fbank
        src/test_fbank.cpp
        src/voice_features.cpp
        src/mel_extract_stft.cpp)
    target_link_libraries(test-fbank PRIVATE ggml)
    target_include_directories(test-fbank PRIVATE ggml/include src)

    add_executable(test-voice-embedding
        src/test_voice_embedding.cpp
        src/campplus.cpp
        src/voice_features.cpp
        src/mel_extract_stft.cpp)
    target_link_libraries(test-voice-embedding PRIVATE ggml)
    target_include_directories(test-voice-embedding PRIVATE ggml/include src)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(test-voice-embedding PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_executable(test-s3tokenizer
        src/test_s3tokenizer.cpp
        src/s3tokenizer.cpp)
    target_link_libraries(test-s3tokenizer PRIVATE ggml)
    target_include_directories(test-s3tokenizer PRIVATE ggml/include src)

    add_executable(test-streaming
        src/test_streaming.cpp
        src/chatterbox_tts.cpp)
    target_link_libraries(test-streaming PRIVATE ggml)
    target_include_directories(test-streaming PRIVATE ggml/include src include)

    add_executable(test-metal-ops src/test_metal_ops.cpp)
    target_link_libraries(test-metal-ops PRIVATE ggml)
    target_include_directories(test-metal-ops PRIVATE ggml/include src)
    if (GGML_METAL)
        target_compile_definitions(test-metal-ops PRIVATE GGML_USE_METAL)
    endif()
endif()

if (QVAC_TTS_INSTALL AND QVAC_TTS_BUILD_LIBRARY)
    install(TARGETS qvac-tts
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(DIRECTORY include/qvac-tts
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h"
    )
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/qvac-tts-cppConfig.cmake
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/qvac-tts-cppConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_DATADIR}/qvac-tts-cpp
    )
endif()

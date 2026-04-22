include(CMakeFindDependencyMacro)
find_dependency(ggml CONFIG)

get_filename_component(_QVAC_TTS_CPP_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

find_library(QVAC_TTS_LIBRARY
    NAMES qvac-tts
    PATHS "${_QVAC_TTS_CPP_PREFIX}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)

find_path(QVAC_TTS_INCLUDE_DIR
    NAMES qvac-tts/qvac-tts.h
    PATHS "${_QVAC_TTS_CPP_PREFIX}/include"
    NO_DEFAULT_PATH
    REQUIRED
)

if(NOT TARGET qvac-tts::qvac-tts)
    add_library(qvac-tts::qvac-tts STATIC IMPORTED)
    set_target_properties(qvac-tts::qvac-tts PROPERTIES
        IMPORTED_LOCATION             "${QVAC_TTS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${QVAC_TTS_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES      "ggml::ggml"
    )
endif()

unset(_QVAC_TTS_CPP_PREFIX)

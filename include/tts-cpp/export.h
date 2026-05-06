#pragma once

// TTS_CPP_API marks exported symbols when building/using a shared library;
// expands to nothing for static builds (the default).
//
// TTS_CPP_SHARED  define when linking against or building libtts-cpp as a
//                 DLL / shared object.  Set automatically as a PUBLIC
//                 compile definition on the tts-cpp target when
//                 BUILD_SHARED_LIBS=ON, so consumers picking the target
//                 up via find_package(tts-cpp) inherit it transparently.
//
// TTS_CPP_BUILD   define only inside translation units that compile the
//                 library itself (set as a PRIVATE compile definition on
//                 the tts-cpp target).  Flips Windows from `dllimport`
//                 (consumer side) to `dllexport` (library side).
//
// Both are no-ops in static builds, so static consumers see exactly the
// same surface as before this header was introduced.

#ifdef TTS_CPP_SHARED
#  if defined(_WIN32) && !defined(__MINGW32__)
#    ifdef TTS_CPP_BUILD
#      define TTS_CPP_API __declspec(dllexport)
#    else
#      define TTS_CPP_API __declspec(dllimport)
#    endif
#  else
#    define TTS_CPP_API __attribute__((visibility("default")))
#  endif
#else
#  define TTS_CPP_API
#endif

# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY NewtonianEulerLimiters)

set(LIBRARY_SOURCES
  Helpers.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  )

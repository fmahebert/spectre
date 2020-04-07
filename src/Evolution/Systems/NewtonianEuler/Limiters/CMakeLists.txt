# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY NewtonianEulerLimiters)

set(LIBRARY_SOURCES
  CharacteristicHelpers.cpp
  Minmod.cpp
  Weno.cpp
  WenoType.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  Domain
  ErrorHandling
  Limiters
  NewtonianEuler
  Utilities
  )

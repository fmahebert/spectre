# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY NewtonianEuler)

set(LIBRARY_SOURCES
  Characteristics.cpp
  ConservativeFromPrimitive.cpp
  FixConsMeansToAtmosphere.cpp
  FixToAtmosphere.cpp
  Fluxes.cpp
  PostLimiterFlattening.cpp
  PostLimiterPpFlux.cpp
  PrimitiveFromConservative.cpp
  SoundSpeedSquared.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  INTERFACE Hydro
  )

add_subdirectory(Limiters)
add_subdirectory(NumericalFluxes)
add_subdirectory(Sources)

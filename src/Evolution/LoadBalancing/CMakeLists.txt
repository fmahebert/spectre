# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY LoadBalancing)

set(LIBRARY_SOURCES
  DistributionStrategies.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )


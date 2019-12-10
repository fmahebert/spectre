# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_NewtonianEulerLimiters")

set(LIBRARY_SOURCES
  Test_Flattener.cpp
  Test_Minmod.cpp
  Test_Weno.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/NewtonianEuler/Limiters"
  "${LIBRARY_SOURCES}"
  "NewtonianEulerLimiters"
  )

# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_NewtonianEulerNumericalFluxes")

set(LIBRARY_SOURCES
  Test_Hllc.cpp
  Test_MaxNormalFluxes.cpp
  Test_PositivityPreservingLaxFriedrichs.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/NewtonianEuler/NumericalFluxes"
  "${LIBRARY_SOURCES}"
  "NewtonianEulerNumericalFluxes"
  )

# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Convergence
  CoordinateMaps
  DataStructures
  DiscontinuousGalerkin
  Domain
  DomainCreators
  Elliptic
  EllipticDg
  ErrorHandling
  Events
  EventsAndTriggers
  Informer
  Initialization
  IO
  LinearOperators
  MathFunctions
  Options
  Parallel
  ParallelLinearSolver
  ParallelNonlinearSolver
  Utilities
  Xcts
  XctsAnalyticData
  XctsBoundaryConditions
  XctsSolutions
  )

add_spectre_parallel_executable(
  SolveXcts
  SolveXcts
  Elliptic/Executables/Xcts
  "Metavariables"
  "${LIBS_TO_LINK}"
  )

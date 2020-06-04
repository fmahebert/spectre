# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Domain
  LoadBalancing
  Parallel
  )

function(add_simple_load_balancer_executable DIM)
  add_spectre_parallel_executable(
    "SimpleLoadBalancerTest${DIM}D"
    SimpleLoadBalancingTest
    Evolution/Executables/LoadBalancing
    "EvolutionMetavars<${DIM}, false>"
    "${LIBS_TO_LINK}"
    )
  add_spectre_parallel_executable(
    "SimpleLoadBalancerTest${DIM}DWithSync"
    SimpleLoadBalancingTest
    Evolution/Executables/LoadBalancing
    "EvolutionMetavars<${DIM}, true>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_simple_load_balancer_executable)

add_simple_load_balancer_executable(1)
add_simple_load_balancer_executable(2)
add_simple_load_balancer_executable(3)

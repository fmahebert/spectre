// Distributed under the MIT License.
// See LICENSE.txt for details.
#include "Evolution/LoadBalancing/Actions/InitializeLoadBalancingTestArray.hpp"

#include <charm++.h>
#include <unordered_map>
#include <vector>

#include "AlgorithmArray.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Evolution/LoadBalancing/LoadBalancingTestArray.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "Utilities/Gsl.hpp"

namespace Lb::Action {
void set_this_element_migratable(
    Parallel::ConstGlobalCache<Metavariables>& cache,
    const ElementId<Dim>& array_index) noexcept; {
  auto& lb_element_array =
      Parallel::get_parallel_component<LoadBalancingTestArray<Metavariables>>(
          cache);
  lb_element_array(array_index).ckLocal()->setMigratable(true);
}
}  // namespace Lb::Action

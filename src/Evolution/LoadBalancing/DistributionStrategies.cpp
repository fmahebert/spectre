// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/LoadBalancing/DistributionStrategies.hpp"

// TODO CMAKE

namespace Lb::Distribution {

/// \cond
PUP::able::PUP_ID RoundRobin::my_PUP_ID = 0;
PUP::able::PUP_ID KnownEvenOptimal::my_PUP_ID = 0;
PUP::able::PUP_ID KnownMaxFragmented::my_PUP_ID = 0;
/// \endcond
}

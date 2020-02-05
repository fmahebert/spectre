// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace Initialization {
namespace Actions {
/// \ingroup InitializationGroup
/// \brief Allocate items for minmod limiter
///
/// DataBox changes:
/// - Adds:
///   * `Tags::LimiterDiagnostics`
///   * `Tags::SizeOfElement<Dim>`
///
/// - Removes: nothing
/// - Modifies: nothing
template <size_t Dim>
struct Minmod {
  using simple_tags = db::AddSimpleTags<::Tags::LimiterDiagnostics>;
  using compute_tags = tmpl::list<domain::Tags::SizeOfElementCompute<Dim>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const size_t num_grid_points =
        db::get<domain::Tags::Mesh<Dim>>(box).number_of_grid_points();
    Scalar<DataVector> limiter_diagnostics(num_grid_points, 0.);
    return std::make_tuple(
        merge_into_databox<Minmod, simple_tags, compute_tags>(
            std::move(box), std::move(limiter_diagnostics)));
  }
};
}  // namespace Actions
}  // namespace Initialization

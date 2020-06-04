// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "AlgorithmArray.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/LoadBalancing/LoadBalancingTestArray.hpp"
#include "Evolution/LoadBalancing/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Tags.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Lb {
/// \cond
template <typename Metavariables>
struct LoadBalancingTestArray;
/// \endcond

namespace Actions {

template <size_t Dim>
struct InitializeLoadBalancingTestArray {
  using initialization_tags =
      tmpl::list<domain::Tags::InitialExtents<Dim>,
                 domain::Tags::InitialRefinementLevels<Dim>,
                 Tags::ExecutionLoad, Tags::InternalStorageSize,
                 Tags::CommunicationSize>;
  using initialization_tags_to_keep =
      tmpl::list<Tags::ExecutionLoad, Tags::InternalStorageSize,
                 Tags::CommunicationSize,
                 domain::Tags::InitialRefinementLevels<Dim>>;

  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<Dim>>;
  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ElementId<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    LBTurnInstrumentOn();

    using simple_tags =
        db::AddSimpleTags<domain::Tags::Element<Dim>,
                          domain::Tags::ElementMap<Dim>, Tags::InternalStorage,
                          Tags::NeighborData<Dim>, Tags::StepNumber>;
    std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<double> dist{0.1, 1.0};
    std::vector<DataVector> internal_storage(
        db::get<Tags::InternalStorageSize>(box));
    for (auto& vector_entry : internal_storage) {
      vector_entry = DataVector{10_st};
      for (auto& val : vector_entry) {
        val = dist(gen);
      }
    }
    const auto& initial_refinement =
        db::get<domain::Tags::InitialRefinementLevels<Dim>>(box);
    const auto& domain = db::get<domain::Tags::Domain<Dim>>(box);
    const ElementId<Dim> element_id{array_index};
    const auto& my_block = domain.blocks()[element_id.block_id()];
    Element<Dim> element = domain::Initialization::create_initial_element(
        element_id, my_block, initial_refinement);
    ElementMap<Dim, Frame::Inertial> element_map{
        element_id, my_block.stationary_map().get_clone()};

    FixedHashMap<maximum_number_of_neighbors(Dim), dg::MortarId<Dim>,
                 std::vector<DataVector>, boost::hash<dg::MortarId<Dim>>>
        neighbor_data;
    for (const auto& direction_and_neighbors : element.neighbors()) {
      for (const auto& neighbor : direction_and_neighbors.second) {
        std::vector<DataVector> neighbor_data_entry(
            db::get<Tags::CommunicationSize>(box));
        for (auto& vector : neighbor_data_entry) {
          vector = DataVector{10, 0.0};
        }
        neighbor_data.emplace(
            dg::MortarId<Dim>(direction_and_neighbors.first, neighbor),
            std::move(neighbor_data_entry));
      }
    }
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeLoadBalancingTestArray, simple_tags, db::AddComputeTags<>,
            Initialization::MergePolicy::Overwrite>(
            std::move(box), std::move(element), std::move(element_map),
            std::move(internal_storage), std::move(neighbor_data), 0_st));
  }
};
}  // namespace Actions
}  // namespace Lb

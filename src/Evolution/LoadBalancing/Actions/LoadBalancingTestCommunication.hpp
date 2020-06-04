// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/HasReceivedFromAllMortars.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

#include "Parallel/Printf.hpp"

namespace Lb {

/// \cond
template <typename Metavariables>
struct LoadBalancingTestArray;
/// \endcond

// TODO this should probably get moved to parallel somewhere.
template <typename Metavariables, typename FromComponent, typename ToComponent,
          typename DbTagList, size_t Dim>
void record_communication_event_projections_stream(
    const std::array<int, Dim>* from_array_index,
    const std::array<int, Dim>* to_array_index,
    const db::DataBox<DbTagList>& box) noexcept {
  (void)from_array_index;
  (void)to_array_index;
  (void)box;
  // if we're tracing with projections, insert into the event stream the user
  // note of the origin and destination

  // ideally, these should immediately precede a true communication event when
  // the two are on different PEs and not otherwise.
#ifdef SPECTRE_CHARM_PROJECTIONS
  std::string from_array_index_string = std::to_string((*from_array_index)[0]);
  for (size_t i = 1; i < Dim; ++i) {
    from_array_index_string += ":" + std::to_string((*from_array_index)[i]);
  }

  std::string to_array_index_string = std::to_string((*to_array_index)[0]);
  for (size_t i = 1; i < Dim; ++i) {
    to_array_index_string += ":" + std::to_string((*to_array_index)[i]);
  }

  std::string note = "C: " + pretty_type::short_name<FromComponent>() + " " +
                     from_array_index_string + " " +
                     pretty_type::short_name<ToComponent>() + " " +
                     to_array_index_string + "\n";
  // This ends up being... several records. Ideally we'd be able to record a
  // note, but I didn't manage to get that to work.
  traceUserSuppliedData(-1);
  traceUserSuppliedData(db::get<Tags::GraphDumpLabel>(box));
  traceUserSuppliedData(Parallel::my_proc());
  traceUserSuppliedData(tmpl::index_of<typename Metavariables::component_list,
                                       FromComponent>::value);
  for (size_t i = 0; i < Dim; ++i) {
    traceUserSuppliedData((*from_array_index)[i]);
  }
  traceUserSuppliedData(tmpl::index_of<typename Metavariables::component_list,
                                       ToComponent>::value);
  for (size_t i = 0; i < Dim; ++i) {
    traceUserSuppliedData((*to_array_index)[i]);
  }
  traceUserSuppliedData(-2);
#endif
}

template <typename DbTagList>
void increment_graph_label(db::DataBox<DbTagList>& box) noexcept {
  (void)box;
#ifdef SPECTRE_CHARM_PROJECTIONS
  db::mutate<Tags::GraphDumpLabel>(
      make_not_null(&box),
      [](const gsl::not_null<size_t*> label) noexcept { ++(*label); });
#endif
}

void record_communication_event_projections_done_messages() noexcept {
#ifdef SPECTRE_CHARM_PROJECTIONS
  traceUserSuppliedData(-3);
#endif
}

namespace Actions {

template <size_t Dim>
struct SendDataToNeighbors {
  static constexpr size_t volume_dim = Dim;

  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const ElementId<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    if (db::get<Tags::StepNumber>(box) >= db::get<Tags::NumberOfSteps>(box)) {
      return std::forward_as_tuple(std::move(box));
    }
    bool is_triggered = false;
#ifdef SPECTRE_CHARM_PROJECTIONS
    is_triggered = db::get<Tags::GraphDumpTrigger<
        typename Metavariables::graph_dump_triggers>>(box)
                       .is_triggered(box);
#endif
    const auto& element = db::get<domain::Tags::Element<volume_dim>>(box);
    auto& receiver_proxy =
        Parallel::get_parallel_component<LoadBalancingTestArray<Metavariables>>(
            cache);
    auto& internal_storage = db::get<Tags::InternalStorage>(box);
    size_t i = 0;
    // just send the first collection of vectors to the neighbors
    for (const auto& direction_and_neighbors : element.neighbors()) {
      for (const auto& neighbor : direction_and_neighbors.second) {
        std::vector<DataVector> bundled_data(
            db::get<Tags::CommunicationSize>(box));
        for (auto& vector_entry : bundled_data) {
          vector_entry = internal_storage[i % internal_storage.size()];
          ++i;
        }
        const auto& direction = direction_and_neighbors.first;
        const auto& orientation = direction_and_neighbors.second.orientation();
        const auto direction_from_neighbor = orientation(direction.opposite());

        // if this isn't enforced, sometimes when running on few cores, the
        // element will mistakenly start back up the algorithm on an element
        // that's still in Initialization, which will cause incomprehensible
        // errors
        if (is_triggered) {
          record_communication_event_projections_stream<
              Metavariables, LoadBalancingTestArray<Metavariables>,
              LoadBalancingTestArray<Metavariables>>(
              reinterpret_cast<const std::array<int, Dim>*>(&array_index),
              reinterpret_cast<const std::array<int, Dim>*>(&neighbor), box);
        }
        if (db::get<Tags::StepNumber>(box) > 1) {
          Parallel::receive_data<ReceiveTags::LoadBalancingCommunication<Dim>>(
              receiver_proxy[neighbor], db::get<Tags::StepNumber>(box),
              std::make_pair(dg::MortarId<volume_dim>{direction_from_neighbor,
                                                      element.id()},
                             std::move(bundled_data)),
              true);
        } else {
          Parallel::receive_data<ReceiveTags::LoadBalancingCommunication<Dim>>(
              receiver_proxy[neighbor], db::get<Tags::StepNumber>(box),
              std::make_pair(dg::MortarId<volume_dim>{direction_from_neighbor,
                                                      element.id()},
                             std::move(bundled_data)));
        }
        if (is_triggered) {
          record_communication_event_projections_done_messages();
        }
      }
    }
    if (is_triggered) {
      increment_graph_label(box);
    }
    return std::forward_as_tuple(std::move(box));
  }
};

template <size_t Dim>
struct ReceiveDataFromNeighbors {
  using inbox_tags = tmpl::list<ReceiveTags::LoadBalancingCommunication<Dim>>;
  static constexpr size_t volume_dim = Dim;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    if (UNLIKELY(
            get<domain::Tags::Element<volume_dim>>(box).number_of_neighbors() ==
                0 or
            db::get<Tags::StepNumber>(box) == 0)) {
      return {std::move(box)};
    }

    const auto& step = db::get<Tags::StepNumber>(box) - 1;
    auto& inbox =
        tuples::get<ReceiveTags::LoadBalancingCommunication<Dim>>(inboxes)
            .find(step)
            ->second;
    db::mutate<Tags::NeighborData<Dim>>(
        make_not_null(&box),
        [&inbox](const gsl::not_null<FixedHashMap<
                     maximum_number_of_neighbors(Dim), dg::MortarId<Dim>,
                     std::vector<DataVector>, boost::hash<dg::MortarId<Dim>>>*>
                     neighbor_data) noexcept {
          for (auto& received_mortar_data : inbox) {
            neighbor_data->at(received_mortar_data.first) =
                std::move(received_mortar_data.second);
          }
        });
    tuples::get<ReceiveTags::LoadBalancingCommunication<Dim>>(inboxes).erase(
        step);
    return {std::move(box)};
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(
      const db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) noexcept {
    return db::get<Tags::StepNumber>(box) == 0 or
           dg::has_received_from_all_mortars<
               ReceiveTags::LoadBalancingCommunication<Dim>>(
               db::get<Tags::StepNumber>(box) - 1,
               get<domain::Tags::Element<volume_dim>>(box), inboxes);
  }
};

}  // namespace Actions
}  // namespace Lb

// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "Evolution/LoadBalancing/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Gsl.hpp"

namespace Lb {
namespace Actions {

struct DefaultLoadFunction {
  static void apply(
      const gsl::not_null<std::vector<DataVector>*> internal_storage,
      const std::vector<DataVector>& neighbor_data) noexcept {
    for (size_t i = 0; i < internal_storage->size(); ++i) {
      (*internal_storage)[i] =
          (*internal_storage)[i * 3 % internal_storage->size()] +
          0.01 *
              (square((*internal_storage)[i * 5 % internal_storage->size()]) +
               (*internal_storage)[i * 7 % internal_storage->size()] *
                   neighbor_data[i % neighbor_data.size()] +
               neighbor_data[i * 3 % neighbor_data.size()] -
               neighbor_data[i * 5 % neighbor_data.size()]);
      // keep things below 1:
      for (auto& val : (*internal_storage)[i]) {
        if (val > 1.0 or val < -1.0) {
          val = 1.0 / val;
        }
      }
    }
  }
  static void apply(
      const gsl::not_null<std::vector<DataVector>*> internal_storage) noexcept {
    for (size_t i = 0; i < internal_storage->size(); ++i) {
      (*internal_storage)[i] =
          (*internal_storage)[i * 3 % internal_storage->size()] +
          0.1 * (square((*internal_storage)[i * 5 % internal_storage->size()]) /
                 max((*internal_storage)[i * 5 % internal_storage->size()]));
      // keep things below 1:
      for (auto& val : (*internal_storage)[i]) {
        if (val > 1.0 or val < -1.0) {
          val = 1.0 / val;
        }
      }
    }
  }

  static size_t load_weight(
      const gsl::not_null<std::vector<DataVector>*> internal_storage,
      const std::vector<DataVector>& /*neighbor_data*/) noexcept {
    return load_weight(internal_storage);
  }
  static size_t load_weight(
      const gsl::not_null<std::vector<DataVector>*> internal_storage) noexcept {
    return internal_storage->size();
  }
};

template <typename LoadFunction = DefaultLoadFunction>
struct EmulateLoad {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    size_t i = 0;
    while (i < db::get<Tags::ExecutionLoad>(box)) {
      if (UNLIKELY(get<domain::Tags::Element<Metavariables::volume_dim>>(box)
                       .number_of_neighbors() == 0)) {
        db::mutate<Tags::InternalStorage>(
            make_not_null(&box),
            [&i](const gsl::not_null<std::vector<DataVector>*>
                     internal_storage) noexcept {
              LoadFunction::apply(internal_storage);
              i += LoadFunction::load_weight(internal_storage);
            });
        if (i > db::get<Tags::ExecutionLoad>(box)) {
          break;
        }
      } else {
        for (const auto& neighbor_data :
             db::get<Tags::NeighborData<Metavariables::volume_dim>>(box)) {
          db::mutate<Tags::InternalStorage>(
              make_not_null(&box),
              [&i](const gsl::not_null<std::vector<DataVector>*>
                       internal_storage,
                   const std::vector<DataVector>& neighbor_data) noexcept {
                LoadFunction::apply(internal_storage, neighbor_data);
                i += LoadFunction::load_weight(internal_storage, neighbor_data);
              },
              neighbor_data.second);
          if (i > db::get<Tags::ExecutionLoad>(box)) {
            break;
          }
        }
      }
    }
    return {std::move(box)};
  }
};

}  // namespace Actions
}  // namespace Lb

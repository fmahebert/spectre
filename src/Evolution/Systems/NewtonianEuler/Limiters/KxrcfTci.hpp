// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Element.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"
#include "Domain/OrientationMapHelpers.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Systems/NewtonianEuler/SoundSpeedSquared.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"

#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "Options/Options.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
template <size_t VolumeDim>
class Direction;
template <size_t VolumeDim>
class ElementId;
/// \endcond

namespace NewtonianEuler {
namespace Limiters {
namespace Tci {

template <size_t VolumeDim, typename PackagedData>
bool kxrcf_indicator(
    const Scalar<DataVector>& cons_mass_density,
    const tnsr::I<DataVector, VolumeDim>& cons_momentum_density,
    const Scalar<DataVector>& cons_energy_density,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data,
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const std::unordered_map<Direction<VolumeDim>,
                             tnsr::i<DataVector, VolumeDim>>&
        internal_unit_normals,
    const double kxrcf_constant) noexcept {
  // Enforce restrictions on h-refinement, p-refinement
  if (UNLIKELY(alg::any_of(element.neighbors(),
                           [](const auto& direction_neighbors) noexcept {
                             return direction_neighbors.second.size() != 1;
                           }))) {
    ERROR("The Kxrcf TCI does not yet support h-refinement");
    // Removing this limitation will require adapting the surface integrals to
    // correctly acount for,
    // - multiple (smaller) neighbors contributing to the integral
    // - only a portion of a (larger) neighbor contributing to the integral
  }
  alg::for_each(neighbor_data, [&mesh](const auto& neighbor_and_data) noexcept {
    if (UNLIKELY(neighbor_and_data.second.mesh != mesh)) {
      ERROR("The Kxrcf TCI does not yet support p-refinement");
      // Removing this limitation will require generalizing the surface
      // integrals to make sure the meshes are consistent.
    }
  });

  bool inflow_boundaries_present = false;
  double inflow_area = 0.;
  double inflow_delta_density = 0.;
  double inflow_delta_energy = 0.;

  for (const auto& neighbor_and_data : neighbor_data) {
    const auto& neighbor = neighbor_and_data.first;
    const auto& dir = neighbor.first;
    if (internal_unit_normals.find(dir) == internal_unit_normals.end()) {
      // This direction is to an external neighbor, so we skip it
      continue;
    }

    const size_t sliced_dim = dir.dimension();
    const size_t index_of_edge =
        (dir.side() == Side::Lower ? 0 : mesh.extents()[sliced_dim] - 1);
    const auto momentum_on_edge = data_on_slice(
        cons_momentum_density, mesh.extents(), sliced_dim, index_of_edge);
    const auto momentum_dot_normal =
        dot_product(momentum_on_edge, internal_unit_normals.at(dir));

    if (min(get(momentum_dot_normal)) > -1e-12) {
      // Boundary has no (significant) inflow, so we skip it
      // TODO: check cutoff ... should this be tuneable?
      continue;
    }

    // This boundary has inflow
    inflow_boundaries_present = true;

    // This mask has value 1. for momentum_dot_normal < 0.
    //                     0. for momentum_dot_normal >= 0.
    const DataVector inflow_mask = 1. - step_function(get(momentum_dot_normal));

    // TODO add jacobians
    inflow_area += definite_integral(inflow_mask, mesh.slice_away(sliced_dim));

    const auto neighbor_vars_on_edge =
        data_on_slice(neighbor_data.at(neighbor).volume_data, mesh.extents(),
                      sliced_dim, index_of_edge);

    const auto density_on_edge = data_on_slice(
        cons_mass_density, mesh.extents(), sliced_dim, index_of_edge);
    const auto& neighbor_density_on_edge =
        get<NewtonianEuler::Tags::MassDensityCons>(neighbor_vars_on_edge);
    inflow_delta_density += definite_integral(
        (get(density_on_edge) - get(neighbor_density_on_edge)) * inflow_mask,
        mesh.slice_away(sliced_dim));

    const auto energy_on_edge = data_on_slice(
        cons_energy_density, mesh.extents(), sliced_dim, index_of_edge);
    const auto& neighbor_energy_on_edge =
        get<NewtonianEuler::Tags::EnergyDensity>(neighbor_vars_on_edge);
    inflow_delta_energy += definite_integral(
        (get(energy_on_edge) - get(neighbor_energy_on_edge)) * inflow_mask,
        mesh.slice_away(sliced_dim));
  }

  if (not inflow_boundaries_present) {
    // No boundaries had inflow, so not a troubled cell
    return false;
  }

  // KXRCF take h to be the radius of the circumscribed circle
  // TODO FH's first version was missing the 1/2 here, test effect of this
  const double h = 0.5 * magnitude(element_size);
  const double h_pow = pow(h, 0.5 * mesh.extents(0));

  // TODO add jacobians
  ASSERT(mean_value(square(get(cons_mass_density)), mesh) > 0., "oops");
  const double norm_density =
      sqrt(mean_value(square(get(cons_mass_density)), mesh));
  ASSERT(inflow_area > 0., "oops");
  ASSERT(norm_density > 0., "oops");
  const double ratio_for_density =
      abs(inflow_delta_density) / (h_pow * inflow_area * norm_density);

  const double norm_energy =
      sqrt(mean_value(square(get(cons_energy_density)), mesh));
  ASSERT(norm_energy > 0., "oops");
  const double ratio_for_energy =
      abs(inflow_delta_energy) / (h_pow * inflow_area * norm_energy);

  return (ratio_for_density > kxrcf_constant or
          ratio_for_energy > kxrcf_constant);
}

}  // namespace Tci
}  // namespace Limiters
}  // namespace NewtonianEuler

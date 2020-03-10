// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/PostLimiterPpFlux.hpp"

#include <pup.h>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Overloader.hpp"

namespace VariableFixing {
namespace NewtonianEuler {

template <size_t Dim>
PostLimiterPpFlux<Dim>::PostLimiterPpFlux(const bool disable_for_debugging)
    : disable_for_debugging_(disable_for_debugging) {}

// clang-tidy: google-runtime-references
template <size_t Dim>
void PostLimiterPpFlux<Dim>::pup(PUP::er& p) noexcept {  // NOLINT
  p | disable_for_debugging_;
}

template <size_t Dim>
bool PostLimiterPpFlux<Dim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> mass_density,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const Mesh<Dim>& mesh,
    const tnsr::I<DataVector, Dim, Frame::Logical>& logical_coordinates,
    const ElementMap<Dim, Frame::Inertial>& map,
    const std::unordered_map<Direction<Dim>,
                             tnsr::i<DataVector, Dim, Frame::Inertial>>&
        unnormalized_normals,
    const std::unordered_map<Direction<Dim>,
                             tnsr::i<DataVector, Dim, Frame::Inertial>>&
        boundary_unnormalized_normals,
    const TimeDelta& time_step) const noexcept {
  if (UNLIKELY(disable_for_debugging_)) {
    // Return without flattening
    return false;
  }

  const DataVector det_jacobian =
      get(determinant(map.jacobian(logical_coordinates)));

  const auto add_mass_outflow_rate = [&momentum_density, &mesh, &det_jacobian ](
      const Direction<Dim>& dir,
      const tnsr::i<DataVector, Dim>& unnorm_normal) noexcept {
    const size_t sliced_dim = dir.dimension();
    const size_t index_of_edge =
        (dir.side() == Side::Lower ? 0 : mesh.extents()[sliced_dim] - 1);

    const auto mass_flux_on_edge = data_on_slice(
        *momentum_density, mesh.extents(), sliced_dim, index_of_edge);
    const DataVector mass_outflow_rate_on_edge =
        get(dot_product(mass_flux_on_edge, unnorm_normal));

    return definite_integral(mass_outflow_rate_on_edge,
                             mesh.slice_away(sliced_dim));
  };

  // positive => outflowing mass
  // negative => inflowing mass
  double mass_outflow_rate = 0.;

  for (const auto& dir_and_normal : unnormalized_normals) {
    const auto& dir = dir_and_normal.first;
    const auto& unnorm_normal = dir_and_normal.second;
    mass_outflow_rate += add_mass_outflow_rate(dir, unnorm_normal);
  }

  for (const auto& dir_and_normal : boundary_unnormalized_normals) {
    const auto& dir = dir_and_normal.first;
    const auto& unnorm_normal = dir_and_normal.second;
    mass_outflow_rate += add_mass_outflow_rate(dir, unnorm_normal);
  }

  constexpr double safety_factor = 0.9;
  const double size_of_time_step = time_step.value();
  const double total_mass = mean_value(get(*mass_density), mesh);
  const double max_physical_outflow_rate = total_mass / size_of_time_step;
  const double max_allowed_outflow_rate =
      safety_factor * max_physical_outflow_rate;

  if (mass_outflow_rate > max_allowed_outflow_rate) {
    // Set everything to means
    get(*mass_density) = mean_value(get(*mass_density), mesh);
    for (size_t i = 0; i < Dim; ++i) {
      momentum_density->get(i) = mean_value(momentum_density->get(i), mesh);
    }
    get(*energy_density) = mean_value(get(*energy_density), mesh);
    return true;
  }

  return false;
}

template <size_t LocalDim>
bool operator==(const PostLimiterPpFlux<LocalDim>& lhs,
                const PostLimiterPpFlux<LocalDim>& rhs) noexcept {
  return lhs.disable_for_debugging_ == rhs.disable_for_debugging_;
}

template <size_t Dim>
bool operator!=(const PostLimiterPpFlux<Dim>& lhs,
                const PostLimiterPpFlux<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                \
  template class PostLimiterPpFlux<DIM(data)>;                                \
  template bool operator==(const PostLimiterPpFlux<DIM(data)>& lhs,           \
                           const PostLimiterPpFlux<DIM(data)>& rhs) noexcept; \
  template bool operator!=(const PostLimiterPpFlux<DIM(data)>& lhs,           \
                           const PostLimiterPpFlux<DIM(data)>& rhs) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef DIM
#undef INSTANTIATION

}  // namespace NewtonianEuler
}  // namespace VariableFixing

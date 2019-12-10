// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/PostLimiterFlattening.hpp"

#include <pup.h>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Overloader.hpp"

namespace VariableFixing {
namespace NewtonianEuler {

template <size_t Dim, size_t ThermodynamicDim>
PostLimiterFlattening<Dim, ThermodynamicDim>::PostLimiterFlattening(
    const bool disable_for_debugging)
    : disable_for_debugging_(disable_for_debugging) {}

// clang-tidy: google-runtime-references
template <size_t Dim, size_t ThermodynamicDim>
void PostLimiterFlattening<Dim, ThermodynamicDim>::pup(
    PUP::er& p) noexcept {  // NOLINT
  p | disable_for_debugging_;
}

template <size_t Dim, size_t ThermodynamicDim>
void PostLimiterFlattening<Dim, ThermodynamicDim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> mass_density,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const Mesh<Dim>& mesh,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state) const noexcept {
  if (UNLIKELY(disable_for_debugging_)) {
    // Return without flattening
    return;
  }

  ASSERT(mean_value(get(*mass_density), mesh) > 0., "oops:\n"
      << "rho = " << *mass_density << "\n"
      << "rhou = " << *momentum_density << "\n"
      << "e = " << *energy_density);
  if (ThermodynamicDim == 2) {
    ASSERT(mean_value(get(*energy_density), mesh) >= 0., "oops:\n"
        << "rho = " << *mass_density << "\n"
        << "rhou = " << *momentum_density << "\n"
        << "e = " << *energy_density);
  }

  // If density is negative, then flatten
  if (min(get(*mass_density)) < 0.) {
    get(*mass_density) = mean_value(get(*mass_density), mesh);
    for (size_t i = 0; i < Dim; ++i) {
      momentum_density->get(i) = mean_value(momentum_density->get(i), mesh);
    }
    get(*energy_density) = mean_value(get(*energy_density), mesh);
    return;
  }

  // Check pressure
  Scalar<DataVector> pressure{};
  make_overloader(
      [&pressure, &
       mass_density ](const EquationsOfState::EquationOfState<false, 1>&
                          the_equation_of_state) noexcept {
        pressure = the_equation_of_state.pressure_from_density(*mass_density);
      },
      [&pressure, &mass_density, &momentum_density, &
       energy_density ](const EquationsOfState::EquationOfState<false, 2>&
                            the_equation_of_state) noexcept {
        const auto specific_internal_energy = Scalar<DataVector>{
            get(*energy_density) / get(*mass_density) -
            0.5 * get(dot_product(*momentum_density, *momentum_density)) /
                square(get(*mass_density))};
        pressure = the_equation_of_state.pressure_from_density_and_energy(
            *mass_density, specific_internal_energy);
      })(equation_of_state);

  // If pressure is negative, then flatten
  if (min(get(pressure)) < 0.) {
    get(*mass_density) = mean_value(get(*mass_density), mesh);
    for (size_t i = 0; i < Dim; ++i) {
      momentum_density->get(i) = mean_value(momentum_density->get(i), mesh);
    }
    get(*energy_density) = mean_value(get(*energy_density), mesh);
  }
}

template <size_t LocalDim, size_t LocalThermodynamicDim>
bool operator==(
    const PostLimiterFlattening<LocalDim, LocalThermodynamicDim>& lhs,
    const PostLimiterFlattening<LocalDim, LocalThermodynamicDim>&
        rhs) noexcept {
  return lhs.disable_for_debugging_ == rhs.disable_for_debugging_;
}

template <size_t Dim, size_t ThermodynamicDim>
bool operator!=(
    const PostLimiterFlattening<Dim, ThermodynamicDim>& lhs,
    const PostLimiterFlattening<Dim, ThermodynamicDim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(r, data)                                                 \
  template class PostLimiterFlattening<DIM(data), THERMO_DIM(data)>;           \
  template bool operator==(                                                    \
      const PostLimiterFlattening<DIM(data), THERMO_DIM(data)>& lhs,           \
      const PostLimiterFlattening<DIM(data), THERMO_DIM(data)>& rhs) noexcept; \
  template bool operator!=(                                                    \
      const PostLimiterFlattening<DIM(data), THERMO_DIM(data)>& lhs,           \
      const PostLimiterFlattening<DIM(data), THERMO_DIM(data)>& rhs) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2))

#undef DIM
#undef THERMO_DIM
#undef INSTANTIATION

}  // namespace NewtonianEuler
}  // namespace VariableFixing

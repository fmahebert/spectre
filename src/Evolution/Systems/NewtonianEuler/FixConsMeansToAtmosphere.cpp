// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/FixConsMeansToAtmosphere.hpp"

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
FixConsMeansToAtmosphere<Dim, ThermodynamicDim>::FixConsMeansToAtmosphere(
    const bool disable_for_debugging)
    : disable_for_debugging_(disable_for_debugging) {}

// clang-tidy: google-runtime-references
template <size_t Dim, size_t ThermodynamicDim>
void FixConsMeansToAtmosphere<Dim, ThermodynamicDim>::pup(
    PUP::er& p) noexcept {  // NOLINT
  p | disable_for_debugging_;
}

template <size_t Dim, size_t ThermodynamicDim>
void FixConsMeansToAtmosphere<Dim, ThermodynamicDim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> mass_density,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const Mesh<Dim>& mesh,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state) const noexcept {
  if (UNLIKELY(disable_for_debugging_)) {
    // Return without fixing
    return;
  }

  // TODO get this from input file?
  constexpr double density_floor = 1.e-16;
  const double mean_density = mean_value(get(*mass_density), mesh);
  if (mean_density <= 0.) {
    get(*mass_density) = density_floor;
    for (size_t i = 0; i < Dim; ++i) {
      momentum_density->get(i) = 0.;
    }
    make_overloader(
        [&density_floor, &mass_density, &
         energy_density ](const EquationsOfState::EquationOfState<false, 1>&
                              the_equation_of_state) noexcept {
          get(*energy_density) =
              get(*mass_density) *
              get(the_equation_of_state.specific_internal_energy_from_density(
                  Scalar<double>{density_floor}));
        },
        [&energy_density](const EquationsOfState::EquationOfState<false, 2>&
                          /*the_equation_of_state*/) noexcept {
          get(*energy_density) = 0.;
        })(equation_of_state);
  }
}

template <size_t LocalDim, size_t LocalThermodynamicDim>
bool operator==(
    const FixConsMeansToAtmosphere<LocalDim, LocalThermodynamicDim>& lhs,
    const FixConsMeansToAtmosphere<LocalDim, LocalThermodynamicDim>&
        rhs) noexcept {
  return lhs.disable_for_debugging_ == rhs.disable_for_debugging_;
}

template <size_t Dim, size_t ThermodynamicDim>
bool operator!=(
    const FixConsMeansToAtmosphere<Dim, ThermodynamicDim>& lhs,
    const FixConsMeansToAtmosphere<Dim, ThermodynamicDim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(r, data)                                          \
  template class FixConsMeansToAtmosphere<DIM(data), THERMO_DIM(data)>; \
  template bool operator==(                                             \
      const FixConsMeansToAtmosphere<DIM(data), THERMO_DIM(data)>& lhs, \
      const FixConsMeansToAtmosphere<DIM(data), THERMO_DIM(data)>&      \
          rhs) noexcept;                                                \
  template bool operator!=(                                             \
      const FixConsMeansToAtmosphere<DIM(data), THERMO_DIM(data)>& lhs, \
      const FixConsMeansToAtmosphere<DIM(data), THERMO_DIM(data)>&      \
          rhs) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2))

#undef DIM
#undef THERMO_DIM
#undef INSTANTIATION

}  // namespace NewtonianEuler
}  // namespace VariableFixing

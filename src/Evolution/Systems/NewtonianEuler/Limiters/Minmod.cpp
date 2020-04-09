// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Limiters/Minmod.hpp"

#include <array>
#include <cstdlib>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/OrientationMap.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.tpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodType.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/Flattener.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianEuler {
namespace Limiters {

template <size_t VolumeDim>
Minmod<VolumeDim>::Minmod(const ::Limiters::MinmodType minmod_type,
                          const double tvb_constant, const bool apply_flattener,
                          const bool disable_for_debugging) noexcept
    : minmod_type_(minmod_type),
      tvb_constant_(tvb_constant),
      apply_flattener_(apply_flattener),
      disable_for_debugging_(disable_for_debugging) {
  ASSERT(tvb_constant >= 0.0, "The TVB constant must be non-negative.");
}

template <size_t VolumeDim>
void Minmod<VolumeDim>::pup(PUP::er& p) noexcept {
  p | minmod_type_;
  p | tvb_constant_;
  p | apply_flattener_;
  p | disable_for_debugging_;
}

template <size_t VolumeDim>
void Minmod<VolumeDim>::package_data(
    const gsl::not_null<PackagedData*> packaged_data,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, VolumeDim>& momentum_density,
    const Scalar<DataVector>& energy_density, const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const OrientationMap<VolumeDim>& orientation_map) const noexcept {
  const ConservativeVarsMinmod minmod(minmod_type_, tvb_constant_,
                                      disable_for_debugging_);
  minmod.package_data(packaged_data, mass_density_cons, momentum_density,
                      energy_density, mesh, element_size, orientation_map);
}

template <size_t VolumeDim>
template <size_t ThermodynamicDim>
bool Minmod<VolumeDim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const gsl::not_null<Scalar<DataVector>*> limiter_diagnostics,
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
    const tnsr::I<DataVector, VolumeDim, Frame::Logical>& logical_coords,
    const std::array<double, VolumeDim>& element_size,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data) const noexcept {
  if (UNLIKELY(disable_for_debugging_)) {
    // Do not modify input tensors
    return false;
  }

  // Checks for the post-timestep, pre-limiter NewtonianEuler state, e.g.:
  const double mean_density = mean_value(get(*mass_density_cons), mesh);
  ASSERT(mean_density > 0.0,
         "Positivity was violated on a cell-average level.");
  if (ThermodynamicDim == 2) {
    const double mean_energy = mean_value(get(*energy_density), mesh);
    ASSERT(mean_energy > 0.0,
           "Positivity was violated on a cell-average level.");
  }
  // End pre-limiter checks

  const ConservativeVarsMinmod minmod(minmod_type_, tvb_constant_,
                                      disable_for_debugging_);
  const bool limiter_activated =
      minmod(mass_density_cons, momentum_density, energy_density, mesh, element,
             logical_coords, element_size, neighbor_data);

  size_t flattener_status = 0;
  if (apply_flattener_) {
    flattener_status =
        flatten_solution(mass_density_cons, momentum_density, energy_density,
                         mesh, equation_of_state);
  }

  // Checks for the post-limiter NewtonianEuler state, e.g.:
  ASSERT(min(get(*mass_density_cons)) > 0.0, "Bad density after limiting.");
  if (ThermodynamicDim == 2) {
    Scalar<DataVector> pressure{};
    // Need this make_overloader until we can use `if constexpr`
    make_overloader(
        [&pressure,
         &mass_density_cons](const EquationsOfState::EquationOfState<false, 1>&
                                 the_equation_of_state) noexcept {
          pressure =
              the_equation_of_state.pressure_from_density(*mass_density_cons);
        },
        [&pressure, &mass_density_cons, &momentum_density,
         &energy_density](const EquationsOfState::EquationOfState<false, 2>&
                              the_equation_of_state) noexcept {
          const auto specific_internal_energy = Scalar<DataVector>{
              get(*energy_density) / get(*mass_density_cons) -
              0.5 * get(dot_product(*momentum_density, *momentum_density)) /
                  square(get(*mass_density_cons))};
          pressure = the_equation_of_state.pressure_from_density_and_energy(
              *mass_density_cons, specific_internal_energy);
        })(equation_of_state);
    ASSERT(min(get(pressure)) > 0.0, "Bad pressure after limiting.");
  }
  // End post-limiter checks

  get(*limiter_diagnostics) =
      (limiter_activated ? 1.0 : 0.0) + (flattener_status > 0 ? 1.0 : 0.0);
  return limiter_activated;
}

template <size_t LocalDim>
bool operator==(const Minmod<LocalDim>& lhs,
                const Minmod<LocalDim>& rhs) noexcept {
  return lhs.minmod_type_ == rhs.minmod_type_ and
         lhs.tvb_constant_ == rhs.tvb_constant_ and
         lhs.apply_flattener_ == rhs.apply_flattener_ and
         lhs.disable_for_debugging_ == rhs.disable_for_debugging_;
}

template <size_t VolumeDim>
bool operator!=(const Minmod<VolumeDim>& lhs,
                const Minmod<VolumeDim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data) template class Minmod<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE

#define INSTANTIATE(_, data)                                                   \
  template bool Minmod<DIM(data)>::operator()(                                 \
      const gsl::not_null<Scalar<DataVector>*>,                                \
      const gsl::not_null<tnsr::I<DataVector, DIM(data)>*>,                    \
      const gsl::not_null<Scalar<DataVector>*>,                                \
      const gsl::not_null<Scalar<DataVector>*>, const Mesh<DIM(data)>&,        \
      const Element<DIM(data)>&,                                               \
      const tnsr::I<DataVector, DIM(data), Frame::Logical>&,                   \
      const std::array<double, DIM(data)>&,                                    \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>&,       \
      const std::unordered_map<                                                \
          std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>, PackagedData, \
          boost::hash<                                                         \
              std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>>>&)        \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (1, 2))

#undef INSTANTIATE
#undef DIM
#undef THERMO_DIM

}  // namespace Limiters
}  // namespace NewtonianEuler

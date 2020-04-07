// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Limiters/Weno.hpp"

#include <array>
#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <unordered_map>
#include <utility>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Element.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Weno.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoType.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/Flattener.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/WenoType.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace {
Limiters::WenoType weno_type_from_newtonian_euler_weno_type(
    const NewtonianEuler::Limiters::WenoType in) noexcept {
  switch (in) {
    case NewtonianEuler::Limiters::WenoType::ConservativeHweno:
      return Limiters::WenoType::Hweno;
    case NewtonianEuler::Limiters::WenoType::ConservativeSimpleWeno:
      return Limiters::WenoType::SimpleWeno;
    default:
      // TODO(FH) clean this up... need an uninitialized value maybe?
      ERROR("bad newtonian euler weno type");
  }
}
}  // namespace

namespace NewtonianEuler {
namespace Limiters {

template <size_t VolumeDim>
Weno<VolumeDim>::Weno(const WenoType weno_type,
                      const double neighbor_linear_weight,
                      const double tvb_constant, const bool apply_flattener,
                      const bool disable_for_debugging) noexcept
    : weno_type_(weno_type),
      neighbor_linear_weight_(neighbor_linear_weight),
      tvb_constant_(tvb_constant),
      apply_flattener_(apply_flattener),
      disable_for_debugging_(disable_for_debugging) {
  ASSERT(tvb_constant >= 0.0, "The TVB constant must be non-negative.");
}

template <size_t VolumeDim>
// NOLINTNEXTLINE(google-runtime-references)
void Weno<VolumeDim>::pup(PUP::er& p) noexcept {
  p | weno_type_;
  p | neighbor_linear_weight_;
  p | tvb_constant_;
  p | apply_flattener_;
  p | disable_for_debugging_;
}

template <size_t VolumeDim>
void Weno<VolumeDim>::package_data(
    const gsl::not_null<PackagedData*> packaged_data,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, VolumeDim>& momentum_density,
    const Scalar<DataVector>& energy_density, const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const OrientationMap<VolumeDim>& orientation_map) const noexcept {
  // Convert NewtonianEuler::Limiters::WenoType -> Limiters::WenoType
  const ::Limiters::WenoType generic_weno_type =
      weno_type_from_newtonian_euler_weno_type(weno_type_);
  const ConservativeVarsWeno weno(generic_weno_type, neighbor_linear_weight_,
                                  tvb_constant_, disable_for_debugging_);
  weno.package_data(packaged_data, mass_density_cons, momentum_density,
                    energy_density, mesh, element_size, orientation_map);
}

template <size_t VolumeDim>
template <size_t ThermodynamicDim>
bool Weno<VolumeDim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const gsl::not_null<Scalar<DataVector>*> limiter_diagnostics,
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
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

  // Enforce restrictions on h-refinement, p-refinement
  if (UNLIKELY(alg::any_of(element.neighbors(),
                           [](const auto& direction_neighbors) noexcept {
                             return direction_neighbors.second.size() != 1;
                           }))) {
    ERROR("The Weno limiter does not yet support h-refinement");
    // Removing this limitation will require:
    // - Generalizing the computation of the modified neighbor solutions.
    // - Generalizing the WENO weighted sum for multiple neighbors in each
    //   direction.
  }
  alg::for_each(neighbor_data, [&mesh](const auto& neighbor_and_data) noexcept {
    if (UNLIKELY(neighbor_and_data.second.mesh != mesh)) {
      ERROR("The Weno limiter does not yet support p-refinement");
      // Removing this limitation will require generalizing the
      // computation of the modified neighbor solutions.
    }
  });

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

  // Convert NewtonianEuler::Limiters::WenoType -> Limiters::WenoType
  const ::Limiters::WenoType generic_weno_type =
      weno_type_from_newtonian_euler_weno_type(weno_type_);
  const ConservativeVarsWeno weno(generic_weno_type, neighbor_linear_weight_,
                                  tvb_constant_, disable_for_debugging_);
  const bool limiter_activated =
      weno(mass_density_cons, momentum_density, energy_density, mesh, element,
           element_size, neighbor_data);

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
    ASSERT(min(get(pressure)) > 0.0, "Bad energy after limiting.");
  }
  // End post-limiter checks

  get(*limiter_diagnostics) =
      (limiter_activated ? 1.0 : 0.0) + (flattener_status > 0 ? 1.0 : 0.0);
  return limiter_activated;
}

template <size_t LocalDim>
bool operator==(const Weno<LocalDim>& lhs, const Weno<LocalDim>& rhs) noexcept {
  return lhs.weno_type_ == rhs.weno_type_ and
         lhs.neighbor_linear_weight_ == rhs.neighbor_linear_weight_ and
         lhs.tvb_constant_ == rhs.tvb_constant_ and
         lhs.apply_flattener_ == rhs.apply_flattener_ and
         lhs.disable_for_debugging_ == rhs.disable_for_debugging_;
}

template <size_t VolumeDim>
bool operator!=(const Weno<VolumeDim>& lhs,
                const Weno<VolumeDim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data) template class Weno<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE

#define INSTANTIATE(_, data)                                                   \
  template bool Weno<DIM(data)>::operator()(                                   \
      const gsl::not_null<Scalar<DataVector>*>,                                \
      const gsl::not_null<tnsr::I<DataVector, DIM(data)>*>,                    \
      const gsl::not_null<Scalar<DataVector>*>,                                \
      const gsl::not_null<Scalar<DataVector>*>, const Mesh<DIM(data)>&,        \
      const Element<DIM(data)>&, const std::array<double, DIM(data)>&,         \
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

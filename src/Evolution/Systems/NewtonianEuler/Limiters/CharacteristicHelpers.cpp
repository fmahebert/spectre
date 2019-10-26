// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Limiters/CharacteristicHelpers.hpp"

#include <array>
#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Element.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Systems/NewtonianEuler/Characteristics.hpp"
#include "Evolution/Systems/NewtonianEuler/SoundSpeedSquared.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace NewtonianEuler {
namespace Limiters {

template <size_t VolumeDim, size_t ThermodynamicDim>
std::pair<Matrix, Matrix> compute_eigenvectors(
    const Scalar<double>& mean_density,
    const tnsr::I<double, VolumeDim>& mean_momentum,
    const Scalar<double>& mean_energy,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state,
    const tnsr::i<double, VolumeDim>& unit_normal) noexcept {
  // Prims from means
  const auto velocity = [&mean_density, &mean_momentum]() noexcept {
    auto result = mean_momentum;
    for (size_t i = 0; i < VolumeDim; ++i) {
      result.get(i) /= get(mean_density);
    }
    return result;
  }();
  const auto specific_internal_energy = [&mean_density, &mean_energy,
                                         &mean_momentum]() noexcept {
    auto result = mean_energy;
    get(result) /= get(mean_density);
    get(result) -= 0.5 * get(dot_product(mean_momentum, mean_momentum)) /
                   square(get(mean_density));
    return result;
  }();

  Scalar<double> pressure{};
  Scalar<double> kappa_over_density{};
  make_overloader(
      [&mean_density, &pressure,
       &kappa_over_density](const EquationsOfState::EquationOfState<false, 1>&
                                the_equation_of_state) noexcept {
        pressure = the_equation_of_state.pressure_from_density(mean_density);
        get(kappa_over_density) =
            get(the_equation_of_state
                    .kappa_times_p_over_rho_squared_from_density(
                        mean_density)) *
            get(mean_density) / get(pressure);
      },
      [&mean_density, &specific_internal_energy, &pressure,
       &kappa_over_density](const EquationsOfState::EquationOfState<false, 2>&
                                the_equation_of_state) noexcept {
        pressure = the_equation_of_state.pressure_from_density_and_energy(
            mean_density, specific_internal_energy);
        get(kappa_over_density) =
            get(the_equation_of_state
                    .kappa_times_p_over_rho_squared_from_density_and_energy(
                        mean_density, specific_internal_energy)) *
            get(mean_density) / get(pressure);
      })(equation_of_state);

  const Scalar<double> specific_enthalpy{
      {{(get(mean_energy) + get(pressure)) / get(mean_density)}}};
  const Scalar<double> sound_speed_squared =
      NewtonianEuler::sound_speed_squared(
          mean_density, specific_internal_energy, equation_of_state);

  return std::make_pair(right_eigenvectors<VolumeDim>(
                            velocity, sound_speed_squared, specific_enthalpy,
                            kappa_over_density, unit_normal),
                        left_eigenvectors<VolumeDim>(
                            velocity, sound_speed_squared, specific_enthalpy,
                            kappa_over_density, unit_normal));
}

template <size_t VolumeDim>
void char_means_from_cons_means(
    const gsl::not_null<
        tuples::TaggedTuple<::Tags::Mean<NewtonianEuler::Tags::UMinus>,
                            ::Tags::Mean<NewtonianEuler::Tags::U0<VolumeDim>>,
                            ::Tags::Mean<NewtonianEuler::Tags::UPlus>>*>
        char_means,
    const tuples::TaggedTuple<
        ::Tags::Mean<NewtonianEuler::Tags::MassDensityCons>,
        ::Tags::Mean<NewtonianEuler::Tags::MomentumDensity<VolumeDim>>,
        ::Tags::Mean<NewtonianEuler::Tags::EnergyDensity>>& cons_means,
    const Matrix& left) noexcept {
  auto& char_uminus =
      get<::Tags::Mean<NewtonianEuler::Tags::UMinus>>(*char_means);
  auto& char_u0 =
      get<::Tags::Mean<NewtonianEuler::Tags::U0<VolumeDim>>>(*char_means);
  auto& char_uplus =
      get<::Tags::Mean<NewtonianEuler::Tags::UPlus>>(*char_means);

  const auto& cons_mass_density =
      get<::Tags::Mean<NewtonianEuler::Tags::MassDensityCons>>(cons_means);
  const auto& cons_momentum_density =
      get<::Tags::Mean<NewtonianEuler::Tags::MomentumDensity<VolumeDim>>>(
          cons_means);
  const auto& cons_energy_density =
      get<::Tags::Mean<NewtonianEuler::Tags::EnergyDensity>>(cons_means);

  get(char_uminus) = left(0, 0) * get(cons_mass_density);
  for (size_t j = 0; j < VolumeDim; ++j) {
    char_u0.get(j) = left(j + 1, 0) * get(cons_mass_density);
  }
  get(char_uplus) = left(VolumeDim + 1, 0) * get(cons_mass_density);

  for (size_t i = 0; i < VolumeDim; ++i) {
    get(char_uminus) += left(0, i + 1) * cons_momentum_density.get(i);
    for (size_t j = 0; j < VolumeDim; ++j) {
      char_u0.get(j) += left(j + 1, i + 1) * cons_momentum_density.get(i);
    }
    get(char_uplus) +=
        left(VolumeDim + 1, i + 1) * cons_momentum_density.get(i);
  }

  get(char_uminus) += left(0, VolumeDim + 1) * get(cons_energy_density);
  for (size_t j = 0; j < VolumeDim; ++j) {
    char_u0.get(j) += left(j + 1, VolumeDim + 1) * get(cons_energy_density);
  }
  get(char_uplus) +=
      left(VolumeDim + 1, VolumeDim + 1) * get(cons_energy_density);
}

template <size_t VolumeDim>
void char_tensors_from_cons_tensors(
    const gsl::not_null<Scalar<DataVector>*> char_uminus,
    const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> char_u0,
    const gsl::not_null<Scalar<DataVector>*> char_uplus,
    const Scalar<DataVector>& cons_mass_density,
    const tnsr::I<DataVector, VolumeDim>& cons_momentum_density,
    const Scalar<DataVector>& cons_energy_density,
    const Matrix& left) noexcept {
  get(*char_uminus) = left(0, 0) * get(cons_mass_density);
  for (size_t j = 0; j < VolumeDim; ++j) {
    char_u0->get(j) = left(j + 1, 0) * get(cons_mass_density);
  }
  get(*char_uplus) = left(VolumeDim + 1, 0) * get(cons_mass_density);

  for (size_t i = 0; i < VolumeDim; ++i) {
    get(*char_uminus) += left(0, i + 1) * cons_momentum_density.get(i);
    for (size_t j = 0; j < VolumeDim; ++j) {
      char_u0->get(j) += left(j + 1, i + 1) * cons_momentum_density.get(i);
    }
    get(*char_uplus) +=
        left(VolumeDim + 1, i + 1) * cons_momentum_density.get(i);
  }

  get(*char_uminus) += left(0, VolumeDim + 1) * get(cons_energy_density);
  for (size_t j = 0; j < VolumeDim; ++j) {
    char_u0->get(j) += left(j + 1, VolumeDim + 1) * get(cons_energy_density);
  }
  get(*char_uplus) +=
      left(VolumeDim + 1, VolumeDim + 1) * get(cons_energy_density);
}

template <size_t VolumeDim>
void cons_tensors_from_char_tensors(
    const gsl::not_null<Scalar<DataVector>*> cons_mass_density,
    const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> cons_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> cons_energy_density,
    const Scalar<DataVector>& char_uminus,
    const tnsr::I<DataVector, VolumeDim>& char_u0,
    const Scalar<DataVector>& char_uplus, const Matrix& right) noexcept {
  get(*cons_mass_density) = right(0, 0) * get(char_uminus);
  for (size_t j = 0; j < VolumeDim; ++j) {
    cons_momentum_density->get(j) = right(j + 1, 0) * get(char_uminus);
  }
  get(*cons_energy_density) = right(VolumeDim + 1, 0) * get(char_uminus);

  for (size_t i = 0; i < VolumeDim; ++i) {
    get(*cons_mass_density) += right(0, i + 1) * char_u0.get(i);
    for (size_t j = 0; j < VolumeDim; ++j) {
      cons_momentum_density->get(j) += right(j + 1, i + 1) * char_u0.get(i);
    }
    get(*cons_energy_density) += right(VolumeDim + 1, i + 1) * char_u0.get(i);
  }

  get(*cons_mass_density) += right(0, VolumeDim + 1) * get(char_uplus);
  for (size_t j = 0; j < VolumeDim; ++j) {
    cons_momentum_density->get(j) +=
        right(j + 1, VolumeDim + 1) * get(char_uplus);
  }
  get(*cons_energy_density) +=
      right(VolumeDim + 1, VolumeDim + 1) * get(char_uplus);
}

template <size_t VolumeDim>
void char_vars_from_cons_vars(
    const gsl::not_null<Variables<tmpl::list<
        NewtonianEuler::Tags::UMinus, NewtonianEuler::Tags::U0<VolumeDim>,
        NewtonianEuler::Tags::UPlus>>*>
        char_vars,
    const Variables<tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                               NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                               NewtonianEuler::Tags::EnergyDensity>>& cons_vars,
    const Matrix& left) noexcept {
  char_tensors_from_cons_tensors(
      make_not_null(&get<NewtonianEuler::Tags::UMinus>(*char_vars)),
      make_not_null(&get<NewtonianEuler::Tags::U0<VolumeDim>>(*char_vars)),
      make_not_null(&get<NewtonianEuler::Tags::UPlus>(*char_vars)),
      get<NewtonianEuler::Tags::MassDensityCons>(cons_vars),
      get<NewtonianEuler::Tags::MomentumDensity<VolumeDim>>(cons_vars),
      get<NewtonianEuler::Tags::EnergyDensity>(cons_vars), left);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMODIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                            \
  template std::pair<Matrix, Matrix> compute_eigenvectors(              \
      const Scalar<double>&, const tnsr::I<double, DIM(data)>&,         \
      const Scalar<double>&,                                            \
      const EquationsOfState::EquationOfState<false, THERMODIM(data)>&, \
      const tnsr::i<double, DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (1, 2))

#undef THERMODIM
#undef INSTANTIATE

#define INSTANTIATE(_, data)                                                 \
  template void char_means_from_cons_means(                                  \
      const gsl::not_null<tuples::TaggedTuple<                               \
          ::Tags::Mean<NewtonianEuler::Tags::UMinus>,                        \
          ::Tags::Mean<NewtonianEuler::Tags::U0<DIM(data)>>,                 \
          ::Tags::Mean<NewtonianEuler::Tags::UPlus>>*>,                      \
      const tuples::TaggedTuple<                                             \
          ::Tags::Mean<NewtonianEuler::Tags::MassDensityCons>,               \
          ::Tags::Mean<NewtonianEuler::Tags::MomentumDensity<DIM(data)>>,    \
          ::Tags::Mean<NewtonianEuler::Tags::EnergyDensity>>&,               \
      const Matrix&) noexcept;                                               \
  template void char_tensors_from_cons_tensors(                              \
      const gsl::not_null<Scalar<DataVector>*>,                              \
      const gsl::not_null<tnsr::I<DataVector, DIM(data)>*>,                  \
      const gsl::not_null<Scalar<DataVector>*>, const Scalar<DataVector>&,   \
      const tnsr::I<DataVector, DIM(data)>&, const Scalar<DataVector>&,      \
      const Matrix&) noexcept;                                               \
  template void cons_tensors_from_char_tensors(                              \
      const gsl::not_null<Scalar<DataVector>*>,                              \
      const gsl::not_null<tnsr::I<DataVector, DIM(data)>*>,                  \
      const gsl::not_null<Scalar<DataVector>*>, const Scalar<DataVector>&,   \
      const tnsr::I<DataVector, DIM(data)>&, const Scalar<DataVector>&,      \
      const Matrix&) noexcept;                                               \
  template void char_vars_from_cons_vars(                                    \
      const gsl::not_null<Variables<tmpl::list<                              \
          NewtonianEuler::Tags::UMinus, NewtonianEuler::Tags::U0<DIM(data)>, \
          NewtonianEuler::Tags::UPlus>>*>,                                   \
      const Variables<                                                       \
          tmpl::list<NewtonianEuler::Tags::MassDensityCons,                  \
                     NewtonianEuler::Tags::MomentumDensity<DIM(data)>,       \
                     NewtonianEuler::Tags::EnergyDensity>>&,                 \
      const Matrix&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace Limiters
}  // namespace NewtonianEuler

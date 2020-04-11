// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/NewtonianEuler/Characteristics.hpp"
#include "Evolution/Systems/NewtonianEuler/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianEuler/Fluxes.hpp"
#include "Evolution/Systems/NewtonianEuler/NumericalFluxes/PositivityPreservingLaxFriedrichs.hpp"
#include "Evolution/Systems/NewtonianEuler/SoundSpeedSquared.hpp"
#include "Evolution/Systems/NewtonianEuler/System.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/LocalLaxFriedrichs.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/RiemannProblem.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {

void test_pplf_flux_1d() noexcept {
  const size_t number_of_points = 1;
  const Mesh<0> mesh{};

  tnsr::i<DataVector, 1> normal_int{};
  get<0>(normal_int) = DataVector(number_of_points, 1.);

  tnsr::i<DataVector, 1> normal_ext{};
  get<0>(normal_ext) = DataVector(number_of_points, -1.);

  const EquationsOfState::IdealFluid<false> equation_of_state{5. / 3.};

  // Internal data
  // primitives
  const Scalar<DataVector> density_int(DataVector{{0.1}});
  const tnsr::I<DataVector, 1> velocity_int(DataVector{{-0.1}});
  const Scalar<DataVector> specific_internal_energy_int(DataVector{{1.3}});
  const Scalar<DataVector> pressure_int =
      equation_of_state.pressure_from_density_and_energy(
          density_int, specific_internal_energy_int);
  // conservatives
  Scalar<DataVector> mass_density_int;
  tnsr::I<DataVector, 1> momentum_density_int;
  Scalar<DataVector> energy_density_int;
  NewtonianEuler::ConservativeFromPrimitive<1>::apply(
      make_not_null(&mass_density_int), make_not_null(&momentum_density_int),
      make_not_null(&energy_density_int), density_int, velocity_int,
      specific_internal_energy_int);
  // fluxes
  tnsr::I<DataVector, 1> flux_mass_density_int(number_of_points);
  tnsr::IJ<DataVector, 1> flux_momentum_density_int(number_of_points);
  tnsr::I<DataVector, 1> flux_energy_density_int(number_of_points);
  NewtonianEuler::ComputeFluxes<1>::apply(
      make_not_null(&flux_mass_density_int),
      make_not_null(&flux_momentum_density_int),
      make_not_null(&flux_energy_density_int), momentum_density_int,
      energy_density_int, velocity_int, pressure_int);
  // normal dot fluxes
  const Scalar<DataVector> n_dot_f_mass_density_int =
      dot_product(flux_mass_density_int, normal_int);
  const Scalar<DataVector> n_dot_f_energy_density_int =
      dot_product(flux_energy_density_int, normal_int);
  tnsr::I<DataVector, 1> n_dot_f_momentum_density_int(number_of_points);
  for (size_t i = 0; i < 1; ++i) {
    n_dot_f_momentum_density_int.get(i) = 0.0;
    for (size_t j = 0; j < 1; ++j) {
      n_dot_f_momentum_density_int.get(i) +=
          flux_momentum_density_int.get(i, j) * normal_int.get(j);
    }
  }
  // char speeds
  const DataVector c2_int = get(NewtonianEuler::sound_speed_squared(
      density_int, specific_internal_energy_int, equation_of_state));
  const Scalar<DataVector> sound_speed_int(sqrt(c2_int));
  const std::array<DataVector, 3> char_speeds_int =
      NewtonianEuler::characteristic_speeds(velocity_int, sound_speed_int,
                                            normal_int);
  Scalar<DataVector> max_abs_char_speed_int(number_of_points);
  for (size_t s = 0; s < get(sound_speed_int).size(); ++s) {
    get(max_abs_char_speed_int)[s] = 0.;
    for (size_t i = 0; i < char_speeds_int.size(); ++i) {
      get(max_abs_char_speed_int)[s] =
          std::max(get(max_abs_char_speed_int)[s], char_speeds_int[i][s]);
    }
  }

  // External data
  // primitives
  const Scalar<DataVector> density_ext(DataVector{{1.3}});
  const tnsr::I<DataVector, 1> velocity_ext(DataVector{{-0.2}});
  const Scalar<DataVector> specific_internal_energy_ext(DataVector{{1.4}});
  const Scalar<DataVector> pressure_ext =
      equation_of_state.pressure_from_density_and_energy(
          density_ext, specific_internal_energy_ext);
  // conservatives
  Scalar<DataVector> mass_density_ext;
  tnsr::I<DataVector, 1> momentum_density_ext;
  Scalar<DataVector> energy_density_ext;
  NewtonianEuler::ConservativeFromPrimitive<1>::apply(
      make_not_null(&mass_density_ext), make_not_null(&momentum_density_ext),
      make_not_null(&energy_density_ext), density_ext, velocity_ext,
      specific_internal_energy_ext);
  // fluxes
  tnsr::I<DataVector, 1> flux_mass_density_ext(number_of_points);
  tnsr::IJ<DataVector, 1> flux_momentum_density_ext(number_of_points);
  tnsr::I<DataVector, 1> flux_energy_density_ext(number_of_points);
  NewtonianEuler::ComputeFluxes<1>::apply(
      make_not_null(&flux_mass_density_ext),
      make_not_null(&flux_momentum_density_ext),
      make_not_null(&flux_energy_density_ext), momentum_density_ext,
      energy_density_ext, velocity_ext, pressure_ext);
  // normal dot fluxes (same but opposite normal)
  const Scalar<DataVector> minus_n_dot_f_mass_density_ext =
      dot_product(flux_mass_density_ext, normal_ext);
  const Scalar<DataVector> minus_n_dot_f_energy_density_ext =
      dot_product(flux_energy_density_ext, normal_ext);
  tnsr::I<DataVector, 1> minus_n_dot_f_momentum_density_ext(number_of_points);
  for (size_t i = 0; i < 1; ++i) {
    minus_n_dot_f_momentum_density_ext.get(i) = 0.0;
    for (size_t j = 0; j < 1; ++j) {
      minus_n_dot_f_momentum_density_ext.get(i) +=
          flux_momentum_density_ext.get(i, j) * normal_ext.get(j);
    }
  }
  // char speeds
  const DataVector c2_ext = get(NewtonianEuler::sound_speed_squared(
      density_ext, specific_internal_energy_ext, equation_of_state));
  const Scalar<DataVector> sound_speed_ext(sqrt(c2_ext));
  const std::array<DataVector, 3> char_speeds_ext =
      NewtonianEuler::characteristic_speeds(velocity_ext, sound_speed_ext,
                                            normal_ext);
  Scalar<DataVector> max_abs_char_speed_ext(number_of_points);
  for (size_t s = 0; s < get(sound_speed_ext).size(); ++s) {
    get(max_abs_char_speed_ext)[s] = 0.;
    for (size_t i = 0; i < char_speeds_ext.size(); ++i) {
      get(max_abs_char_speed_ext)[s] =
          std::max(get(max_abs_char_speed_ext)[s], char_speeds_ext[i][s]);
    }
  }

  // Start with very big bounds, so flux matches generic case
  Scalar<DataVector> max_n_dot_f_mass_density_int(DataVector{{1e3}});
  Scalar<DataVector> max_n_dot_f_energy_density_int(DataVector{{1e3}});
  Scalar<DataVector> max_n_dot_f_mass_density_ext(DataVector{{1e3}});
  Scalar<DataVector> max_n_dot_f_energy_density_ext(DataVector{{1e3}});

  // Test that limiter matches the generic LLF flux
  Scalar<DataVector> n_dot_numerical_f_mass_density(number_of_points);
  tnsr::I<DataVector, 1> n_dot_numerical_f_momentum_density(number_of_points);
  Scalar<DataVector> n_dot_numerical_f_energy_density(number_of_points);
  const NewtonianEuler::NumericalFluxes::PositivityPreservingLaxFriedrichs<1>
      pplf{true};
  pplf(make_not_null(&n_dot_numerical_f_mass_density),
       make_not_null(&n_dot_numerical_f_momentum_density),
       make_not_null(&n_dot_numerical_f_energy_density),
       n_dot_f_mass_density_int, n_dot_f_momentum_density_int,
       n_dot_f_energy_density_int, mass_density_int, momentum_density_int,
       energy_density_int, max_abs_char_speed_int, max_n_dot_f_mass_density_int,
       max_n_dot_f_energy_density_int, minus_n_dot_f_mass_density_ext,
       minus_n_dot_f_momentum_density_ext, minus_n_dot_f_energy_density_ext,
       mass_density_ext, momentum_density_ext, energy_density_ext,
       max_abs_char_speed_ext, max_n_dot_f_mass_density_ext,
       max_n_dot_f_energy_density_ext);

  Scalar<DataVector> generic_n_dot_numerical_f_mass_density(number_of_points);
  tnsr::I<DataVector, 1> generic_n_dot_numerical_f_momentum_density(
      number_of_points);
  Scalar<DataVector> generic_n_dot_numerical_f_energy_density(number_of_points);
  using dummy_id = NewtonianEuler::Solutions::RiemannProblem<1>;
  using dummy_eos = typename dummy_id::equation_of_state_type;
  using system = typename NewtonianEuler::System<1, dummy_eos, dummy_id>;
  const dg::NumericalFluxes::LocalLaxFriedrichs<system> llf{};
  llf(make_not_null(&generic_n_dot_numerical_f_mass_density),
      make_not_null(&generic_n_dot_numerical_f_momentum_density),
      make_not_null(&generic_n_dot_numerical_f_energy_density),
      n_dot_f_mass_density_int, n_dot_f_momentum_density_int,
      n_dot_f_energy_density_int, mass_density_int, momentum_density_int,
      energy_density_int, max_abs_char_speed_int,
      minus_n_dot_f_mass_density_ext, minus_n_dot_f_momentum_density_ext,
      minus_n_dot_f_energy_density_ext, mass_density_ext, momentum_density_ext,
      energy_density_ext, max_abs_char_speed_ext);

  CHECK_ITERABLE_APPROX(n_dot_numerical_f_mass_density,
                        generic_n_dot_numerical_f_mass_density);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_momentum_density,
                        generic_n_dot_numerical_f_momentum_density);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_energy_density,
                        generic_n_dot_numerical_f_energy_density);

  // Now impose bounds on fluxes
  // 1) interior side constraints mass flux - should not change outcome because
  //    velocity (=> flux) is right-to-left
  get(max_n_dot_f_mass_density_int) = DataVector{{1e-3}};
  pplf(make_not_null(&n_dot_numerical_f_mass_density),
       make_not_null(&n_dot_numerical_f_momentum_density),
       make_not_null(&n_dot_numerical_f_energy_density),
       n_dot_f_mass_density_int, n_dot_f_momentum_density_int,
       n_dot_f_energy_density_int, mass_density_int, momentum_density_int,
       energy_density_int, max_abs_char_speed_int, max_n_dot_f_mass_density_int,
       max_n_dot_f_energy_density_int, minus_n_dot_f_mass_density_ext,
       minus_n_dot_f_momentum_density_ext, minus_n_dot_f_energy_density_ext,
       mass_density_ext, momentum_density_ext, energy_density_ext,
       max_abs_char_speed_ext, max_n_dot_f_mass_density_ext,
       max_n_dot_f_energy_density_ext);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_mass_density,
                        generic_n_dot_numerical_f_mass_density);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_momentum_density,
                        generic_n_dot_numerical_f_momentum_density);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_energy_density,
                        generic_n_dot_numerical_f_energy_density);

  // 2) exterior side constrains mass flux - change
  get(max_n_dot_f_mass_density_int) = DataVector{{1e3}};
  get(max_n_dot_f_mass_density_ext) = DataVector{{0.5}};
  pplf(make_not_null(&n_dot_numerical_f_mass_density),
       make_not_null(&n_dot_numerical_f_momentum_density),
       make_not_null(&n_dot_numerical_f_energy_density),
       n_dot_f_mass_density_int, n_dot_f_momentum_density_int,
       n_dot_f_energy_density_int, mass_density_int, momentum_density_int,
       energy_density_int, max_abs_char_speed_int, max_n_dot_f_mass_density_int,
       max_n_dot_f_energy_density_int, minus_n_dot_f_mass_density_ext,
       minus_n_dot_f_momentum_density_ext, minus_n_dot_f_energy_density_ext,
       mass_density_ext, momentum_density_ext, energy_density_ext,
       max_abs_char_speed_ext, max_n_dot_f_mass_density_ext,
       max_n_dot_f_energy_density_ext);
  auto n_dot_f_mass_density_ext = minus_n_dot_f_mass_density_ext;
  get(n_dot_f_mass_density_ext) *= -1.0;
  auto n_dot_f_momentum_density_ext = minus_n_dot_f_momentum_density_ext;
  get<0>(n_dot_f_momentum_density_ext) *= -1.0;
  auto n_dot_f_energy_density_ext = minus_n_dot_f_energy_density_ext;
  get(n_dot_f_energy_density_ext) *= -1.0;
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_mass_density,
                        n_dot_f_mass_density_ext);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_momentum_density,
                        n_dot_f_momentum_density_ext);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_energy_density,
                        n_dot_f_energy_density_ext);

  // Test conservation: F(L,R) = -F(R,L) (b/c of our normal sign conventions)
  Scalar<DataVector> cons_n_dot_numerical_f_mass_density(number_of_points);
  tnsr::I<DataVector, 1> cons_n_dot_numerical_f_momentum_density(
      number_of_points);
  Scalar<DataVector> cons_n_dot_numerical_f_energy_density(number_of_points);
  pplf(make_not_null(&cons_n_dot_numerical_f_mass_density),
       make_not_null(&cons_n_dot_numerical_f_momentum_density),
       make_not_null(&cons_n_dot_numerical_f_energy_density),
       minus_n_dot_f_mass_density_ext, minus_n_dot_f_momentum_density_ext,
       minus_n_dot_f_energy_density_ext, mass_density_ext, momentum_density_ext,
       energy_density_ext, max_abs_char_speed_ext, max_n_dot_f_mass_density_ext,
       max_n_dot_f_energy_density_ext, n_dot_f_mass_density_int,
       n_dot_f_momentum_density_int, n_dot_f_energy_density_int,
       mass_density_int, momentum_density_int, energy_density_int,
       max_abs_char_speed_int, max_n_dot_f_mass_density_int,
       max_n_dot_f_energy_density_int);
  get(cons_n_dot_numerical_f_mass_density) *= -1.0;
  get<0>(cons_n_dot_numerical_f_momentum_density) *= -1.0;
  get(cons_n_dot_numerical_f_energy_density) *= -1.0;
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_mass_density,
                        cons_n_dot_numerical_f_mass_density);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_momentum_density,
                        cons_n_dot_numerical_f_momentum_density);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_energy_density,
                        cons_n_dot_numerical_f_energy_density);

  // Test consistency F(L,-L) = F(L) (where -L in 2nd arg is from sign flip)
  auto minus_n_dot_f_mass_density_int = n_dot_f_mass_density_int;
  get(minus_n_dot_f_mass_density_int) *= -1.0;
  auto minus_n_dot_f_momentum_density_int = n_dot_f_momentum_density_int;
  get<0>(minus_n_dot_f_momentum_density_int) *= -1.0;
  auto minus_n_dot_f_energy_density_int = n_dot_f_energy_density_int;
  get(minus_n_dot_f_energy_density_int) *= -1.0;
  pplf(make_not_null(&n_dot_numerical_f_mass_density),
       make_not_null(&n_dot_numerical_f_momentum_density),
       make_not_null(&n_dot_numerical_f_energy_density),
       n_dot_f_mass_density_int, n_dot_f_momentum_density_int,
       n_dot_f_energy_density_int, mass_density_int, momentum_density_int,
       energy_density_int, max_abs_char_speed_int, max_n_dot_f_mass_density_int,
       max_n_dot_f_energy_density_int, minus_n_dot_f_mass_density_int,
       minus_n_dot_f_momentum_density_int, minus_n_dot_f_energy_density_int,
       mass_density_int, momentum_density_int, energy_density_int,
       max_abs_char_speed_int, max_n_dot_f_mass_density_int,
       max_n_dot_f_energy_density_int);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_mass_density,
                        n_dot_f_mass_density_int);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_momentum_density,
                        n_dot_f_momentum_density_int);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_energy_density,
                        n_dot_f_energy_density_int);
}

void test_pplf_flux_2d() noexcept {
  const size_t number_of_points = 3;
  const Mesh<1> mesh(number_of_points, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);

  tnsr::i<DataVector, 2> normal_int{};
  get<0>(normal_int) = DataVector(number_of_points, 0.);
  get<1>(normal_int) = DataVector(number_of_points, -1.);

  tnsr::i<DataVector, 2> normal_ext{};
  get<0>(normal_ext) = DataVector(number_of_points, 0.);
  get<1>(normal_ext) = DataVector(number_of_points, 1.);

  const EquationsOfState::IdealFluid<false> equation_of_state{5. / 3.};

  // Internal data
  // primitives
  const Scalar<DataVector> density_int(DataVector{{0.1, 0.2, 0.3}});
  const tnsr::I<DataVector, 2> velocity_int(DataVector{{0.01, -0.01, -0.1}});
  const Scalar<DataVector> specific_internal_energy_int(
      DataVector{{1.1, 1.2, 1.3}});
  const Scalar<DataVector> pressure_int =
      equation_of_state.pressure_from_density_and_energy(
          density_int, specific_internal_energy_int);
  // conservatives
  Scalar<DataVector> mass_density_int;
  tnsr::I<DataVector, 2> momentum_density_int;
  Scalar<DataVector> energy_density_int;
  NewtonianEuler::ConservativeFromPrimitive<2>::apply(
      make_not_null(&mass_density_int), make_not_null(&momentum_density_int),
      make_not_null(&energy_density_int), density_int, velocity_int,
      specific_internal_energy_int);
  // fluxes
  tnsr::I<DataVector, 2> flux_mass_density_int(number_of_points);
  tnsr::IJ<DataVector, 2> flux_momentum_density_int(number_of_points);
  tnsr::I<DataVector, 2> flux_energy_density_int(number_of_points);
  NewtonianEuler::ComputeFluxes<2>::apply(
      make_not_null(&flux_mass_density_int),
      make_not_null(&flux_momentum_density_int),
      make_not_null(&flux_energy_density_int), momentum_density_int,
      energy_density_int, velocity_int, pressure_int);
  // normal dot fluxes
  const Scalar<DataVector> n_dot_f_mass_density_int =
      dot_product(flux_mass_density_int, normal_int);
  const Scalar<DataVector> n_dot_f_energy_density_int =
      dot_product(flux_energy_density_int, normal_int);
  tnsr::I<DataVector, 2> n_dot_f_momentum_density_int(number_of_points);
  for (size_t i = 0; i < 2; ++i) {
    n_dot_f_momentum_density_int.get(i) = 0.0;
    for (size_t j = 0; j < 2; ++j) {
      n_dot_f_momentum_density_int.get(i) +=
          flux_momentum_density_int.get(i, j) * normal_int.get(j);
    }
  }
  // char speeds
  const DataVector c2_int = get(NewtonianEuler::sound_speed_squared(
      density_int, specific_internal_energy_int, equation_of_state));
  const Scalar<DataVector> sound_speed_int(sqrt(c2_int));
  const std::array<DataVector, 4> char_speeds_int =
      NewtonianEuler::characteristic_speeds(velocity_int, sound_speed_int,
                                            normal_int);
  Scalar<DataVector> max_abs_char_speed_int(number_of_points);
  for (size_t s = 0; s < get(sound_speed_int).size(); ++s) {
    get(max_abs_char_speed_int)[s] = 0.;
    for (size_t i = 0; i < char_speeds_int.size(); ++i) {
      get(max_abs_char_speed_int)[s] =
          std::max(get(max_abs_char_speed_int)[s], char_speeds_int[i][s]);
    }
  }

  // External data
  // primitives
  const Scalar<DataVector> density_ext(DataVector{{0.3, 0.2, 0.1}});
  const tnsr::I<DataVector, 2> velocity_ext(DataVector{{-0.2, -0.1, 0.2}});
  const Scalar<DataVector> specific_internal_energy_ext(
      DataVector{{1.4, 1.5, 1.6}});
  const Scalar<DataVector> pressure_ext =
      equation_of_state.pressure_from_density_and_energy(
          density_ext, specific_internal_energy_ext);
  // conservatives
  Scalar<DataVector> mass_density_ext;
  tnsr::I<DataVector, 2> momentum_density_ext;
  Scalar<DataVector> energy_density_ext;
  NewtonianEuler::ConservativeFromPrimitive<2>::apply(
      make_not_null(&mass_density_ext), make_not_null(&momentum_density_ext),
      make_not_null(&energy_density_ext), density_ext, velocity_ext,
      specific_internal_energy_ext);
  // fluxes
  tnsr::I<DataVector, 2> flux_mass_density_ext(number_of_points);
  tnsr::IJ<DataVector, 2> flux_momentum_density_ext(number_of_points);
  tnsr::I<DataVector, 2> flux_energy_density_ext(number_of_points);
  NewtonianEuler::ComputeFluxes<2>::apply(
      make_not_null(&flux_mass_density_ext),
      make_not_null(&flux_momentum_density_ext),
      make_not_null(&flux_energy_density_ext), momentum_density_ext,
      energy_density_ext, velocity_ext, pressure_ext);
  // normal dot fluxes (same but opposite normal)
  const Scalar<DataVector> minus_n_dot_f_mass_density_ext =
      dot_product(flux_mass_density_ext, normal_ext);
  const Scalar<DataVector> minus_n_dot_f_energy_density_ext =
      dot_product(flux_energy_density_ext, normal_ext);
  tnsr::I<DataVector, 2> minus_n_dot_f_momentum_density_ext(number_of_points);
  for (size_t i = 0; i < 2; ++i) {
    minus_n_dot_f_momentum_density_ext.get(i) = 0.0;
    for (size_t j = 0; j < 2; ++j) {
      minus_n_dot_f_momentum_density_ext.get(i) +=
          flux_momentum_density_ext.get(i, j) * normal_ext.get(j);
    }
  }
  // char speeds
  const DataVector c2_ext = get(NewtonianEuler::sound_speed_squared(
      density_ext, specific_internal_energy_ext, equation_of_state));
  const Scalar<DataVector> sound_speed_ext(sqrt(c2_ext));
  const std::array<DataVector, 4> char_speeds_ext =
      NewtonianEuler::characteristic_speeds(velocity_ext, sound_speed_ext,
                                            normal_ext);
  Scalar<DataVector> max_abs_char_speed_ext(number_of_points);
  for (size_t s = 0; s < get(sound_speed_ext).size(); ++s) {
    get(max_abs_char_speed_ext)[s] = 0.;
    for (size_t i = 0; i < char_speeds_ext.size(); ++i) {
      get(max_abs_char_speed_ext)[s] =
          std::max(get(max_abs_char_speed_ext)[s], char_speeds_ext[i][s]);
    }
  }

  // Start with very big bounds, so flux matches generic case
  Scalar<DataVector> max_n_dot_f_mass_density_int(DataVector{{1e3}});
  Scalar<DataVector> max_n_dot_f_energy_density_int(DataVector{{1e3}});
  Scalar<DataVector> max_n_dot_f_mass_density_ext(DataVector{{1e3}});
  Scalar<DataVector> max_n_dot_f_energy_density_ext(DataVector{{1e3}});

  // Test that limiter matches the generic LLF flux
  Scalar<DataVector> n_dot_numerical_f_mass_density(number_of_points);
  tnsr::I<DataVector, 2> n_dot_numerical_f_momentum_density(number_of_points);
  Scalar<DataVector> n_dot_numerical_f_energy_density(number_of_points);
  const NewtonianEuler::NumericalFluxes::PositivityPreservingLaxFriedrichs<2>
      pplf{true};
  pplf(make_not_null(&n_dot_numerical_f_mass_density),
       make_not_null(&n_dot_numerical_f_momentum_density),
       make_not_null(&n_dot_numerical_f_energy_density),
       n_dot_f_mass_density_int, n_dot_f_momentum_density_int,
       n_dot_f_energy_density_int, mass_density_int, momentum_density_int,
       energy_density_int, max_abs_char_speed_int, max_n_dot_f_mass_density_int,
       max_n_dot_f_energy_density_int, minus_n_dot_f_mass_density_ext,
       minus_n_dot_f_momentum_density_ext, minus_n_dot_f_energy_density_ext,
       mass_density_ext, momentum_density_ext, energy_density_ext,
       max_abs_char_speed_ext, max_n_dot_f_mass_density_ext,
       max_n_dot_f_energy_density_ext);

  Scalar<DataVector> generic_n_dot_numerical_f_mass_density(number_of_points);
  tnsr::I<DataVector, 2> generic_n_dot_numerical_f_momentum_density(
      number_of_points);
  Scalar<DataVector> generic_n_dot_numerical_f_energy_density(number_of_points);
  using dummy_id = NewtonianEuler::Solutions::RiemannProblem<2>;
  using dummy_eos = typename dummy_id::equation_of_state_type;
  using system = typename NewtonianEuler::System<2, dummy_eos, dummy_id>;
  const dg::NumericalFluxes::LocalLaxFriedrichs<system> llf{};
  llf(make_not_null(&generic_n_dot_numerical_f_mass_density),
      make_not_null(&generic_n_dot_numerical_f_momentum_density),
      make_not_null(&generic_n_dot_numerical_f_energy_density),
      n_dot_f_mass_density_int, n_dot_f_momentum_density_int,
      n_dot_f_energy_density_int, mass_density_int, momentum_density_int,
      energy_density_int, max_abs_char_speed_int,
      minus_n_dot_f_mass_density_ext, minus_n_dot_f_momentum_density_ext,
      minus_n_dot_f_energy_density_ext, mass_density_ext, momentum_density_ext,
      energy_density_ext, max_abs_char_speed_ext);

  CHECK_ITERABLE_APPROX(n_dot_numerical_f_mass_density,
                        generic_n_dot_numerical_f_mass_density);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_momentum_density,
                        generic_n_dot_numerical_f_momentum_density);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_energy_density,
                        generic_n_dot_numerical_f_energy_density);

  // Now impose bounds on fluxes
  // 1) interior side constraints mass flux - change
  get(max_n_dot_f_mass_density_int) = DataVector{{0.01}};
  pplf(make_not_null(&n_dot_numerical_f_mass_density),
       make_not_null(&n_dot_numerical_f_momentum_density),
       make_not_null(&n_dot_numerical_f_energy_density),
       n_dot_f_mass_density_int, n_dot_f_momentum_density_int,
       n_dot_f_energy_density_int, mass_density_int, momentum_density_int,
       energy_density_int, max_abs_char_speed_int, max_n_dot_f_mass_density_int,
       max_n_dot_f_energy_density_int, minus_n_dot_f_mass_density_ext,
       minus_n_dot_f_momentum_density_ext, minus_n_dot_f_energy_density_ext,
       mass_density_ext, momentum_density_ext, energy_density_ext,
       max_abs_char_speed_ext, max_n_dot_f_mass_density_ext,
       max_n_dot_f_energy_density_ext);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_mass_density,
                        n_dot_f_mass_density_int);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_momentum_density,
                        n_dot_f_momentum_density_int);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_energy_density,
                        n_dot_f_energy_density_int);

  // 2) exterior side constrains mass flux - no change
  get(max_n_dot_f_mass_density_int) = DataVector{{1e3}};
  get(max_n_dot_f_mass_density_ext) = DataVector{{1e-3}};
  pplf(make_not_null(&n_dot_numerical_f_mass_density),
       make_not_null(&n_dot_numerical_f_momentum_density),
       make_not_null(&n_dot_numerical_f_energy_density),
       n_dot_f_mass_density_int, n_dot_f_momentum_density_int,
       n_dot_f_energy_density_int, mass_density_int, momentum_density_int,
       energy_density_int, max_abs_char_speed_int, max_n_dot_f_mass_density_int,
       max_n_dot_f_energy_density_int, minus_n_dot_f_mass_density_ext,
       minus_n_dot_f_momentum_density_ext, minus_n_dot_f_energy_density_ext,
       mass_density_ext, momentum_density_ext, energy_density_ext,
       max_abs_char_speed_ext, max_n_dot_f_mass_density_ext,
       max_n_dot_f_energy_density_ext);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_mass_density,
                        generic_n_dot_numerical_f_mass_density);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_momentum_density,
                        generic_n_dot_numerical_f_momentum_density);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_energy_density,
                        generic_n_dot_numerical_f_energy_density);

  // Test conservation: F(L,R) = -F(R,L) (b/c of our normal sign conventions)
  Scalar<DataVector> cons_n_dot_numerical_f_mass_density(number_of_points);
  tnsr::I<DataVector, 2> cons_n_dot_numerical_f_momentum_density(
      number_of_points);
  Scalar<DataVector> cons_n_dot_numerical_f_energy_density(number_of_points);
  pplf(make_not_null(&cons_n_dot_numerical_f_mass_density),
       make_not_null(&cons_n_dot_numerical_f_momentum_density),
       make_not_null(&cons_n_dot_numerical_f_energy_density),
       minus_n_dot_f_mass_density_ext, minus_n_dot_f_momentum_density_ext,
       minus_n_dot_f_energy_density_ext, mass_density_ext, momentum_density_ext,
       energy_density_ext, max_abs_char_speed_ext, max_n_dot_f_mass_density_ext,
       max_n_dot_f_energy_density_ext, n_dot_f_mass_density_int,
       n_dot_f_momentum_density_int, n_dot_f_energy_density_int,
       mass_density_int, momentum_density_int, energy_density_int,
       max_abs_char_speed_int, max_n_dot_f_mass_density_int,
       max_n_dot_f_energy_density_int);
  get(cons_n_dot_numerical_f_mass_density) *= -1.0;
  get<0>(cons_n_dot_numerical_f_momentum_density) *= -1.0;
  get<1>(cons_n_dot_numerical_f_momentum_density) *= -1.0;
  get(cons_n_dot_numerical_f_energy_density) *= -1.0;
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_mass_density,
                        cons_n_dot_numerical_f_mass_density);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_momentum_density,
                        cons_n_dot_numerical_f_momentum_density);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_energy_density,
                        cons_n_dot_numerical_f_energy_density);

  // Test consistency F(L,-L) = F(L) (where -L in 2nd arg is from sign flip)
  auto minus_n_dot_f_mass_density_int = n_dot_f_mass_density_int;
  get(minus_n_dot_f_mass_density_int) *= -1.0;
  auto minus_n_dot_f_momentum_density_int = n_dot_f_momentum_density_int;
  get<0>(minus_n_dot_f_momentum_density_int) *= -1.0;
  get<1>(minus_n_dot_f_momentum_density_int) *= -1.0;
  auto minus_n_dot_f_energy_density_int = n_dot_f_energy_density_int;
  get(minus_n_dot_f_energy_density_int) *= -1.0;
  pplf(make_not_null(&n_dot_numerical_f_mass_density),
       make_not_null(&n_dot_numerical_f_momentum_density),
       make_not_null(&n_dot_numerical_f_energy_density),
       n_dot_f_mass_density_int, n_dot_f_momentum_density_int,
       n_dot_f_energy_density_int, mass_density_int, momentum_density_int,
       energy_density_int, max_abs_char_speed_int, max_n_dot_f_mass_density_int,
       max_n_dot_f_energy_density_int, minus_n_dot_f_mass_density_int,
       minus_n_dot_f_momentum_density_int, minus_n_dot_f_energy_density_int,
       mass_density_int, momentum_density_int, energy_density_int,
       max_abs_char_speed_int, max_n_dot_f_mass_density_int,
       max_n_dot_f_energy_density_int);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_mass_density,
                        n_dot_f_mass_density_int);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_momentum_density,
                        n_dot_f_momentum_density_int);
  CHECK_ITERABLE_APPROX(n_dot_numerical_f_energy_density,
                        n_dot_f_energy_density_int);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.NumericalFluxes.PosPreservingLf",
    "[Unit][Evolution]") {
  test_pplf_flux_1d();
  test_pplf_flux_2d();
}

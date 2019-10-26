// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/CharacteristicHelpers.hpp"
#include "Evolution/Systems/NewtonianEuler/SoundSpeedSquared.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Limiters/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"

namespace {

template <size_t Dim>
void test_char_transformation_helpers() noexcept {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-1., 1.);
  std::uniform_real_distribution<> distribution_positive(1e-3, 1.);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);
  const auto nn_distribution_positive = make_not_null(&distribution_positive);

  // This computes a unit normal. It is NOT uniformly distributed in angle,
  // but for this test the angular distribution is not important.
  const auto unit_normal = [&]() noexcept {
    const double double_used_for_size = 0.;
    auto result = make_with_random_values<tnsr::i<double, Dim>>(
        nn_generator, nn_distribution, double_used_for_size);
    double normal_magnitude = get(magnitude(result));
    // Though highly unlikely, the normal could have length nearly 0. If this
    // happens, we edit the normal to make it non-zero.
    if (normal_magnitude < 1e-3) {
      get<0>(result) += 0.9;
      normal_magnitude = get(magnitude(result));
    }
    for (auto& n_i : result) {
      n_i /= normal_magnitude;
    }
    return result;
  }();

  // Test grid has 3 points per dim
  const Mesh<Dim> mesh(3, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
  const DataVector used_for_size(pow<Dim>(3), 0.);

  // Derive everything from primitive variables
  const auto density = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution_positive, used_for_size);
  const auto velocity = make_with_random_values<tnsr::I<DataVector, Dim>>(
      nn_generator, nn_distribution, used_for_size);
  const auto specific_internal_energy =
      make_with_random_values<Scalar<DataVector>>(
          nn_generator, nn_distribution_positive, used_for_size);

  const EquationsOfState::IdealFluid<false> equation_of_state{5. / 3.};
  const auto pressure = equation_of_state.pressure_from_density_and_energy(
      density, specific_internal_energy);

  const Scalar<DataVector>& mass_density = density;
  const tnsr::I<DataVector, Dim> momentum_density = [&density,
                                                     &velocity]() noexcept {
    auto result = velocity;
    for (size_t i = 0; i < Dim; ++i) {
      result.get(i) *= get(density);
    }
    return result;
  }();
  const Scalar<DataVector> energy_density{
      get(density) * (get(specific_internal_energy) +
                      0.5 * get(dot_product(velocity, velocity)))};

  const Scalar<double> mean_mass_density{mean_value(get(mass_density), mesh)};
  const tnsr::I<double, Dim> mean_momentum_density = [&momentum_density,
                                                      &mesh]() noexcept {
    tnsr::I<double, Dim> result{};
    for (size_t i = 0; i < Dim; ++i) {
      result.get(i) = mean_value(momentum_density.get(i), mesh);
    }
    return result;
  }();
  const Scalar<double> mean_energy_density{
      mean_value(get(energy_density), mesh)};

  const auto right_and_left =
      NewtonianEuler::Limiters::compute_eigenvectors<Dim>(
          mean_mass_density, mean_momentum_density, mean_energy_density,
          equation_of_state, unit_normal);
  const auto& right = right_and_left.first;
  const auto& left = right_and_left.second;

  // test the wrapper functions
  Scalar<DataVector> u_minus{};
  tnsr::I<DataVector, Dim> u0{};
  Scalar<DataVector> u_plus{};
  NewtonianEuler::Limiters::char_tensors_from_cons_tensors(
      make_not_null(&u_minus), make_not_null(&u0), make_not_null(&u_plus),
      mass_density, momentum_density, energy_density, left);
  Scalar<DataVector> recovered_mass_density{};
  tnsr::I<DataVector, Dim> recovered_momentum_density{};
  Scalar<DataVector> recovered_energy_density{};
  NewtonianEuler::Limiters::cons_tensors_from_char_tensors(
      make_not_null(&recovered_mass_density),
      make_not_null(&recovered_momentum_density),
      make_not_null(&recovered_energy_density), u_minus, u0, u_plus, right);
  CHECK_ITERABLE_APPROX(mass_density, recovered_mass_density);
  CHECK_ITERABLE_APPROX(momentum_density, recovered_momentum_density);
  CHECK_ITERABLE_APPROX(energy_density, recovered_energy_density);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.Limiters.CharacteristicHelpers",
    "[Unit][Evolution]") {
  test_char_transformation_helpers<1>();
  test_char_transformation_helpers<2>();
  test_char_transformation_helpers<3>();
}

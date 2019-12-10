// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/Flattener.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"

namespace {

void test_flattener_1d() noexcept {
  const Mesh<1> mesh(4, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const EquationsOfState::IdealFluid<false> equation_of_state{5. / 3.};

  // First, test a case with no flattening needed
  {
    Scalar<DataVector> mass_density_cons(DataVector{{0.8, 1.6, 2.2, 1.2}});
    tnsr::I<DataVector, 1> momentum_density(DataVector{{-1., 0., 0., -2.}});
    Scalar<DataVector> energy_density(DataVector{{1.0, 1.6, 1.9, 1.9}});

    const auto expected_mass_density_cons = mass_density_cons;
    const auto expected_momentum_density = momentum_density;
    const auto expected_energy_density = energy_density;

    size_t flattener_status = NewtonianEuler::Limiters::flatten_solution(
        make_not_null(&mass_density_cons), make_not_null(&momentum_density),
        make_not_null(&energy_density), mesh, equation_of_state);

    CHECK(flattener_status == 0);
    CHECK_ITERABLE_APPROX(mass_density_cons, expected_mass_density_cons);
    CHECK_ITERABLE_APPROX(momentum_density, expected_momentum_density);
    CHECK_ITERABLE_APPROX(energy_density, expected_energy_density);
  }

  // Second, test a case where a negative density leads to a small flattening
  {
    Scalar<DataVector> mass_density_cons(DataVector{{0.8, -0.25, 2.25, 1.2}});
    tnsr::I<DataVector, 1> momentum_density(DataVector{{-1., 0., 0., -2.}});
    Scalar<DataVector> energy_density(DataVector{{1.0, 1.6, 1.9, 1.9}});

    // Expect flattening factor (about mean) of 0.8 times the 0.95 safety = 0.76
    // density mean = 1.0
    // momentum mean = -0.25
    // energy mean = 1.7
    const Scalar<DataVector> expected_mass_density_cons(
        DataVector{{0.848, 0.05, 1.95, 1.152}});
    const tnsr::I<DataVector, 1> expected_momentum_density(
        DataVector{{-0.82, -0.06, -0.06, -1.58}});
    const Scalar<DataVector> expected_energy_density(
        DataVector{{1.168, 1.624, 1.852, 1.852}});

    size_t flattener_status = NewtonianEuler::Limiters::flatten_solution(
        make_not_null(&mass_density_cons), make_not_null(&momentum_density),
        make_not_null(&energy_density), mesh, equation_of_state);

    CHECK(flattener_status == 1);
    CHECK_ITERABLE_APPROX(mass_density_cons, expected_mass_density_cons);
    CHECK_ITERABLE_APPROX(momentum_density, expected_momentum_density);
    CHECK_ITERABLE_APPROX(energy_density, expected_energy_density);
  }

  // Finally, test a case where a negative pressure leads to constant output
  {
    Scalar<DataVector> mass_density_cons(DataVector{{0.8, -0.25, 2.25, 1.2}});
    tnsr::I<DataVector, 1> momentum_density(DataVector{{-1., 0., 0., -2.}});
    Scalar<DataVector> energy_density(DataVector{{1.0, 1.6, 1.9, 0.7}});

    // density mean = 1.0
    // momentum mean = -0.25
    // energy mean = 1.6
    const auto expected_mass_density_cons =
        make_with_value<Scalar<DataVector>>(mass_density_cons, 1.);
    const auto expected_momentum_density =
        make_with_value<tnsr::I<DataVector, 1>>(momentum_density, -0.25);
    const auto expected_energy_density =
        make_with_value<Scalar<DataVector>>(energy_density, 1.6);

    size_t flattener_status = NewtonianEuler::Limiters::flatten_solution(
        make_not_null(&mass_density_cons), make_not_null(&momentum_density),
        make_not_null(&energy_density), mesh, equation_of_state);

    CHECK(flattener_status == 2);
    CHECK_ITERABLE_APPROX(mass_density_cons, expected_mass_density_cons);
    CHECK_ITERABLE_APPROX(momentum_density, expected_momentum_density);
    CHECK_ITERABLE_APPROX(energy_density, expected_energy_density);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Limiters.Flattener",
                  "[Unit][Evolution]") {
  test_flattener_1d();
}

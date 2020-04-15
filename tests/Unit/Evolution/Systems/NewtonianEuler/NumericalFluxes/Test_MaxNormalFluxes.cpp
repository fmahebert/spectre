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
#include "Evolution/Systems/NewtonianEuler/NumericalFluxes/MaxNormalFluxes.hpp"
#include "Evolution/Systems/NewtonianEuler/NumericalFluxes/PositivityPreservingLaxFriedrichs.hpp"
#include "Evolution/Systems/NewtonianEuler/SoundSpeedSquared.hpp"
#include "Evolution/Systems/NewtonianEuler/System.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Limiters/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/LocalLaxFriedrichs.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/RiemannProblem.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {

void test_max_normal_fluxes_1d() noexcept {
  const Mesh<1> mesh(3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const size_t number_of_points = mesh.number_of_grid_points();
  const std::array<double, 1> element_size = {{0.4}};
  const TimeDelta dt(Slab(0., 1.), Rational(1, 10));
  const auto element = TestHelpers::Limiters::make_element<1>();

  const Scalar<DataVector> scalar(DataVector{{1.2, 1.4, 1.6}});
  const Scalar<DataVector> expected_max_n_dot_f(
      DataVector(number_of_points, 2.8));

  Scalar<DataVector> max_n_dot_f;
  NewtonianEuler::max_n_dot_f_of_positive_scalar(
      make_not_null(&max_n_dot_f), scalar, mesh, element, element_size, dt);
  CHECK_ITERABLE_APPROX(max_n_dot_f, expected_max_n_dot_f);
}

void test_max_normal_fluxes_2d() noexcept {
  const Mesh<2> mesh({{3, 3}}, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const size_t number_of_points = mesh.number_of_grid_points();
  const std::array<double, 2> element_size = {{0.4, 0.4}};
  const TimeDelta dt(Slab(0., 1.), Rational(1, 10));
  const auto element = TestHelpers::Limiters::make_element<2>();

  const Scalar<DataVector> scalar(
      DataVector{{1.2, 1.4, 1.6, 1.1, 1.3, 1.5, 1.0, 1.2, 1.4}});
  const Scalar<DataVector> expected_max_n_dot_f(
      DataVector(number_of_points, 1.3));

  Scalar<DataVector> max_n_dot_f;
  NewtonianEuler::max_n_dot_f_of_positive_scalar(
      make_not_null(&max_n_dot_f), scalar, mesh, element, element_size, dt);
  CHECK_ITERABLE_APPROX(max_n_dot_f, expected_max_n_dot_f);
}

void test_max_normal_fluxes_3d() noexcept {
  const Mesh<3> mesh({{3, 3, 3}}, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const size_t number_of_points = mesh.number_of_grid_points();
  const std::array<double, 3> element_size = {{0.4, 0.4, 0.4}};
  const TimeDelta dt(Slab(0., 1.), Rational(1, 10));
  const auto element = TestHelpers::Limiters::make_element<3>();

  const Scalar<DataVector> scalar(DataVector{
      {1.2, 1.4, 1.6, 1.1, 1.3, 1.5, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 1.5, 1.7,
       1.9, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 1.9, 2.1, 2.3, 1.8, 2.0, 2.2}});
  const Scalar<DataVector> expected_max_n_dot_f(
      DataVector(number_of_points, 1.1333333333333333));

  Scalar<DataVector> max_n_dot_f;
  NewtonianEuler::max_n_dot_f_of_positive_scalar(
      make_not_null(&max_n_dot_f), scalar, mesh, element, element_size, dt);
  CHECK_ITERABLE_APPROX(max_n_dot_f, expected_max_n_dot_f);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.NumericalFluxes.MaxNormalFluxes",
    "[Unit][Evolution]") {
  test_max_normal_fluxes_1d();
  test_max_normal_fluxes_2d();
  test_max_normal_fluxes_3d();
}

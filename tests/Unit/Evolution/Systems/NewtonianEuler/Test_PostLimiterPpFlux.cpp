// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Direction.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/NewtonianEuler/PostLimiterPpFlux.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Time/Time.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/StdHelpers.hpp"
//#include "tests/Unit/Domain/DomainTestHelpers.hpp"
//#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {

using Affine = domain::CoordinateMaps::Affine;
using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;

void test_plppf_1d() noexcept {
  const Mesh<1> mesh(3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const ElementId<1> self_id(1, {{{2, 0}}});
  auto element_map = ElementMap<1, Frame::Inertial>(
      self_id,
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Affine(-1., 1., 3., 7.)));

  std::unordered_map<Direction<1>, tnsr::i<DataVector, 1>>
      unnormalized_normals{};
  for (const auto& dir : Direction<1>::all_directions()) {
    const auto n =
        unnormalized_face_normal(mesh.slice_away(0), element_map, dir);
    unnormalized_normals.insert(std::make_pair(dir, n));
  }
  const std::unordered_map<Direction<1>, tnsr::i<DataVector, 1>>
      unnormalized_boundary_normals{};

  const TimeDelta dt(Slab(0., 1.), Rational(1, 10));

  DataVector zero(mesh.number_of_grid_points(), 0.);
  Scalar<DataVector> mass_density(zero);
  tnsr::I<DataVector, 1> velocity(zero);
  Scalar<DataVector> energy_density(zero);

  // test with uniform density, should not trigger
  get(mass_density) = 2.;
  get<0>(velocity) = 0.4;
  auto momentum_density = velocity;
  get<0>(momentum_density) *= get(mass_density);

  bool ppflux_triggered;
  VariableFixing::NewtonianEuler::PostLimiterPpFlux<1> plppf{};
  ppflux_triggered =
      plppf(make_not_null(&mass_density), make_not_null(&momentum_density),
            make_not_null(&energy_density), mesh, logical_coords, element_map,
            unnormalized_normals, unnormalized_boundary_normals, dt);
  CHECK_FALSE(ppflux_triggered);

  // test with small outflow, should not trigger
  // total mass in cell = rho * vol = 2 * 4 = 8
  get<0>(velocity) = DataVector{{-1.5, 0., 2.}};
  momentum_density = velocity;
  get<0>(momentum_density) *= get(mass_density);

  ppflux_triggered =
      plppf(make_not_null(&mass_density), make_not_null(&momentum_density),
            make_not_null(&energy_density), mesh, logical_coords, element_map,
            unnormalized_normals, unnormalized_boundary_normals, dt);
  CHECK_FALSE(ppflux_triggered);

  // test with larger outflow, should trigger
  get<0>(velocity) = DataVector{{-5.5, -2., -1.}};
  momentum_density = velocity;
  get<0>(momentum_density) *= get(mass_density);

  ppflux_triggered =
      plppf(make_not_null(&mass_density), make_not_null(&momentum_density),
            make_not_null(&energy_density), mesh, logical_coords, element_map,
            unnormalized_normals, unnormalized_boundary_normals, dt);
  CHECK(ppflux_triggered);
}

void test_plppf_2d() noexcept {
  const Mesh<2> mesh(3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const ElementId<2> self_id(1, {{{2, 0}, {1, 0}}});
  const Affine xi_map{-1., 1., 3., 7.};
  const Affine eta_map{-1., 1., 7., 3.};
  auto element_map = ElementMap<2, Frame::Inertial>(
      self_id,
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Affine2D(xi_map, eta_map)));

  std::unordered_map<Direction<2>, tnsr::i<DataVector, 2>>
      unnormalized_normals{};
  for (const auto& dir : Direction<2>::all_directions()) {
    const auto n = unnormalized_face_normal(mesh.slice_away(dir.dimension()),
                                            element_map, dir);
    unnormalized_normals.insert(std::make_pair(dir, n));
  }
  const std::unordered_map<Direction<2>, tnsr::i<DataVector, 2>>
      unnormalized_boundary_normals{};

  const TimeDelta dt(Slab(0., 1.), Rational(1, 10));

  DataVector zero(mesh.number_of_grid_points(), 0.);
  Scalar<DataVector> mass_density(zero);
  tnsr::I<DataVector, 2> velocity(zero);
  Scalar<DataVector> energy_density(zero);

  // test with uniform density, should not trigger
  get(mass_density) = 2.;
  get<0>(velocity) = 0.4;
  get<1>(velocity) = 2.4;
  auto momentum_density = velocity;
  get<0>(momentum_density) *= get(mass_density);
  get<1>(momentum_density) *= get(mass_density);

  bool ppflux_triggered;
  VariableFixing::NewtonianEuler::PostLimiterPpFlux<2> plppf{};
  ppflux_triggered =
      plppf(make_not_null(&mass_density), make_not_null(&momentum_density),
            make_not_null(&energy_density), mesh, logical_coords, element_map,
            unnormalized_normals, unnormalized_boundary_normals, dt);
  CHECK_FALSE(ppflux_triggered);

  // test with small outflow, should not trigger
  // total mass in cell = rho * vol = 2 * 4 * 4 = 32
  get<0>(velocity) = DataVector{0., 0., 1., 0., 0., 1., 0., 0., 1.};
  get<1>(velocity) = DataVector{1., 1., 1., 0., 0., 0., -1., -1.5, -2.};
  momentum_density = velocity;
  get<0>(momentum_density) *= get(mass_density);
  get<1>(momentum_density) *= get(mass_density);

  ppflux_triggered =
      plppf(make_not_null(&mass_density), make_not_null(&momentum_density),
            make_not_null(&energy_density), mesh, logical_coords, element_map,
            unnormalized_normals, unnormalized_boundary_normals, dt);
  CHECK_FALSE(ppflux_triggered);

  // test with larger outflow, should trigger
  get<0>(velocity) = DataVector{0., 0., 1.2, 0., 0., 1., 0., 0., 1.};
  get<1>(velocity) = DataVector{1., 1., 1., 0., 0., 0., -2., -2., -2.};
  momentum_density = velocity;
  get<0>(momentum_density) *= get(mass_density);
  get<1>(momentum_density) *= get(mass_density);

  ppflux_triggered =
      plppf(make_not_null(&mass_density), make_not_null(&momentum_density),
            make_not_null(&energy_density), mesh, logical_coords, element_map,
            unnormalized_normals, unnormalized_boundary_normals, dt);
  CHECK(ppflux_triggered);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.PostLimiterPpFlux",
                  "[Unit][Evolution]") {
  test_plppf_1d();
  test_plppf_2d();
}

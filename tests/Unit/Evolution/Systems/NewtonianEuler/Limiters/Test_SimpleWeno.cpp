// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/Helpers.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/SimpleWeno.hpp"
#include "Evolution/Systems/NewtonianEuler/SoundSpeedSquared.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "tests/Unit/Evolution/DiscontinuousGalerkin/Limiters/TestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

template <size_t VolumeDim>
using VariablesMap = std::unordered_map<
    std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
    Variables<tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                         NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                         NewtonianEuler::Tags::EnergyDensity>>,
    boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>;

template <size_t VolumeDim>
std::unordered_map<
    std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
    typename Limiters::SimpleWeno<VolumeDim, 2>::PackagedData,
    boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
make_neighbor_data_from_neighbor_vars(
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const VariablesMap<VolumeDim>& neighbor_vars) noexcept {
  const auto make_tuple_of_means = [&mesh](
      const Variables<
          tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                     NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                     NewtonianEuler::Tags::EnergyDensity>>&
          vars_to_average) noexcept {
    tuples::TaggedTuple<
        ::Tags::Mean<NewtonianEuler::Tags::MassDensityCons>,
        ::Tags::Mean<NewtonianEuler::Tags::MomentumDensity<VolumeDim>>,
        ::Tags::Mean<NewtonianEuler::Tags::EnergyDensity>>
        result;
    get(get<::Tags::Mean<NewtonianEuler::Tags::MassDensityCons>>(result)) =
        mean_value(
            get(get<NewtonianEuler::Tags::MassDensityCons>(vars_to_average)),
            mesh);
    for (size_t d = 0; d < VolumeDim; ++d) {
      get<::Tags::Mean<NewtonianEuler::Tags::MomentumDensity<VolumeDim>>>(
          result)
          .get(d) = mean_value(
          get<NewtonianEuler::Tags::MomentumDensity<VolumeDim>>(vars_to_average)
              .get(d),
          mesh);
    }
    get(get<::Tags::Mean<NewtonianEuler::Tags::EnergyDensity>>(result)) =
        mean_value(
            get(get<NewtonianEuler::Tags::EnergyDensity>(vars_to_average)),
            mesh);
    return result;
  };

  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
      typename Limiters::SimpleWeno<VolumeDim, 2>::PackagedData,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      neighbor_data{};

  for (const auto& neighbor : element.neighbors()) {
    const auto dir = neighbor.first;
    const auto id = *(neighbor.second.cbegin());
    const auto dir_and_id = std::make_pair(dir, id);
    neighbor_data[dir_and_id].volume_data = neighbor_vars.at(dir_and_id);
    neighbor_data[dir_and_id].means =
        make_tuple_of_means(neighbor_vars.at(dir_and_id));
    neighbor_data[dir_and_id].mesh = mesh;
    neighbor_data[dir_and_id].element_size = element_size;
  }

  return neighbor_data;
}

void test_charweno_1d_redo(
    const std::unordered_set<Direction<1>>& directions_of_external_boundaries =
        {}) noexcept {
  INFO("Test simple WENO limiter in 1D");
  CAPTURE(directions_of_external_boundaries);
  const auto element =
      TestHelpers::Limiters::make_element<1>(directions_of_external_boundaries);
  const auto mesh =
      Mesh<1>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = make_array<1>(1.2);
  const EquationsOfState::IdealFluid<false> equation_of_state{1.4};

  // Conserved variable values
  const auto make_center_vars =
      [](const tnsr::I<DataVector, 1, Frame::Logical>& coords) noexcept {
    const auto& x = get<0>(coords);
    Variables<tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                         NewtonianEuler::Tags::MomentumDensity<1>,
                         NewtonianEuler::Tags::EnergyDensity>>
        vars(x.size());
    get(get<NewtonianEuler::Tags::MassDensityCons>(vars)) = 3. - x + square(x);
    get<0>(get<NewtonianEuler::Tags::MomentumDensity<1>>(vars)) =
        0.3 * x + 0.4 * square(x);
    get(get<NewtonianEuler::Tags::EnergyDensity>(vars)) =
        1.4 - 0.2 * x + 0.3 * square(x);
    return vars;
  };
  const auto make_lower_xi_vars =
      [](const tnsr::I<DataVector, 1, Frame::Logical>& coords,
         const double offset = 0.) noexcept {
    const auto x = get<0>(coords) + offset;
    Variables<tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                         NewtonianEuler::Tags::MomentumDensity<1>,
                         NewtonianEuler::Tags::EnergyDensity>>
        vars(x.size());
    get(get<NewtonianEuler::Tags::MassDensityCons>(vars)) =
        -1. - 10. * x - square(x);
    get<0>(get<NewtonianEuler::Tags::MomentumDensity<1>>(vars)) =
        -0.1 + 0.3 * x - 0.1 * square(x);
    get(get<NewtonianEuler::Tags::EnergyDensity>(vars)) =
        2.4 + 0.2 * x - 0.1 * square(x);
    return vars;
  };
  const auto make_upper_xi_vars =
      [](const tnsr::I<DataVector, 1, Frame::Logical>& coords,
         const double offset = 0.) noexcept {
    const auto x = get<0>(coords) + offset;
    Variables<tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                         NewtonianEuler::Tags::MomentumDensity<1>,
                         NewtonianEuler::Tags::EnergyDensity>>
        vars(x.size());
    get(get<NewtonianEuler::Tags::MassDensityCons>(vars)) =
        2.3 - x + 0.5 * square(x);
    get<0>(get<NewtonianEuler::Tags::MomentumDensity<1>>(vars)) =
        0.6 * x - 0.3 * square(x);
    get(get<NewtonianEuler::Tags::EnergyDensity>(vars)) =
        1.8 - 0.2 * x + 0.2 * square(x);
    return vars;
  };
  const auto local_vars = make_center_vars(logical_coords);
  VariablesMap<1> neighbor_vars{};
  const auto lower_xi =
      std::make_pair(Direction<1>::lower_xi(), ElementId<1>(1));
  neighbor_vars[lower_xi] = make_lower_xi_vars(logical_coords, -2.);
  const auto upper_xi =
      std::make_pair(Direction<1>::upper_xi(), ElementId<1>(2));
  neighbor_vars[upper_xi] = make_upper_xi_vars(logical_coords, 2.);

  const auto neighbor_data = make_neighbor_data_from_neighbor_vars(
      element, mesh, element_size, neighbor_vars);

  // Limit the conserved variables using the characteristic limiter
  const double neighbor_linear_weight = 0.001;
  using CharWeno = Limiters::SimpleWeno<1, 2>;

  auto density = get<NewtonianEuler::Tags::MassDensityCons>(local_vars);
  auto momentum = get<NewtonianEuler::Tags::MomentumDensity<1>>(local_vars);
  auto energy = get<NewtonianEuler::Tags::EnergyDensity>(local_vars);

  // After we reconstruct, we expect to recover the same mean
  const double expected_density_mean = mean_value(get(density), mesh);
  const auto expected_momentum_means = [&momentum, &mesh ]() noexcept {
    std::array<double, 1> means;
    for (size_t d = 0; d < 1; ++d) {
      gsl::at(means, d) = mean_value(momentum.get(d), mesh);
    }
    return means;
  }
  ();
  const double expected_energy_mean = mean_value(get(energy), mesh);

  const double tvb_constant = 0.0;
  const CharWeno weno(neighbor_linear_weight, tvb_constant);
  const bool activated = weno(make_not_null(&density), make_not_null(&momentum),
                              make_not_null(&energy), mesh, element,
                              element_size, equation_of_state, neighbor_data);

  // Sanity check that char limiter conserves the means
  CHECK(mean_value(get(density), mesh) == approx(expected_density_mean));
  for (size_t d = 0; d < 1; ++d) {
    CHECK(mean_value(momentum.get(d), mesh) ==
          approx(gsl::at(expected_momentum_means, d)));
  }
  CHECK(mean_value(get(energy), mesh) == approx(expected_energy_mean));

  // Compute char transformation matrices
  const auto mean_density = Scalar<double>{mean_value(
      get(get<NewtonianEuler::Tags::MassDensityCons>(local_vars)), mesh)};
  const auto mean_momentum = [&local_vars, &mesh ]() noexcept {
    tnsr::I<double, 1> result{};
    for (size_t i = 0; i < 1; ++i) {
      result.get(i) = mean_value(
          get<NewtonianEuler::Tags::MomentumDensity<1>>(local_vars).get(i),
          mesh);
    }
    return result;
  }
  ();
  const auto mean_energy = Scalar<double>{mean_value(
      get(get<NewtonianEuler::Tags::EnergyDensity>(local_vars)), mesh)};

  const tnsr::i<double, 1> normal{{{1.0}}};
  const auto right_and_left = NewtonianEuler::compute_eigenvectors(
      mean_density, mean_momentum, mean_energy, equation_of_state, normal);
  const auto& right = right_and_left.first;
  const auto& left = right_and_left.second;

  // Convert to characteristic variable values
  Variables<
      tmpl::list<NewtonianEuler::Tags::UMinus, NewtonianEuler::Tags::U0<1>,
                 NewtonianEuler::Tags::UPlus>>
      local_char_vars(mesh.number_of_grid_points());
  Limiters::NewtonianEulerWeno_detail::char_vars_from_cons_vars(
      make_not_null(&local_char_vars), local_vars, left);

  using Weno = Limiters::Weno<
      1, tmpl::list<NewtonianEuler::Tags::UMinus, NewtonianEuler::Tags::U0<1>,
                    NewtonianEuler::Tags::UPlus>>;
  std::unordered_map<std::pair<Direction<1>, ElementId<1>>,
                     typename Weno::PackagedData,
                     boost::hash<std::pair<Direction<1>, ElementId<1>>>>
      neighbor_char_data{};
  for (const auto& kv : neighbor_data) {
    const auto& key = kv.first;
    const auto& data = kv.second;
    neighbor_char_data[key].volume_data.initialize(
        mesh.number_of_grid_points());
    neighbor_char_data[key].mesh = data.mesh;
    neighbor_char_data[key].element_size = data.element_size;
  }
  for (const auto& kv : neighbor_data) {
    const auto& key = kv.first;
    const auto& data = kv.second;
    Limiters::NewtonianEulerWeno_detail::char_vars_from_cons_vars(
        make_not_null(&(neighbor_char_data[key].volume_data)), data.volume_data,
        left);
    Limiters::NewtonianEulerWeno_detail::char_means_from_cons_means(
        make_not_null(&(neighbor_char_data[key].means)), data.means, left);
  }

  // Limit the characteristics using the generic limmiter
  auto uminus = get<NewtonianEuler::Tags::UMinus>(local_char_vars);
  auto u0 = get<NewtonianEuler::Tags::U0<1>>(local_char_vars);
  auto uplus = get<NewtonianEuler::Tags::UPlus>(local_char_vars);

  const Weno weno2(Limiters::WenoType::SimpleWeno, neighbor_linear_weight,
                   tvb_constant);
  const bool activated2 =
      weno2(make_not_null(&uminus), make_not_null(&u0), make_not_null(&uplus),
            mesh, element, element_size, neighbor_char_data);

  // Transform limited characteristics back to conserved
  auto density2 = density;
  auto momentum2 = momentum;
  auto energy2 = energy;
  Limiters::NewtonianEulerWeno_detail::cons_tensors_from_char_tensors(
      make_not_null(&density2), make_not_null(&momentum2),
      make_not_null(&energy2), uminus, u0, uplus, right);

  // Check outputs are identical
  CHECK_ITERABLE_APPROX(density, density2);
  CHECK_ITERABLE_APPROX(momentum, momentum2);
  CHECK_ITERABLE_APPROX(energy, energy2);
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Limiters.SimpleWeno",
                  "[Unit][Evolution]") {
  // TODO(FH) higher dims
  test_charweno_1d_redo();
}

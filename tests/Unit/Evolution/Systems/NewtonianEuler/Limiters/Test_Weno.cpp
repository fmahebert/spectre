// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Weno.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoType.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/CharacteristicHelpers.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/Weno.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Limiters/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

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
    typename NewtonianEuler::Limiters::Weno<VolumeDim>::PackagedData,
    boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
make_neighbor_data_from_neighbor_vars(
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size,
    const VariablesMap<VolumeDim>& neighbor_vars) noexcept {
  const auto make_tuple_of_means =
      [&mesh](const Variables<
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
            mean_value(get(get<NewtonianEuler::Tags::MassDensityCons>(
                           vars_to_average)),
                       mesh);
        for (size_t d = 0; d < VolumeDim; ++d) {
          get<::Tags::Mean<NewtonianEuler::Tags::MomentumDensity<VolumeDim>>>(
              result)
              .get(d) =
              mean_value(get<NewtonianEuler::Tags::MomentumDensity<VolumeDim>>(
                             vars_to_average)
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
      typename NewtonianEuler::Limiters::Weno<VolumeDim>::PackagedData,
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

template <size_t VolumeDim>
void test_limiter_work(const Scalar<DataVector>& input_density,
                       const tnsr::I<DataVector, VolumeDim>& input_momentum,
                       const Scalar<DataVector>& input_energy,
                       const Mesh<VolumeDim>& mesh,
                       const std::array<double, VolumeDim>& element_size,
                       const EquationsOfState::EquationOfState<false, 1>& eos,
                       const VariablesMap<VolumeDim>& neighbor_vars) noexcept {
  const auto element = TestHelpers::Limiters::make_element<VolumeDim>();
  const auto neighbor_data = make_neighbor_data_from_neighbor_vars(
      mesh, element, element_size, neighbor_vars);

  auto density_generic = input_density;
  auto momentum_generic = input_momentum;
  auto energy_generic = input_energy;
  const double neighbor_linear_weight = 0.001;
  const double tvb_constant = 0.0;  // TODO default this
  const Limiters::Weno<
      VolumeDim, tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                            NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                            NewtonianEuler::Tags::EnergyDensity>>
      weno_generic(Limiters::WenoType::SimpleWeno, neighbor_linear_weight,
                   tvb_constant);
  const bool activated_generic = weno_generic(
      make_not_null(&density_generic), make_not_null(&momentum_generic),
      make_not_null(&energy_generic), mesh, element, element_size,
      neighbor_data);

  auto density_specialized = input_density;
  auto momentum_specialized = input_momentum;
  auto energy_specialized = input_energy;
  auto diagnostics = make_with_value<Scalar<DataVector>>(input_density, 0.0);
  const bool apply_flattener = false;
  const double kxrcf_constant = 1e99;  // unused
  const NewtonianEuler::Limiters::Weno<VolumeDim> weno_specialized(
      NewtonianEuler::Limiters::WenoType::ConservativeSimpleWeno,
      neighbor_linear_weight, tvb_constant, kxrcf_constant, apply_flattener);
  const bool activated_specialized = weno_specialized(
      make_not_null(&density_specialized), make_not_null(&momentum_specialized),
      make_not_null(&energy_specialized), make_not_null(&diagnostics), mesh,
      element, element_size, {}, eos, neighbor_data);

  CHECK(activated_generic);
  CHECK(diagnostics == make_with_value<Scalar<DataVector>>(input_density, 1.0));
  CHECK(activated_generic == activated_specialized);
  CHECK_ITERABLE_APPROX(density_generic, density_specialized);
  CHECK_ITERABLE_APPROX(momentum_generic, momentum_specialized);
  CHECK_ITERABLE_APPROX(energy_generic, energy_specialized);
}

void test_weno_limiter_1d() noexcept {
  INFO("Test NewtonianEuler::Limiters::Weno limiter in 1D");
  const auto mesh =
      Mesh<1>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = make_array<1>(0.5);
  const EquationsOfState::PolytropicFluid<false> polytropic_eos{1., 2.};

  const auto& x = get<0>(logical_coords);
  const auto mass_density_cons = [&x]() noexcept {
    return Scalar<DataVector>{{{1.0 + 0.2 * x + 0.05 * square(x)}}};
  }();
  const auto momentum_density = [&x]() noexcept {
    return tnsr::I<DataVector, 1>{{{0.2 - 0.3 * x}}};
  }();
  const auto energy_density = [&mass_density_cons, &momentum_density,
                               &polytropic_eos]() noexcept {
    const auto& rho = get(mass_density_cons);
    CAPTURE(rho);
    const DataVector energy =
        rho * get(polytropic_eos.specific_internal_energy_from_density(
                  mass_density_cons)) +
        0.5 * get(dot_product(momentum_density, momentum_density)) / rho;
    return Scalar<DataVector>{{{energy}}};
  }();

  VariablesMap<1> neighbor_vars;
  const std::array<std::pair<Direction<1>, ElementId<1>>, 2> dir_keys = {
      {{Direction<1>::lower_xi(), ElementId<1>(1)},
       {Direction<1>::upper_xi(), ElementId<1>(2)}}};
  for (const auto& id_pair : dir_keys) {
    neighbor_vars[id_pair].initialize(mesh.number_of_grid_points());
  }

  using rho = NewtonianEuler::Tags::MassDensityCons;
  get(get<rho>(neighbor_vars[dir_keys[0]])) = 0.4;
  get(get<rho>(neighbor_vars[dir_keys[1]])) = 1.1;

  using rhou = NewtonianEuler::Tags::MomentumDensity<1>;
  get<0>(get<rhou>(neighbor_vars[dir_keys[0]])) = 0.2;
  get<0>(get<rhou>(neighbor_vars[dir_keys[1]])) = -0.1;

  using eps = NewtonianEuler::Tags::EnergyDensity;
  get(get<eps>(neighbor_vars[dir_keys[0]])) = 0.0;
  get(get<eps>(neighbor_vars[dir_keys[1]])) = 0.0;

  test_limiter_work(mass_density_cons, momentum_density, energy_density, mesh,
                    element_size, polytropic_eos, neighbor_vars);
}

void test_weno_limiter_2d() noexcept {
  INFO("Test NewtonianEuler::Limiters::Weno limiter in 2D");
  const auto mesh =
      Mesh<2>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = make_array(0.5, 1.0);
  const EquationsOfState::PolytropicFluid<false> polytropic_eos{1., 2.};

  const auto& x = get<0>(logical_coords);
  const auto& y = get<1>(logical_coords);
  const auto mass_density_cons = [&x, &y]() noexcept {
    return Scalar<DataVector>{
        {{1.0 + 0.2 * x - 0.1 * y + 0.05 * x * square(y)}}};
  }();
  const auto momentum_density = [&x, &y]() noexcept {
    tnsr::I<DataVector, 2> momentum;
    get<0>(momentum) = 0.2 - 0.3 * x + 0.1 * y;
    get<1>(momentum) = -0.1 + 0.03 * y - 0.01 * square(x) * square(y);
    return momentum;
  }();
  const auto energy_density = [&mass_density_cons, &momentum_density,
                               &polytropic_eos]() noexcept {
    const auto& rho = get(mass_density_cons);
    const DataVector energy =
        rho * get(polytropic_eos.specific_internal_energy_from_density(
                  mass_density_cons)) +
        0.5 * get(dot_product(momentum_density, momentum_density)) / rho;
    return Scalar<DataVector>{{{energy}}};
  }();

  VariablesMap<2> neighbor_vars;
  const std::array<std::pair<Direction<2>, ElementId<2>>, 4> dir_keys = {
      {{Direction<2>::lower_xi(), ElementId<2>(1)},
       {Direction<2>::upper_xi(), ElementId<2>(2)},
       {Direction<2>::lower_eta(), ElementId<2>(3)},
       {Direction<2>::upper_eta(), ElementId<2>(4)}}};
  for (const auto& id_pair : dir_keys) {
    neighbor_vars[id_pair].initialize(mesh.number_of_grid_points());
  }

  using rho = NewtonianEuler::Tags::MassDensityCons;
  get(get<rho>(neighbor_vars[dir_keys[0]])) = 0.4;
  get(get<rho>(neighbor_vars[dir_keys[1]])) = 1.1;
  get(get<rho>(neighbor_vars[dir_keys[2]])) = 2.1;
  get(get<rho>(neighbor_vars[dir_keys[3]])) = 0.9;

  using rhou = NewtonianEuler::Tags::MomentumDensity<2>;
  get<0>(get<rhou>(neighbor_vars[dir_keys[0]])) = 0.2;
  get<0>(get<rhou>(neighbor_vars[dir_keys[1]])) = -0.1;
  get<0>(get<rhou>(neighbor_vars[dir_keys[2]])) = 0.1;
  get<0>(get<rhou>(neighbor_vars[dir_keys[3]])) = 0.2;
  get<1>(get<rhou>(neighbor_vars[dir_keys[0]])) = -0.8;
  get<1>(get<rhou>(neighbor_vars[dir_keys[1]])) = -0.2;
  get<1>(get<rhou>(neighbor_vars[dir_keys[2]])) = 0.1;
  get<1>(get<rhou>(neighbor_vars[dir_keys[3]])) = -0.1;

  using eps = NewtonianEuler::Tags::EnergyDensity;
  get(get<eps>(neighbor_vars[dir_keys[0]])) = 0.0;
  get(get<eps>(neighbor_vars[dir_keys[1]])) = 0.0;
  get(get<eps>(neighbor_vars[dir_keys[2]])) = 0.0;
  get(get<eps>(neighbor_vars[dir_keys[3]])) = 0.0;

  test_limiter_work(mass_density_cons, momentum_density, energy_density, mesh,
                    element_size, polytropic_eos, neighbor_vars);
}

void test_weno_limiter_3d() noexcept {
  INFO("Test NewtonianEuler::Limiters::Weno limiter in 3D");
  const auto mesh =
      Mesh<3>(std::array<size_t, 3>{{3, 3, 4}}, Spectral::Basis::Legendre,
              Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = make_array(0.5, 1.0, 0.8);
  const EquationsOfState::PolytropicFluid<false> polytropic_eos{1., 2.};

  const auto& x = get<0>(logical_coords);
  const auto& y = get<1>(logical_coords);
  const auto& z = get<2>(logical_coords);
  const auto mass_density_cons = [&x, &y, &z]() noexcept {
    return Scalar<DataVector>{{{1.0 + 0.2 * x - 0.1 * y + 0.4 * z}}};
  }();
  const auto momentum_density = [&x, &y, &z]() noexcept {
    tnsr::I<DataVector, 3> momentum;
    get<0>(momentum) = 0.3 + 0.1 * z;
    get<1>(momentum) = -1.2 - 0.1 * y * z;
    get<2>(momentum) = 0.2 * x + 0.2 * y + 0.1 * z + 0.2 * square(z);
    return momentum;
  }();
  const auto energy_density = [&mass_density_cons, &momentum_density,
                               &polytropic_eos]() noexcept {
    const auto& rho = get(mass_density_cons);
    const DataVector energy =
        rho * get(polytropic_eos.specific_internal_energy_from_density(
                  mass_density_cons)) +
        0.5 * get(dot_product(momentum_density, momentum_density)) / rho;
    return Scalar<DataVector>{{{energy}}};
  }();

  VariablesMap<3> neighbor_vars;
  const std::array<std::pair<Direction<3>, ElementId<3>>, 6> dir_keys = {
      {{Direction<3>::lower_xi(), ElementId<3>(1)},
       {Direction<3>::upper_xi(), ElementId<3>(2)},
       {Direction<3>::lower_eta(), ElementId<3>(3)},
       {Direction<3>::upper_eta(), ElementId<3>(4)},
       {Direction<3>::lower_zeta(), ElementId<3>(5)},
       {Direction<3>::upper_zeta(), ElementId<3>(6)}}};
  for (const auto& id_pair : dir_keys) {
    neighbor_vars[id_pair].initialize(mesh.number_of_grid_points());
  }

  using rho = NewtonianEuler::Tags::MassDensityCons;
  get(get<rho>(neighbor_vars[dir_keys[0]])) = 0.0;
  get(get<rho>(neighbor_vars[dir_keys[1]])) = 0.0;
  get(get<rho>(neighbor_vars[dir_keys[2]])) = 0.1;
  get(get<rho>(neighbor_vars[dir_keys[3]])) = 1.2;
  get(get<rho>(neighbor_vars[dir_keys[4]])) = 1.1;
  get(get<rho>(neighbor_vars[dir_keys[5]])) = 0.9;

  using rhou = NewtonianEuler::Tags::MomentumDensity<3>;
  get<0>(get<rhou>(neighbor_vars[dir_keys[0]])) = 0.2;
  get<0>(get<rhou>(neighbor_vars[dir_keys[1]])) = -0.1;
  get<0>(get<rhou>(neighbor_vars[dir_keys[2]])) = 0.1;
  get<0>(get<rhou>(neighbor_vars[dir_keys[3]])) = 0.2;
  get<0>(get<rhou>(neighbor_vars[dir_keys[4]])) = 0.0;
  get<0>(get<rhou>(neighbor_vars[dir_keys[5]])) = 0.0;
  get<1>(get<rhou>(neighbor_vars[dir_keys[0]])) = 0.0;
  get<1>(get<rhou>(neighbor_vars[dir_keys[1]])) = 0.0;
  get<1>(get<rhou>(neighbor_vars[dir_keys[2]])) = 0.1;
  get<1>(get<rhou>(neighbor_vars[dir_keys[3]])) = 0.2;
  get<1>(get<rhou>(neighbor_vars[dir_keys[4]])) = -0.1;
  get<1>(get<rhou>(neighbor_vars[dir_keys[5]])) = -0.2;
  get<2>(get<rhou>(neighbor_vars[dir_keys[0]])) = 0.0;
  get<2>(get<rhou>(neighbor_vars[dir_keys[1]])) = 0.0;
  get<2>(get<rhou>(neighbor_vars[dir_keys[2]])) = 0.0;
  get<2>(get<rhou>(neighbor_vars[dir_keys[3]])) = 0.0;
  get<2>(get<rhou>(neighbor_vars[dir_keys[4]])) = 1.1;
  get<2>(get<rhou>(neighbor_vars[dir_keys[5]])) = 0.9;

  using eps = NewtonianEuler::Tags::EnergyDensity;
  get(get<eps>(neighbor_vars[dir_keys[0]])) = 0.0;
  get(get<eps>(neighbor_vars[dir_keys[1]])) = 0.0;
  get(get<eps>(neighbor_vars[dir_keys[2]])) = 0.0;
  get(get<eps>(neighbor_vars[dir_keys[3]])) = 0.0;
  get(get<eps>(neighbor_vars[dir_keys[4]])) = 0.0;
  get(get<eps>(neighbor_vars[dir_keys[5]])) = 0.0;

  test_limiter_work(mass_density_cons, momentum_density, energy_density, mesh,
                    element_size, polytropic_eos, neighbor_vars);
}

void test_char_simple_weno_1d() noexcept {
  INFO("Test NewtonianEuler char-var SimpleWeno limiter in 1D");
  const auto element = TestHelpers::Limiters::make_element<1>();
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
        get(get<NewtonianEuler::Tags::MassDensityCons>(vars)) =
            3. - x + square(x);
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
      mesh, element, element_size, neighbor_vars);

  // Limit the conserved variables using the characteristic limiter
  auto density = get<NewtonianEuler::Tags::MassDensityCons>(local_vars);
  auto momentum = get<NewtonianEuler::Tags::MomentumDensity<1>>(local_vars);
  auto energy = get<NewtonianEuler::Tags::EnergyDensity>(local_vars);

  // After we reconstruct, we expect to recover the same mean
  const double expected_density_mean = mean_value(get(density), mesh);
  const auto expected_momentum_means = [&momentum, &mesh]() noexcept {
    std::array<double, 1> means;
    for (size_t d = 0; d < 1; ++d) {
      gsl::at(means, d) = mean_value(momentum.get(d), mesh);
    }
    return means;
  }();
  const double expected_energy_mean = mean_value(get(energy), mesh);

  const double neighbor_linear_weight = 0.001;
  const double tvb_constant = 0.0;
  const double kxrcf_constant = 1e99;  // unused
  const bool apply_flattener = false;
  auto diagnostics = make_with_value<Scalar<DataVector>>(density, 0.0);
  const NewtonianEuler::Limiters::Weno<1> weno(
      NewtonianEuler::Limiters::WenoType::CharacteristicSimpleWeno,
      neighbor_linear_weight, tvb_constant, kxrcf_constant, apply_flattener);
  weno(make_not_null(&density), make_not_null(&momentum),
       make_not_null(&energy), make_not_null(&diagnostics), mesh, element,
       element_size, {}, equation_of_state, neighbor_data);

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
  const auto mean_momentum = [&local_vars, &mesh]() noexcept {
    tnsr::I<double, 1> result{};
    for (size_t i = 0; i < 1; ++i) {
      result.get(i) = mean_value(
          get<NewtonianEuler::Tags::MomentumDensity<1>>(local_vars).get(i),
          mesh);
    }
    return result;
  }();
  const auto mean_energy = Scalar<double>{mean_value(
      get(get<NewtonianEuler::Tags::EnergyDensity>(local_vars)), mesh)};

  const tnsr::i<double, 1> normal{{{1.0}}};
  const auto right_and_left = NewtonianEuler::Limiters::compute_eigenvectors(
      mean_density, mean_momentum, mean_energy, equation_of_state, normal);
  const auto& right = right_and_left.first;
  const auto& left = right_and_left.second;

  // Convert to characteristic variable values
  Variables<
      tmpl::list<NewtonianEuler::Tags::UMinus, NewtonianEuler::Tags::U0<1>,
                 NewtonianEuler::Tags::UPlus>>
      local_char_vars(mesh.number_of_grid_points());
  NewtonianEuler::Limiters::char_vars_from_cons_vars(
      make_not_null(&local_char_vars), local_vars, left);

  using WenoGeneric = Limiters::Weno<
      1, tmpl::list<NewtonianEuler::Tags::UMinus, NewtonianEuler::Tags::U0<1>,
                    NewtonianEuler::Tags::UPlus>>;
  std::unordered_map<std::pair<Direction<1>, ElementId<1>>,
                     typename WenoGeneric::PackagedData,
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
    NewtonianEuler::Limiters::char_vars_from_cons_vars(
        make_not_null(&(neighbor_char_data[key].volume_data)), data.volume_data,
        left);
    NewtonianEuler::Limiters::char_means_from_cons_means(
        make_not_null(&(neighbor_char_data[key].means)), data.means, left);
  }

  // Limit the characteristics using the generic limmiter
  auto uminus = get<NewtonianEuler::Tags::UMinus>(local_char_vars);
  auto u0 = get<NewtonianEuler::Tags::U0<1>>(local_char_vars);
  auto uplus = get<NewtonianEuler::Tags::UPlus>(local_char_vars);

  const WenoGeneric weno_generic(Limiters::WenoType::SimpleWeno,
                                 neighbor_linear_weight, tvb_constant);
  weno_generic(make_not_null(&uminus), make_not_null(&u0),
               make_not_null(&uplus), mesh, element, element_size,
               neighbor_char_data);

  // Transform limited characteristics back to conserved
  auto density_generic = density;
  auto momentum_generic = momentum;
  auto energy_generic = energy;
  NewtonianEuler::Limiters::cons_tensors_from_char_tensors(
      make_not_null(&density_generic), make_not_null(&momentum_generic),
      make_not_null(&energy_generic), uminus, u0, uplus, right);

  // Check outputs are identical
  CHECK_ITERABLE_APPROX(density, density_generic);
  CHECK_ITERABLE_APPROX(momentum, momentum_generic);
  CHECK_ITERABLE_APPROX(energy, energy_generic);
}

template <size_t VolumeDim>
void test_hweno_work(
    const Variables<tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                               NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                               NewtonianEuler::Tags::EnergyDensity>>&
        local_vars,
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size,
    const std::unordered_map<Direction<VolumeDim>,
                             tnsr::i<DataVector, VolumeDim>>&
        internal_unit_normals,
    const EquationsOfState::IdealFluid<false>& equation_of_state,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        typename NewtonianEuler::Limiters::Weno<VolumeDim>::PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data) noexcept {
  auto density = get<NewtonianEuler::Tags::MassDensityCons>(local_vars);
  auto momentum =
      get<NewtonianEuler::Tags::MomentumDensity<VolumeDim>>(local_vars);
  auto energy = get<NewtonianEuler::Tags::EnergyDensity>(local_vars);

  // After we reconstruct, we expect to recover the same mean
  const double expected_density_mean = mean_value(get(density), mesh);
  const auto expected_momentum_means = [&momentum, &mesh]() noexcept {
    std::array<double, VolumeDim> means;
    for (size_t d = 0; d < VolumeDim; ++d) {
      gsl::at(means, d) = mean_value(momentum.get(d), mesh);
    }
    return means;
  }();
  const double expected_energy_mean = mean_value(get(energy), mesh);

  const double neighbor_linear_weight = 0.001;
  const double tvb_constant = 1e99;  // unused
  const double kxrcf_constant = 0.0;
  const bool apply_flattener = true;
  auto diagnostics = make_with_value<Scalar<DataVector>>(density, 0.0);
  const NewtonianEuler::Limiters::Weno<1> weno(
      NewtonianEuler::Limiters::WenoType::CharacteristicHweno,
      neighbor_linear_weight, tvb_constant, kxrcf_constant, apply_flattener);
  const bool activated = weno(
      make_not_null(&density), make_not_null(&momentum), make_not_null(&energy),
      make_not_null(&diagnostics), mesh, element, element_size,
      internal_unit_normals, equation_of_state, neighbor_data);

  CHECK(activated);

  CHECK(density != get<NewtonianEuler::Tags::MassDensityCons>(local_vars));
  CHECK(momentum !=
        get<NewtonianEuler::Tags::MomentumDensity<VolumeDim>>(local_vars));
  CHECK(energy != get<NewtonianEuler::Tags::EnergyDensity>(local_vars));

  CHECK(mean_value(get(density), mesh) == approx(expected_density_mean));
  for (size_t d = 0; d < VolumeDim; ++d) {
    CHECK(mean_value(momentum.get(d), mesh) ==
          approx(gsl::at(expected_momentum_means, d)));
  }
  CHECK(mean_value(get(energy), mesh) == approx(expected_energy_mean));
}

void test_char_hweno_1d() noexcept {
  INFO("Test NewtonianEuler char-var Hweno limiter in 1D");
  const auto element = TestHelpers::Limiters::make_element<1>();
  const auto mesh =
      Mesh<1>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = make_array<1>(1.2);
  const EquationsOfState::IdealFluid<false> equation_of_state{5. / 3.};

  // Functions to produce dummy data on each element
  const auto make_center_vars =
      [](const tnsr::I<DataVector, 1, Frame::Logical>& coords) noexcept {
        const auto& x = get<0>(coords);
        Variables<tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                             NewtonianEuler::Tags::MomentumDensity<1>,
                             NewtonianEuler::Tags::EnergyDensity>>
            vars(x.size());
        get(get<NewtonianEuler::Tags::MassDensityCons>(vars)) =
            2. - 0.3 * x + square(x);
        get<0>(get<NewtonianEuler::Tags::MomentumDensity<1>>(vars)) =
            0.2 * x + 0.3 * square(x);
        get(get<NewtonianEuler::Tags::EnergyDensity>(vars)) = 0.7 - 0.1 * x;
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
            3.3 - x - 0.1 * square(x);
        get<0>(get<NewtonianEuler::Tags::MomentumDensity<1>>(vars)) =
            -0.1 + 0.3 * x + 0.2 * square(x);
        get(get<NewtonianEuler::Tags::EnergyDensity>(vars)) = 1.4 + 0.2 * x;
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
            2.3 - x + 0.1 * square(x);
        get<0>(get<NewtonianEuler::Tags::MomentumDensity<1>>(vars)) =
            0.6 * x - 0.3 * square(x);
        get(get<NewtonianEuler::Tags::EnergyDensity>(vars)) =
            2.1 - 0.2 * x + 0.4 * square(x);
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
      mesh, element, element_size, neighbor_vars);

  std::unordered_map<Direction<1>, tnsr::i<DataVector, 1>>
      internal_unit_normals;
  for (const auto& dir : Direction<1>::all_directions()) {
    internal_unit_normals.insert(
        std::make_pair(dir, tnsr::i<DataVector, 1>{DataVector(
                                1, dir.side() == Side::Lower ? -1. : 1.)}));
  }

  test_hweno_work<1>(local_vars, mesh, element, element_size,
                     internal_unit_normals, equation_of_state, neighbor_data);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Limiters.Weno",
                  "[Limiters][Unit]") {
  // TODO(FH): Write a complete test
  // Check cons-var weno matches the "generic" weno
  test_weno_limiter_1d();
  test_weno_limiter_2d();
  test_weno_limiter_3d();

  // Check char-var simple_weno matches generic simple_weno applies to char vars
  test_char_simple_weno_1d();

  // Very crude check of char-var hweno: just check it is callable as expected
  test_char_hweno_1d();
}

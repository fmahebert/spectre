// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/KxrcfTci.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Limiters/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace {

template <size_t VolumeDim>
struct TestPackagedData {
  Variables<tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                       NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                       NewtonianEuler::Tags::EnergyDensity>>
      volume_data;
  Mesh<VolumeDim> mesh;
};

template <size_t VolumeDim>
DirectionMap<VolumeDim, DataVector> make_dirmap_of_datavectors_from_value(
    const size_t size, const double value) noexcept {
  DirectionMap<VolumeDim, DataVector> result{};
  for (const auto& dir : Direction<VolumeDim>::all_directions()) {
    result[dir] = DataVector(size, value);
  }
  return result;
}

template <size_t VolumeDim>
void test_kxrcf_work(
    const bool expected_detection, const Scalar<DataVector>& cons_density,
    const tnsr::I<DataVector, VolumeDim>& cons_momentum,
    const Scalar<DataVector>& cons_energy,
    const DirectionMap<VolumeDim, DataVector>& neighbor_densities,
    const DirectionMap<VolumeDim, DataVector>& neighbor_energies,
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const double kxrcf_constant) noexcept {
  // Check that this help function is called correctly
  ASSERT(element.neighbors().size() == neighbor_densities.size(),
         "The test helper was passed inconsistent data.");
  ASSERT(element.neighbors().size() == neighbor_energies.size(),
         "The test helper was passed inconsistent data.");
  for (const auto& dir : Direction<VolumeDim>::all_directions()) {
    const auto& ids = element.neighbors().at(dir).ids();
    ASSERT(ids.size() == 1,
           "The test helper test_kxrcf_work isn't set up for h-refinement.");
  }

  // Create and fill neighbor data
  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
      TestPackagedData<VolumeDim>,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      neighbor_data{};
  for (const auto& dir : Direction<VolumeDim>::all_directions()) {
    const auto& id = *(element.neighbors().at(dir).ids().begin());
    TestPackagedData<VolumeDim>& neighbor =
        neighbor_data[std::make_pair(dir, id)];
    neighbor.mesh = mesh;
    neighbor.volume_data.initialize(mesh.number_of_grid_points(), 0.);
    get(get<NewtonianEuler::Tags::MassDensityCons>(neighbor.volume_data)) =
        neighbor_densities.at(dir);
    get(get<NewtonianEuler::Tags::EnergyDensity>(neighbor.volume_data)) =
        neighbor_energies.at(dir);
  }

  // Create and fill unit normals
  std::unordered_map<Direction<VolumeDim>, tnsr::i<DataVector, VolumeDim>>
      outward_unit_normals{};
  for (const auto& dir : Direction<VolumeDim>::all_directions()) {
    const DataVector used_for_size(
        mesh.slice_away(dir.dimension()).number_of_grid_points(), 0.);
    outward_unit_normals.insert(
        std::make_pair(dir, euclidean_basis_vector(dir, used_for_size)));
  }

  const bool tci_detection = NewtonianEuler::Limiters::Tci::kxrcf_indicator(
      cons_density, cons_momentum, cons_energy, neighbor_data, element, mesh,
      element_size, outward_unit_normals, kxrcf_constant);
  CHECK(tci_detection == expected_detection);
}

void test_kxrcf_1d() noexcept {
  const auto element = TestHelpers::Limiters::make_element<1>();
  const Mesh<1> mesh(3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto element_size = make_array<1>(1.2);

  Scalar<DataVector> cons_density{DataVector{{0.4, 0.3, 0.2}}};
  tnsr::I<DataVector, 1> cons_momentum{DataVector{{0.2, 0.1, 0.05}}};
  Scalar<DataVector> cons_energy{DataVector{{0.0, 1.3, 0.0}}};

  const auto neighbor_densities = make_dirmap_of_datavectors_from_value<1>(
      mesh.number_of_grid_points(), 0.);
  const auto neighbor_energies = make_dirmap_of_datavectors_from_value<1>(
      mesh.number_of_grid_points(), 0.);

  // trigger because of density inflow at lower boundary
  double kxrcf_constant = 0.;
  test_kxrcf_work(true, cons_density, cons_momentum, cons_energy,
                  neighbor_densities, neighbor_energies, element, mesh,
                  element_size, kxrcf_constant);

  // no trigger because threshold is not met
  kxrcf_constant = 3.;
  test_kxrcf_work(false, cons_density, cons_momentum, cons_energy,
                  neighbor_densities, neighbor_energies, element, mesh,
                  element_size, kxrcf_constant);

  // no trigger because inflow boundary has no jump in density
  kxrcf_constant = 0.;
  get(cons_density)[0] = 0.;
  test_kxrcf_work(false, cons_density, cons_momentum, cons_energy,
                  neighbor_densities, neighbor_energies, element, mesh,
                  element_size, kxrcf_constant);

  // trigger because of energy inflow at upper boundary
  cons_density = Scalar<DataVector>{DataVector{{0.0, 0.3, 0.0}}};
  cons_momentum = tnsr::I<DataVector, 1>{DataVector{{-0.2, 0.1, -0.05}}};
  cons_energy = Scalar<DataVector>{DataVector{{1.4, 1.3, 1.7}}};
  test_kxrcf_work(true, cons_density, cons_momentum, cons_energy,
                  neighbor_densities, neighbor_energies, element, mesh,
                  element_size, kxrcf_constant);

  // no trigger because inflow boundary has no jump in energy
  get(cons_energy)[mesh.number_of_grid_points() - 1] = 0.;
  test_kxrcf_work(false, cons_density, cons_momentum, cons_energy,
                  neighbor_densities, neighbor_energies, element, mesh,
                  element_size, kxrcf_constant);

  // no trigger because all boundaries are outflow
  cons_density = Scalar<DataVector>{DataVector{{0.4, 0.3, 0.2}}};
  cons_momentum = tnsr::I<DataVector, 1>{DataVector{{-0.2, 0.1, 0.05}}};
  test_kxrcf_work(false, cons_density, cons_momentum, cons_energy,
                  neighbor_densities, neighbor_energies, element, mesh,
                  element_size, kxrcf_constant);
}

void test_kxrcf_2d() noexcept {
  const auto element = TestHelpers::Limiters::make_element<2>();
  const Mesh<2> mesh({{3, 3}}, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto element_size = make_array<2>(1.2);

  const DataVector zero(mesh.number_of_grid_points(), 0.);

  Scalar<DataVector> cons_density{
      DataVector{{0.4, 0.3, 0.2, 0.3, 0.1, 0.2, 0.2, 0.2, 0.2}}};
  tnsr::I<DataVector, 2> cons_momentum{};
  get<0>(cons_momentum) =
      DataVector{{0.2, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  get<1>(cons_momentum) = zero;
  Scalar<DataVector> cons_energy{
      DataVector{{0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0}}};

  const auto neighbor_densities = make_dirmap_of_datavectors_from_value<2>(
      mesh.number_of_grid_points(), 0.);
  const auto neighbor_energies = make_dirmap_of_datavectors_from_value<2>(
      mesh.number_of_grid_points(), 0.);

  // trigger because of density inflow (at lower boundary)
  double kxrcf_constant = 0.;
  test_kxrcf_work(true, cons_density, cons_momentum, cons_energy,
                  neighbor_densities, neighbor_energies, element, mesh,
                  element_size, kxrcf_constant);

  // no trigger because threshold is not met
  kxrcf_constant = 3.;
  test_kxrcf_work(false, cons_density, cons_momentum, cons_energy,
                  neighbor_densities, neighbor_energies, element, mesh,
                  element_size, kxrcf_constant);

  // trigger because of energy inflow (at upper boundary)
  kxrcf_constant = 0.;
  cons_density = Scalar<DataVector>{
      DataVector{{0.0, 0.0, 0.0, 0.0, 1.3, 0.0, 0.0, 0.0, 0.0}}};
  get<0>(cons_momentum) = zero;
  get<1>(cons_momentum) =
      DataVector{{0.0, 0.0, 0.00, 0.0, 0.0, 0.0, -0.8, -1.3, -1.2}};
  cons_energy = Scalar<DataVector>{
      DataVector{{0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.3, 0.2, 0.1}}};
  test_kxrcf_work(true, cons_density, cons_momentum, cons_energy,
                  neighbor_densities, neighbor_energies, element, mesh,
                  element_size, kxrcf_constant);

  // no trigger because all boundaries are outflow
  cons_density = Scalar<DataVector>{
      DataVector{{0.4, 0.3, 0.2, 0.3, 0.1, 0.2, 0.2, 0.2, 0.2}}};
  get<0>(cons_momentum) =
      DataVector{{-0.2, 0.0, 0.05, -1.2, 0.0, 0.3, -0.8, 0.0, 0.2}};
  get<1>(cons_momentum) =
      DataVector{{-0.2, -0.1, -0.05, 0.0, 0.0, 0.0, 0.3, 1.2, 2.3}};
  test_kxrcf_work(false, cons_density, cons_momentum, cons_energy,
                  neighbor_densities, neighbor_energies, element, mesh,
                  element_size, kxrcf_constant);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Limiters.KxrcfTci",
                  "[Unit][Evolution]") {
  test_kxrcf_1d();
  test_kxrcf_2d();
}

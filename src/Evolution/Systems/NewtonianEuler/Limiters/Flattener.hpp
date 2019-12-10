// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianEuler {
namespace Limiters {

template <size_t VolumeDim, size_t ThermodynamicDim>
size_t flatten_solution(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const Mesh<VolumeDim>& mesh,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state) noexcept {
  // Flattening can only fix positivity violations if means are positive
  const double mean_density = mean_value(get(*mass_density_cons), mesh);
  const double mean_energy = mean_value(get(*energy_density), mesh);
  ASSERT(mean_density > 0.0, "Invalid flattener input mass density");
  if (ThermodynamicDim == 2) {
    ASSERT(mean_energy > 0.0, "Invalid flattener input energy density");
  }

  size_t flattener_status = 0;

  // If min(density) is negative, then flatten
  // TODO(FH): should all fields be scaled by the same factor?
  const double min_density = min(get(*mass_density_cons));
  if (min_density < 0.0) {
    constexpr double safety = 0.95;
    const double factor = safety * mean_density / (mean_density - min_density);
    get(*mass_density_cons) =
        mean_density + factor * (get(*mass_density_cons) - mean_density);
    for (size_t i = 0; i < VolumeDim; ++i) {
      const double mean_momentum = mean_value(momentum_density->get(i), mesh);
      momentum_density->get(i) =
          mean_momentum + factor * (momentum_density->get(i) - mean_momentum);
    }
    get(*energy_density) =
        mean_energy + factor * (get(*energy_density) - mean_energy);
    flattener_status = 1;
  }

  // Check no negative pressures
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

    // If min(pressure) is negative, then we give up and set fields to constant
    const double min_pressure = min(get(pressure));
    if (min_pressure < 0.0) {
      get(*mass_density_cons) = mean_density;
      for (size_t i = 0; i < VolumeDim; ++i) {
        momentum_density->get(i) = mean_value(momentum_density->get(i), mesh);
      }
      get(*energy_density) = mean_energy;
      flattener_status = 2;
    }
  }

  return flattener_status;
}

}  // namespace Limiters
}  // namespace NewtonianEuler

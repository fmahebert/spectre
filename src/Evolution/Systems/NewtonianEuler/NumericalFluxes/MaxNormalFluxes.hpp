// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/SizeOfElement.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "Time/Tags.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"

namespace NewtonianEuler {

template <size_t Dim>
void max_n_dot_f_of_positive_scalar(
    const gsl::not_null<Scalar<DataVector>*> max_n_dot_f,
    const Scalar<DataVector>& scalar, const Mesh<Dim>& mesh,
    const std::array<double, Dim>& element_size, const TimeDelta& dt) noexcept {
  const double mean_scalar = mean_value(get(scalar), mesh);
  ASSERT(mean_scalar > 0.0, "scalar already avg-negative :(");
  for (size_t d = 0; d < Dim; ++d) {
    ASSERT(equal_within_roundoff(element_size[d], element_size[0]),
           "currently assumes uniform grid in x/y/z");
  }
  get(*max_n_dot_f) =
      DataVector(mesh.number_of_grid_points(),
                 mean_scalar * element_size[0] / (2. * Dim * dt.value()));
}

namespace Tags {

struct MaxNormalFluxMassDensity : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct MaxNormalFluxMassDensityCompute : db::ComputeTag,
                                         MaxNormalFluxMassDensity {
  using base = MaxNormalFluxMassDensity;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<MassDensityCons, domain::Tags::Mesh<Dim>,
                 domain::Tags::SizeOfElement<Dim>, ::Tags::TimeStep>;

  static constexpr void (*function)(
      gsl::not_null<Scalar<DataVector>*>, const Scalar<DataVector>&,
      const Mesh<Dim>&, const std::array<double, Dim>&,
      const TimeDelta&) = max_n_dot_f_of_positive_scalar<Dim>;
};

struct MaxNormalFluxEnergyDensity : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct MaxNormalFluxEnergyDensityCompute : db::ComputeTag,
                                           MaxNormalFluxEnergyDensity {
  using base = MaxNormalFluxEnergyDensity;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<EnergyDensity, domain::Tags::Mesh<Dim>,
                 domain::Tags::SizeOfElement<Dim>, ::Tags::TimeStep>;

  static constexpr void (*function)(
      gsl::not_null<Scalar<DataVector>*>, const Scalar<DataVector>&,
      const Mesh<Dim>&, const std::array<double, Dim>&,
      const TimeDelta&) = max_n_dot_f_of_positive_scalar<Dim>;
};

}  // namespace Tags
}  // namespace NewtonianEuler

// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace VariableFixing {
namespace NewtonianEuler {

/// \ingroup VariableFixingGroup
/// \brief Fix the primitive variables to an atmosphere in low density regions
///
/// If the mass density is below  \f$\rho_{\textrm{cutoff}}\f$
/// (DensityCutoff), it is set to \f$\rho_{\textrm{atm}}\f$
/// (DensityOfAtmosphere), and the pressure, specific internal energy (for
/// one-dimensional equations of state), and specific enthalpy are adjusted to
/// satisfy the equation of state.  For a two-dimensional equation of state, the
/// specific internal energy is set to zero. In addition, the spatial velocity
/// is set to zero.
// TODO update
template <size_t Dim, size_t ThermodynamicDim>
class FixToAtmosphere {
 public:
  /// \brief Mass density of the atmosphere
  struct DensityOfAtmosphere {
    using type = double;
    static type lower_bound() noexcept { return 0.0; }
    static constexpr OptionString help = {"Density of atmosphere"};
  };
  /// \brief Mass density at which to impose the atmosphere. Should be
  /// greater than or equal to the density of the atmosphere.
  struct DensityCutoff {
    using type = double;
    static type lower_bound() noexcept { return 0.0; }
    static constexpr OptionString help = {
        "Density to impose atmosphere at. Must be >= rho_atm"};
  };

  using options = tmpl::list<DensityOfAtmosphere, DensityCutoff>;
  static constexpr OptionString help = {
      "If the mass density is below DensityCutoff, it is set\n"
      "to DensityOfAtmosphere, and the pressure, specific internal energy\n"
      "(for one-dimensional equations of state), and specific enthalpy are\n"
      "adjusted to satisfy the equation of state. For a two-dimensional\n"
      "equation of state, the specific internal energy is set to zero.\n"
      "In addition, the spatial velocity is set to zero.\n"};

  FixToAtmosphere(double density_of_atmosphere, double density_cutoff,
                  const OptionContext& context = {});

  FixToAtmosphere() = default;
  FixToAtmosphere(const FixToAtmosphere& /*rhs*/) = default;
  FixToAtmosphere& operator=(const FixToAtmosphere& /*rhs*/) = default;
  FixToAtmosphere(FixToAtmosphere&& /*rhs*/) noexcept = default;
  FixToAtmosphere& operator=(FixToAtmosphere&& /*rhs*/) noexcept = default;
  ~FixToAtmosphere() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

  using return_tags =
      tmpl::list<::NewtonianEuler::Tags::MassDensity<DataVector>,
                 ::NewtonianEuler::Tags::SpecificInternalEnergy<DataVector>,
                 ::NewtonianEuler::Tags::Velocity<DataVector, Dim>,
                 ::NewtonianEuler::Tags::Pressure<DataVector>>;
  using argument_tags = tmpl::list<hydro::Tags::EquationOfStateBase>;

  void operator()(
      gsl::not_null<Scalar<DataVector>*> mass_density,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> velocity,
      gsl::not_null<Scalar<DataVector>*> pressure,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
          equation_of_state) const noexcept;

 private:
  template <size_t LocalDim, size_t LocalThermodynamicDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(
      const FixToAtmosphere<LocalDim, LocalThermodynamicDim>& lhs,
      const FixToAtmosphere<LocalDim, LocalThermodynamicDim>& rhs) noexcept;

  double density_of_atmosphere_{std::numeric_limits<double>::signaling_NaN()};
  double density_cutoff_{std::numeric_limits<double>::signaling_NaN()};
};

template <size_t Dim, size_t ThermodynamicDim>
bool operator!=(const FixToAtmosphere<Dim, ThermodynamicDim>& lhs,
                const FixToAtmosphere<Dim, ThermodynamicDim>& rhs) noexcept;

}  // namespace NewtonianEuler
}  // namespace VariableFixing

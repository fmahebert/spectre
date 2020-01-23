// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
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
/// \brief Fix the conservative variable cell-averages to an atmosphere
template <size_t Dim, size_t ThermodynamicDim>
class FixConsMeansToAtmosphere {
 public:
  /// \brief Turn the cell-mean atmosphere off
  ///
  /// This option exists to temporarily disable the cell-mean atmosphere for
  /// debugging purposes. For problems where the cell-mean atmosphere is not
  /// needed, the preferred approach is to not compile the cell-mean atmosphere
  /// into the executable.
  struct DisableForDebugging {
    using type = bool;
    static type default_value() noexcept { return false; }
    static constexpr OptionString help = {"Disable the cell-mean atmosphere"};
  };
  using options = tmpl::list<DisableForDebugging>;
  static constexpr OptionString help = {
      "A cell-mean atmosphere to re-establish positivity over the cell"};
  static std::string name() noexcept { return "FixConsMeansToAtmo"; }

  FixConsMeansToAtmosphere(bool disable_for_debugging = false);
  FixConsMeansToAtmosphere(const FixConsMeansToAtmosphere& /*rhs*/) = default;
  FixConsMeansToAtmosphere& operator=(const FixConsMeansToAtmosphere& /*rhs*/) =
      default;
  FixConsMeansToAtmosphere(FixConsMeansToAtmosphere&& /*rhs*/) noexcept =
      default;
  FixConsMeansToAtmosphere& operator=(
      FixConsMeansToAtmosphere&& /*rhs*/) noexcept = default;
  ~FixConsMeansToAtmosphere() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept;

  using return_tags = tmpl::list<::NewtonianEuler::Tags::MassDensityCons,
                                 ::NewtonianEuler::Tags::MomentumDensity<Dim>,
                                 ::NewtonianEuler::Tags::EnergyDensity>;
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<Dim>, hydro::Tags::EquationOfStateBase>;

  void operator()(
      gsl::not_null<Scalar<DataVector>*> mass_density,
      gsl::not_null<tnsr::I<DataVector, Dim>*> momentum_density,
      gsl::not_null<Scalar<DataVector>*> energy_density, const Mesh<Dim>& mesh,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
          equation_of_state) const noexcept;

 private:
  template <size_t LocalDim, size_t LocalThermodynamicDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(
      const FixConsMeansToAtmosphere<LocalDim, LocalThermodynamicDim>& lhs,
      const FixConsMeansToAtmosphere<LocalDim, LocalThermodynamicDim>&
          rhs) noexcept;

  bool disable_for_debugging_;
};

template <size_t Dim, size_t ThermodynamicDim>
bool operator!=(
    const FixConsMeansToAtmosphere<Dim, ThermodynamicDim>& lhs,
    const FixConsMeansToAtmosphere<Dim, ThermodynamicDim>& rhs) noexcept;

}  // namespace NewtonianEuler
}  // namespace VariableFixing

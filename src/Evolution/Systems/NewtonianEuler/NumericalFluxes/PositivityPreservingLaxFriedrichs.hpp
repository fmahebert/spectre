// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/NewtonianEuler/Characteristics.hpp"
#include "Evolution/Systems/NewtonianEuler/NumericalFluxes/MaxNormalFluxes.hpp"
#include "Evolution/Systems/NewtonianEuler/TagsDeclarations.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
#include "Options/Options.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP

namespace Tags {
template <typename>
struct NormalDotFlux;
template <typename>
struct Normalized;
}  // namespace Tags

class DataVector;
template <typename>
class Variables;
/// \endcond

namespace NewtonianEuler {
namespace NumericalFluxes {

template <size_t Dim>
struct PositivityPreservingLaxFriedrichs
    : tt::ConformsTo<dg::protocols::NumericalFlux> {
  using char_speeds_tag = Tags::CharacteristicSpeedsCompute<Dim>;

  /// The maximum characteristic speed modulus on one side of the interface.
  struct MaxAbsCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
    static std::string name() noexcept { return "MaxAbsCharSpeed"; }
  };

  struct EnablePpChecks {
    using type = bool;
    static constexpr OptionString help = {"Enable the Pos Pres checks"};
  };
  using options = tmpl::list<EnablePpChecks>;
  static constexpr OptionString help = {
      "Compute the PositivityPreservingLaxFriedrichs NewtonianEuler flux."};

  PositivityPreservingLaxFriedrichs() noexcept = default;
  explicit PositivityPreservingLaxFriedrichs(
      const bool enable_pp_checks) noexcept
      : enable_pp_checks_(enable_pp_checks) {}

  // clang-tidy: non-const reference
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | enable_pp_checks_;
  }

  using variables_tags =
      tmpl::list<Tags::MassDensityCons, Tags::MomentumDensity<Dim>,
                 Tags::EnergyDensity>;

  using package_field_tags = tmpl::list<
      ::Tags::NormalDotFlux<Tags::MassDensityCons>,
      ::Tags::NormalDotFlux<Tags::MomentumDensity<Dim>>,
      ::Tags::NormalDotFlux<Tags::EnergyDensity>, Tags::MassDensityCons,
      Tags::MomentumDensity<Dim>, Tags::EnergyDensity, MaxAbsCharSpeed,
      Tags::MaxNormalFluxMassDensity, Tags::MaxNormalFluxEnergyDensity>;
  using package_extra_tags = tmpl::list<>;

  using argument_tags = tmpl::list<
      ::Tags::NormalDotFlux<Tags::MassDensityCons>,
      ::Tags::NormalDotFlux<Tags::MomentumDensity<Dim>>,
      ::Tags::NormalDotFlux<Tags::EnergyDensity>, Tags::MassDensityCons,
      Tags::MomentumDensity<Dim>, Tags::EnergyDensity, char_speeds_tag,
      Tags::MaxNormalFluxMassDensity, Tags::MaxNormalFluxEnergyDensity>;

  void package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_n_dot_f_mass_density,
      gsl::not_null<tnsr::I<DataVector, Dim>*>
          packaged_n_dot_f_momentum_density,
      gsl::not_null<Scalar<DataVector>*> packaged_n_dot_f_energy_density,
      gsl::not_null<Scalar<DataVector>*> packaged_mass_density,
      gsl::not_null<tnsr::I<DataVector, Dim>*> packaged_momentum_density,
      gsl::not_null<Scalar<DataVector>*> packaged_energy_density,
      gsl::not_null<Scalar<DataVector>*> packaged_max_abs_char_speed,
      gsl::not_null<Scalar<DataVector>*> packaged_max_n_dot_f_mass_density,
      gsl::not_null<Scalar<DataVector>*> packaged_max_n_dot_f_energy_density,
      const Scalar<DataVector>& normal_dot_flux_mass_density,
      const tnsr::I<DataVector, Dim>& normal_dot_flux_momentum_density,
      const Scalar<DataVector>& normal_dot_flux_energy_density,
      const Scalar<DataVector>& mass_density,
      const tnsr::I<DataVector, Dim>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const db::const_item_type<char_speeds_tag>& characteristic_speeds,
      const Scalar<DataVector>& max_n_dot_f_mass_density,
      const Scalar<DataVector>& max_n_dot_f_energy_density) const noexcept;

  void operator()(
      gsl::not_null<Scalar<DataVector>*> normal_dot_numerical_flux_mass_density,
      gsl::not_null<tnsr::I<DataVector, Dim>*>
          normal_dot_numerical_flux_momentum_density,
      gsl::not_null<Scalar<DataVector>*>
          normal_dot_numerical_flux_energy_density,
      const Scalar<DataVector>& normal_dot_flux_mass_density_int,
      const tnsr::I<DataVector, Dim>& normal_dot_flux_momentum_density_int,
      const Scalar<DataVector>& normal_dot_flux_energy_density_int,
      const Scalar<DataVector>& mass_density_int,
      const tnsr::I<DataVector, Dim>& momentum_density_int,
      const Scalar<DataVector>& energy_density_int,
      const Scalar<DataVector>& max_abs_char_speed_int,
      const Scalar<DataVector>& max_n_dot_f_mass_density_int,
      const Scalar<DataVector>& max_n_dot_f_energy_density_int,
      const Scalar<DataVector>& minus_normal_dot_flux_mass_density_ext,
      const tnsr::I<DataVector, Dim>&
          minus_normal_dot_flux_momentum_density_ext,
      const Scalar<DataVector>& minus_normal_dot_flux_energy_density_ext,
      const Scalar<DataVector>& mass_density_ext,
      const tnsr::I<DataVector, Dim>& momentum_density_ext,
      const Scalar<DataVector>& energy_density_ext,
      const Scalar<DataVector>& max_abs_char_speed_ext,
      const Scalar<DataVector>& max_n_dot_f_mass_density_ext,
      const Scalar<DataVector>& max_n_dot_f_energy_density_ext) const noexcept;

  bool enable_pp_checks_;
};

}  // namespace NumericalFluxes
}  // namespace NewtonianEuler

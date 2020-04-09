// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <unordered_map>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Element.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Weno.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoType.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t VolumeDim>
class Direction;
template <size_t VolumeDim>
class ElementId;

namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace NewtonianEuler {
namespace Limiters {

template <size_t VolumeDim>
class Weno {
 public:
  using ConservativeVarsWeno = ::Limiters::Weno<
      VolumeDim, tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                            NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                            NewtonianEuler::Tags::EnergyDensity>>;

  struct ApplyFlattener {
    using type = bool;
    static constexpr OptionString help = {
        "Flatten after limiting to restore pointwise positivity"};
  };
  using options =
      tmpl::list<typename ConservativeVarsWeno::Type,
                 typename ConservativeVarsWeno::NeighborWeight,
                 typename ConservativeVarsWeno::TvbConstant, ApplyFlattener,
                 typename ConservativeVarsWeno::DisableForDebugging>;
  static constexpr OptionString help = {
      "A WENO limiter specialized to the NewtonianEuler system"};

  Weno(::Limiters::WenoType weno_type, double neighbor_linear_weight,
       double tvb_constant, bool apply_flattener,
       bool disable_for_debugging = false) noexcept;

  Weno() noexcept = default;
  Weno(const Weno& /*rhs*/) = default;
  Weno& operator=(const Weno& /*rhs*/) = default;
  Weno(Weno&& /*rhs*/) noexcept = default;
  Weno& operator=(Weno&& /*rhs*/) noexcept = default;
  ~Weno() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

  using PackagedData = typename ConservativeVarsWeno::PackagedData;
  using package_argument_tags =
      typename ConservativeVarsWeno::package_argument_tags;

  /// \brief Package data for sending to neighbor elements
  void package_data(gsl::not_null<PackagedData*> packaged_data,
                    const Scalar<DataVector>& mass_density_cons,
                    const tnsr::I<DataVector, VolumeDim>& momentum_density,
                    const Scalar<DataVector>& energy_density,
                    const Mesh<VolumeDim>& mesh,
                    const std::array<double, VolumeDim>& element_size,
                    const OrientationMap<VolumeDim>& orientation_map) const
      noexcept;

  using limit_tags =
      tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                 NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                 NewtonianEuler::Tags::EnergyDensity,
                 ::Tags::LimiterDiagnostics>;
  using limit_argument_tags = tmpl::list<domain::Tags::Mesh<VolumeDim>,
                                         domain::Tags::Element<VolumeDim>,
                                         domain::Tags::SizeOfElement<VolumeDim>,
                                         ::hydro::Tags::EquationOfStateBase>;

  /// \brief Limit the solution on the element
  template <size_t ThermodynamicDim>
  bool operator()(
      gsl::not_null<Scalar<DataVector>*> mass_density_cons,
      gsl::not_null<tnsr::I<DataVector, VolumeDim>*> momentum_density,
      gsl::not_null<Scalar<DataVector>*> energy_density,
      gsl::not_null<Scalar<DataVector>*> limiter_diagnostics,
      const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
      const std::array<double, VolumeDim>& element_size,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
          equation_of_state,
      const std::unordered_map<
          std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
          boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
          neighbor_data) const noexcept;

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration) false positive
  friend bool operator==(const Weno<LocalDim>& lhs,
                         const Weno<LocalDim>& rhs) noexcept;

  ::Limiters::WenoType weno_type_;
  double neighbor_linear_weight_;
  double tvb_constant_;
  bool apply_flattener_;
  bool disable_for_debugging_;
};

template <size_t VolumeDim>
bool operator!=(const Weno<VolumeDim>& lhs,
                const Weno<VolumeDim>& rhs) noexcept;

}  // namespace Limiters
}  // namespace NewtonianEuler

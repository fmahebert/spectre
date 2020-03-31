// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstdlib>
#include <unordered_map>
#include <utility>

#include "DataStructures/Tags.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodType.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t VolumeDim>
class Direction;
template <size_t VolumeDim>
class Element;
template <size_t VolumeDim>
class ElementId;
template <size_t VolumeDim>
class Mesh;
template <size_t VolumeDim>
class OrientationMap;

namespace boost {
template <class T>
struct hash;
}  // namespace boost

namespace PUP {
class er;
}  // namespace PUP

namespace domain {
namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
template <size_t VolumeDim>
struct Element;
template <size_t VolumeDim>
struct Mesh;
template <size_t VolumeDim>
struct SizeOfElement;
}  // namespace Tags
}  // namespace domain
/// \endcond

namespace NewtonianEuler {
namespace Limiters {

/// \ingroup LimitersGroup
/// \brief A general minmod slope limiter
template <size_t VolumeDim>
class Minmod {
 public:
  using ConservativeVarsMinmod = ::Limiters::Minmod<
      VolumeDim, tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                            NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                            NewtonianEuler::Tags::EnergyDensity>>;

  using options =
      tmpl::list<typename ConservativeVarsMinmod::Type,
                 typename ConservativeVarsMinmod::TvbConstant,
                 typename ConservativeVarsMinmod::DisableForDebugging>;
  static constexpr OptionString help = {
      "A Minmod limiter specialized to the NewtonianEuler system"};

  explicit Minmod(::Limiters::MinmodType minmod_type, double tvb_constant = 0.0,
                  bool disable_for_debugging = false) noexcept;

  Minmod() noexcept = default;
  Minmod(const Minmod& /*rhs*/) = default;
  Minmod& operator=(const Minmod& /*rhs*/) = default;
  Minmod(Minmod&& /*rhs*/) noexcept = default;
  Minmod& operator=(Minmod&& /*rhs*/) noexcept = default;
  ~Minmod() = default;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

  using PackagedData = typename ConservativeVarsMinmod::PackagedData;
  using package_argument_tags =
      typename ConservativeVarsMinmod::package_argument_tags;

  /// \brief Package data for sending to neighbor elements.
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
  using limit_argument_tags =
      tmpl::list<domain::Tags::Mesh<VolumeDim>,
                 domain::Tags::Element<VolumeDim>,
                 domain::Tags::Coordinates<VolumeDim, Frame::Logical>,
                 domain::Tags::SizeOfElement<VolumeDim>,
                 ::hydro::Tags::EquationOfStateBase>;

  /// \brief Limits the solution on the element.
  template <size_t ThermodynamicDim>
  bool operator()(
      gsl::not_null<Scalar<DataVector>*> mass_density_cons,
      gsl::not_null<tnsr::I<DataVector, VolumeDim>*> momentum_density,
      gsl::not_null<Scalar<DataVector>*> energy_density,
      gsl::not_null<Scalar<DataVector>*> limiter_diagnostics,
      const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
      const tnsr::I<DataVector, VolumeDim, Frame::Logical>& logical_coords,
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
  friend bool operator==(const Minmod<LocalDim>& lhs,
                         const Minmod<LocalDim>& rhs) noexcept;

  ::Limiters::MinmodType minmod_type_;
  double tvb_constant_;
  bool disable_for_debugging_;
};

template <size_t VolumeDim>
bool operator!=(const Minmod<VolumeDim>& lhs,
                const Minmod<VolumeDim>& rhs) noexcept;

}  // namespace Limiters
}  // namespace NewtonianEuler

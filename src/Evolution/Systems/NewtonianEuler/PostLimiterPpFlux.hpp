// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

//#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Options/Options.hpp"
//#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Time/Tags.hpp"
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
/// \brief Flatten if needed to control fluxes and preserve positivity
template <size_t Dim>
class PostLimiterPpFlux {
 public:
  /// \brief Turn the positivity-preservation flattening
  ///
  /// This option exists to temporarily disable the flattener for debugging
  /// purposes. For problems where flattening is not needed, the preferred
  /// approach is to not compile the flattener into the executable.
  struct DisableForDebugging {
    using type = bool;
    static type default_value() noexcept { return false; }
    static constexpr OptionString help = {"Disable the PpFlux flattener"};
  };
  using options = tmpl::list<DisableForDebugging>;
  static constexpr OptionString help = {"A PpFlux flattener over the cell"};

  PostLimiterPpFlux(bool disable_for_debugging = false);
  PostLimiterPpFlux(const PostLimiterPpFlux& /*rhs*/) = default;
  PostLimiterPpFlux& operator=(const PostLimiterPpFlux& /*rhs*/) = default;
  PostLimiterPpFlux(PostLimiterPpFlux&& /*rhs*/) noexcept = default;
  PostLimiterPpFlux& operator=(PostLimiterPpFlux&& /*rhs*/) noexcept = default;
  ~PostLimiterPpFlux() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept;

  using return_tags = tmpl::list<::NewtonianEuler::Tags::MassDensityCons,
                                 ::NewtonianEuler::Tags::MomentumDensity<Dim>,
                                 ::NewtonianEuler::Tags::EnergyDensity>;
  using argument_tags = tmpl::list<
      domain::Tags::Mesh<Dim>, domain::Tags::Coordinates<Dim, Frame::Logical>,
      domain::Tags::ElementMap<Dim, Frame::Grid>,
      domain::Tags::Interface<domain::Tags::InternalDirections<Dim>,
                              domain::Tags::UnnormalizedFaceNormal<Dim>>,
      domain::Tags::Interface<domain::Tags::BoundaryDirectionsInterior<Dim>,
                              domain::Tags::UnnormalizedFaceNormal<Dim>>,
      ::Tags::TimeStep>;

  bool operator()(
      gsl::not_null<Scalar<DataVector>*> mass_density,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          momentum_density,
      gsl::not_null<Scalar<DataVector>*> energy_density, const Mesh<Dim>& mesh,
      const tnsr::I<DataVector, Dim, Frame::Logical>& logical_coordinates,
      // TODO(FH): THE FRAME IS WRONG - THIS ASSUMES NO MOVING MESH
      const ElementMap<Dim, Frame::Grid>& map,
      const std::unordered_map<Direction<Dim>,
                               tnsr::i<DataVector, Dim, Frame::Inertial>>&
          unnormalized_normals,
      const std::unordered_map<Direction<Dim>,
                               tnsr::i<DataVector, Dim, Frame::Inertial>>&
          boundary_unnormalized_normals,
      const TimeDelta& time_step) const noexcept;

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const PostLimiterPpFlux<LocalDim>& lhs,
                         const PostLimiterPpFlux<LocalDim>& rhs) noexcept;

  bool disable_for_debugging_;
};

template <size_t Dim>
bool operator!=(const PostLimiterPpFlux<Dim>& lhs,
                const PostLimiterPpFlux<Dim>& rhs) noexcept;

}  // namespace NewtonianEuler
}  // namespace VariableFixing

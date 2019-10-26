// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Element.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Systems/NewtonianEuler/Characteristics.hpp"
#include "Evolution/Systems/NewtonianEuler/SoundSpeedSquared.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

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
namespace Tags {
struct UMinus : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Char::UMinus"; }
};

template <size_t VolumeDim>
struct U0 : db::SimpleTag {
  using type = tnsr::I<DataVector, VolumeDim>;
  static std::string name() noexcept { return "Char::U0"; }
};

struct UPlus : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Char::UPlus"; }
};
}  // namespace Tags

template <size_t VolumeDim, size_t ThermodynamicDim>
std::pair<Matrix, Matrix> compute_eigenvectors(
    const Scalar<double>& mean_density,
    const tnsr::I<double, VolumeDim>& mean_momentum,
    const Scalar<double>& mean_energy,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state,
    const tnsr::i<double, VolumeDim>& unit_normal) noexcept;

}  // namespace NewtonianEuler

namespace Limiters {
namespace NewtonianEulerWeno_detail {

template <size_t VolumeDim>
void char_means_from_cons_means(
    const gsl::not_null<
        tuples::TaggedTuple<::Tags::Mean<NewtonianEuler::Tags::UMinus>,
                            ::Tags::Mean<NewtonianEuler::Tags::U0<VolumeDim>>,
                            ::Tags::Mean<NewtonianEuler::Tags::UPlus>>*>
        char_means,
    const tuples::TaggedTuple<
        ::Tags::Mean<NewtonianEuler::Tags::MassDensityCons>,
        ::Tags::Mean<NewtonianEuler::Tags::MomentumDensity<VolumeDim>>,
        ::Tags::Mean<NewtonianEuler::Tags::EnergyDensity>>& cons_means,
    const Matrix& left) noexcept;

template <size_t VolumeDim>
void char_tensors_from_cons_tensors(
    const gsl::not_null<Scalar<DataVector>*> char_uminus,
    const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> char_u0,
    const gsl::not_null<Scalar<DataVector>*> char_uplus,
    const Scalar<DataVector>& cons_mass_density,
    const tnsr::I<DataVector, VolumeDim>& cons_momentum_density,
    const Scalar<DataVector>& cons_energy_density, const Matrix& left) noexcept;

template <size_t VolumeDim>
void cons_tensors_from_char_tensors(
    const gsl::not_null<Scalar<DataVector>*> cons_mass_density,
    const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> cons_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> cons_energy_density,
    const Scalar<DataVector>& char_uminus,
    const tnsr::I<DataVector, VolumeDim>& char_u0,
    const Scalar<DataVector>& char_uplus, const Matrix& right) noexcept;

template <size_t VolumeDim>
void char_vars_from_cons_vars(
    const gsl::not_null<Variables<tmpl::list<
        NewtonianEuler::Tags::UMinus, NewtonianEuler::Tags::U0<VolumeDim>,
        NewtonianEuler::Tags::UPlus>>*>
        char_vars,
    const Variables<tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                               NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                               NewtonianEuler::Tags::EnergyDensity>>& cons_vars,
    const Matrix& left) noexcept;

}  // namespace NewtonianEulerWeno_detail
}  // namespace Limiters

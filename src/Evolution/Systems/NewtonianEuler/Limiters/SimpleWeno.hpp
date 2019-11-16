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
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Element.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"
#include "Domain/OrientationMapHelpers.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodTci.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodType.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/SimpleWenoImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Weno.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoGridHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoType.hpp"
#include "Evolution/Systems/NewtonianEuler/Characteristics.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/Helpers.hpp"
#include "Evolution/Systems/NewtonianEuler/SoundSpeedSquared.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "Options/Options.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
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

// TODO(FH) namespace
namespace Limiters {

/// \ingroup LimitersGroup
/// \brief TODO(FH):
template <size_t VolumeDim, size_t ThermodynamicDim>
class SimpleWeno {
 public:
  using ConsWenoType =
      Weno<VolumeDim,
           tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                      NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                      NewtonianEuler::Tags::EnergyDensity>>;
  using CharWenoType =
      Weno<VolumeDim, tmpl::list<NewtonianEuler::Tags::UMinus,
                                 NewtonianEuler::Tags::U0<VolumeDim>,
                                 NewtonianEuler::Tags::UPlus>>;

  struct TvbConstant {
    using type = double;
    static type default_value() noexcept { return 1.; }
    static type lower_bound() noexcept { return 0.; }
    static constexpr OptionString help = {"Constant in RHS of TVB TCI"};
  };

  using options = tmpl::list<typename ConsWenoType::NeighborWeight, TvbConstant,
                             typename ConsWenoType::DisableForDebugging>;
  static constexpr OptionString help = {
      "A Simple WENO limiter for Newtonian Euler characteristic fields"};

  SimpleWeno(double neighbor_linear_weight, double tvb_constant,
             bool disable_for_debugging = false) noexcept;

  SimpleWeno() noexcept = default;
  SimpleWeno(const SimpleWeno& /*rhs*/) = default;
  SimpleWeno& operator=(const SimpleWeno& /*rhs*/) = default;
  SimpleWeno(SimpleWeno&& /*rhs*/) noexcept = default;
  SimpleWeno& operator=(SimpleWeno&& /*rhs*/) noexcept = default;
  ~SimpleWeno() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

  using PackagedData = typename ConsWenoType::PackagedData;
  using package_argument_tags = typename ConsWenoType::package_argument_tags;

  /// \brief Package data for sending to neighbor elements
  void package_data(const gsl::not_null<PackagedData*> packaged_data,
                    const Scalar<DataVector>& cons_mass_density,
                    const tnsr::I<DataVector, VolumeDim>& cons_momentum_density,
                    const Scalar<DataVector>& cons_energy_density,
                    const Mesh<VolumeDim>& mesh,
                    const std::array<double, VolumeDim>& element_size,
                    const OrientationMap<VolumeDim>& orientation_map) const
      noexcept;

  using limit_tags =
      tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                 NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                 NewtonianEuler::Tags::EnergyDensity>;
  using limit_argument_tags = tmpl::list<domain::Tags::Mesh<VolumeDim>,
                                         domain::Tags::Element<VolumeDim>,
                                         domain::Tags::SizeOfElement<VolumeDim>,
                                         ::hydro::Tags::EquationOfStateBase>;

  /// \brief Limit the solution on the element
  bool operator()(
      gsl::not_null<Scalar<DataVector>*> cons_mass_density,
      gsl::not_null<tnsr::I<DataVector, VolumeDim>*> cons_momentum_density,
      gsl::not_null<Scalar<DataVector>*> cons_energy_density,
      const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
      const std::array<double, VolumeDim>& element_size,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
          equation_of_state,
      const std::unordered_map<
          std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
          boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
          neighbor_data) const noexcept;

 private:
  template <size_t LocalDim, size_t LocalThermoDim>
  // NOLINTNEXTLINE(readability-redundant-declaration) false positive
  friend bool operator==(
      const SimpleWeno<LocalDim, LocalThermoDim>& lhs,
      const SimpleWeno<LocalDim, LocalThermoDim>& rhs) noexcept;

  double neighbor_linear_weight_;
  double tvb_constant_;
  bool disable_for_debugging_;
};

template <size_t VolumeDim, size_t ThermodynamicDim>
SimpleWeno<VolumeDim, ThermodynamicDim>::SimpleWeno(
    const double neighbor_linear_weight, const double tvb_constant,
    const bool disable_for_debugging) noexcept
    : neighbor_linear_weight_(neighbor_linear_weight),
      tvb_constant_(tvb_constant),
      disable_for_debugging_(disable_for_debugging) {}

template <size_t VolumeDim, size_t ThermodynamicDim>
// NOLINTNEXTLINE(google-runtime-references)
void SimpleWeno<VolumeDim, ThermodynamicDim>::pup(PUP::er& p) noexcept {
  p | neighbor_linear_weight_;
  p | tvb_constant_;
  p | disable_for_debugging_;
}

template <size_t VolumeDim, size_t ThermodynamicDim>
void SimpleWeno<VolumeDim, ThermodynamicDim>::package_data(
    const gsl::not_null<PackagedData*> packaged_data,
    const Scalar<DataVector>& cons_mass_density,
    const tnsr::I<DataVector, VolumeDim>& cons_momentum_density,
    const Scalar<DataVector>& cons_energy_density, const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const OrientationMap<VolumeDim>& orientation_map) const noexcept {
  const ConsWenoType limiter(WenoType::SimpleWeno, neighbor_linear_weight_,
                             disable_for_debugging_);
  limiter.package_data(packaged_data, cons_mass_density, cons_momentum_density,
                       cons_energy_density, mesh, element_size,
                       orientation_map);
}

template <size_t VolumeDim, size_t ThermodynamicDim>
bool SimpleWeno<VolumeDim, ThermodynamicDim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> cons_mass_density,
    const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> cons_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> cons_energy_density,
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data) const noexcept {
  if (UNLIKELY(disable_for_debugging_)) {
    // Do not modify input tensors
    return false;
  }

  // Enforce restrictions on h-refinement, p-refinement
  if (UNLIKELY(alg::any_of(element.neighbors(),
                           [](const auto& direction_neighbors) noexcept {
                             return direction_neighbors.second.size() != 1;
                           }))) {
    ERROR("The Weno limiter does not yet support h-refinement");
    // Removing this limitation will require:
    // - Generalizing the computation of the modified neighbor solutions.
    // - Generalizing the WENO weighted sum for multiple neighbors in each
    //   direction.
  }
  alg::for_each(neighbor_data, [&mesh](const auto& neighbor_and_data) noexcept {
    if (UNLIKELY(neighbor_and_data.second.mesh != mesh)) {
      ERROR("The Weno limiter does not yet support p-refinement");
      // Removing this limitation will require generalizing the
      // computation of the modified neighbor solutions.
    }
  });

  // Math for cons-char transformation
  // Means of cons vars
  const auto mean_density =
      Scalar<double>{mean_value(get(*cons_mass_density), mesh)};
  const auto mean_momentum = [&cons_momentum_density, &mesh ]() noexcept {
    tnsr::I<double, VolumeDim> result{};
    for (size_t i = 0; i < VolumeDim; ++i) {
      result.get(i) = mean_value(cons_momentum_density->get(i), mesh);
    }
    return result;
  }
  ();
  const auto mean_energy =
      Scalar<double>{mean_value(get(*cons_energy_density), mesh)};

  // Temp variables for calculations
  Variables<tmpl::list<
      NewtonianEuler::Tags::UMinus, NewtonianEuler::Tags::U0<VolumeDim>,
      NewtonianEuler::Tags::UPlus, ::Tags::TempScalar<0>,
      ::Tags::TempI<0, VolumeDim>, ::Tags::TempScalar<1>, ::Tags::TempScalar<2>,
      ::Tags::TempI<1, VolumeDim>, ::Tags::TempScalar<3>>>
      temp_buffer(mesh.number_of_grid_points());
  auto& local_char_uminus = get<NewtonianEuler::Tags::UMinus>(temp_buffer);
  auto& local_char_u0 = get<NewtonianEuler::Tags::U0<VolumeDim>>(temp_buffer);
  auto& local_char_uplus = get<NewtonianEuler::Tags::UPlus>(temp_buffer);
  auto& temp_cons_mass_density = get<::Tags::TempScalar<0>>(temp_buffer);
  auto& temp_cons_momentum_density =
      get<::Tags::TempI<0, VolumeDim>>(temp_buffer);
  auto& temp_cons_energy_density = get<::Tags::TempScalar<1>>(temp_buffer);
  auto& accumulate_cons_mass_density = get<::Tags::TempScalar<2>>(temp_buffer);
  auto& accumulate_cons_momentum_density =
      get<::Tags::TempI<1, VolumeDim>>(temp_buffer);
  auto& accumulate_cons_energy_density =
      get<::Tags::TempScalar<3>>(temp_buffer);

  // Only initialize the accumulating tensors
  get(accumulate_cons_mass_density) = 0.;
  for (size_t i = 0; i < VolumeDim; ++i) {
    accumulate_cons_momentum_density.get(i) = 0.;
  }
  get(accumulate_cons_energy_density) = 0.;

  // Storage for neighbor data, transformed into char variables
  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
      typename CharWenoType::PackagedData,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      neighbor_char_data{};
  for (const auto& kv : neighbor_data) {
    const auto& key = kv.first;
    const auto& data = kv.second;
    neighbor_char_data[key].volume_data.initialize(
        mesh.number_of_grid_points());
    neighbor_char_data[key].mesh = data.mesh;
    neighbor_char_data[key].element_size = data.element_size;
  }

  // Buffers for TCI
  Minmod_detail::BufferWrapper<VolumeDim> tci_buffer(mesh);
  const auto effective_neighbor_sizes =
      Minmod_detail::compute_effective_neighbor_sizes(element, neighbor_data);

  // Buffers for SimpleWeno extrapolated poly
  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
      intrp::RegularGrid<VolumeDim>,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      interpolator_buffer{};
  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, DataVector,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      modified_neighbor_solution_buffer{};

  bool some_component_was_limited = false;

  for (size_t d = 0; d < VolumeDim; ++d) {
    const auto normal = [&d]() noexcept {
      auto normal_components = make_array<VolumeDim>(0.);
      normal_components[d] = 1.;
      return tnsr::i<double, VolumeDim>(normal_components);
    }
    ();
    const auto right_and_left = NewtonianEuler::compute_eigenvectors(
        mean_density, mean_momentum, mean_energy, equation_of_state, normal);
    const auto& right = right_and_left.first;
    const auto& left = right_and_left.second;

    // Transform all field data to char vars:
    NewtonianEulerWeno_detail::char_tensors_from_cons_tensors(
        make_not_null(&local_char_uminus), make_not_null(&local_char_u0),
        make_not_null(&local_char_uplus), *cons_mass_density,
        *cons_momentum_density, *cons_energy_density, left);
    for (const auto& kv : neighbor_data) {
      const auto& key = kv.first;
      const auto& data = kv.second;
      NewtonianEulerWeno_detail::char_vars_from_cons_vars(
          make_not_null(&(neighbor_char_data[key].volume_data)),
          data.volume_data, left);
      NewtonianEulerWeno_detail::char_means_from_cons_means(
          make_not_null(&(neighbor_char_data[key].means)), data.means, left);
    }

    // Begin SimpleWENO logic
    bool some_component_was_limited_with_this_normal = false;
    const auto wrap_minmod_tci_and_simple_weno_impl = [
      this, &some_component_was_limited,
      &some_component_was_limited_with_this_normal, &tci_buffer,
      &interpolator_buffer, &modified_neighbor_solution_buffer, &mesh, &element,
      &element_size, &neighbor_char_data, &effective_neighbor_sizes
    ](auto tag, const auto tensor) noexcept {
      for (size_t tensor_storage_index = 0;
           tensor_storage_index < tensor->size(); ++tensor_storage_index) {
        // Check TCI
        const double tvb_constant = 0.0;
        const auto effective_neighbor_means =
            Minmod_detail::compute_effective_neighbor_means<decltype(tag)>(
                tensor_storage_index, element, neighbor_char_data);
        const bool component_needs_limiting = Tci::tvb_minmod_indicator(
            make_not_null(&tci_buffer), tvb_constant,
            (*tensor)[tensor_storage_index], mesh, element, element_size,
            effective_neighbor_means, effective_neighbor_sizes);

        if (component_needs_limiting) {
          Weno_detail::simple_weno_impl<decltype(tag)>(
              make_not_null(&interpolator_buffer),
              make_not_null(&modified_neighbor_solution_buffer), tensor,
              neighbor_linear_weight_, tensor_storage_index, mesh, element,
              neighbor_char_data);
          some_component_was_limited = true;
          some_component_was_limited_with_this_normal = true;
        }
      }
    };
    wrap_minmod_tci_and_simple_weno_impl(NewtonianEuler::Tags::UMinus{},
                                         make_not_null(&local_char_uminus));
    wrap_minmod_tci_and_simple_weno_impl(NewtonianEuler::Tags::U0<VolumeDim>{},
                                         make_not_null(&local_char_u0));
    wrap_minmod_tci_and_simple_weno_impl(NewtonianEuler::Tags::UPlus{},
                                         make_not_null(&local_char_uplus));
    // End SimpleWeno logic

    // Transform back to conserved variables. But skip the transformation if no
    // limiting occured with this normal.
    if (some_component_was_limited_with_this_normal) {
      NewtonianEulerWeno_detail::cons_tensors_from_char_tensors(
          make_not_null(&temp_cons_mass_density),
          make_not_null(&temp_cons_momentum_density),
          make_not_null(&temp_cons_energy_density), local_char_uminus,
          local_char_u0, local_char_uplus, right);
    } else {
      temp_cons_mass_density = *cons_mass_density;
      temp_cons_momentum_density = *cons_momentum_density;
      temp_cons_energy_density = *cons_energy_density;
    }

    // Add contribution from this particular choice of left/right matrices to
    // the running sum. Note: can't skip this step, because other normals might
    // contribute to limiting...
    get(accumulate_cons_mass_density) +=
        get(temp_cons_mass_density) / static_cast<double>(VolumeDim);
    for (size_t i = 0; i < VolumeDim; ++i) {
      accumulate_cons_momentum_density.get(i) +=
          temp_cons_momentum_density.get(i) / static_cast<double>(VolumeDim);
    }
    get(accumulate_cons_energy_density) +=
        get(temp_cons_energy_density) / static_cast<double>(VolumeDim);
  }  // for loop over dimensions

  *cons_mass_density = accumulate_cons_mass_density;
  *cons_momentum_density = accumulate_cons_momentum_density;
  *cons_energy_density = accumulate_cons_energy_density;
  return some_component_was_limited;
}

template <size_t LocalDim, size_t LocalThermoDim>
bool operator==(const SimpleWeno<LocalDim, LocalThermoDim>& lhs,
                const SimpleWeno<LocalDim, LocalThermoDim>& rhs) noexcept {
  return lhs.neighbor_linear_weight_ == rhs.neighbor_linear_weight_ and
         lhs.tvb_constant_ == rhs.tvb_constant_ and
         lhs.disable_for_debugging_ == rhs.disable_for_debugging_;
}

template <size_t VolumeDim, size_t ThermodynamicDim>
bool operator!=(const SimpleWeno<VolumeDim, ThermodynamicDim>& lhs,
                const SimpleWeno<VolumeDim, ThermodynamicDim>& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Limiters

// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Limiters/Weno.hpp"

#include <array>
#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <unordered_map>
#include <utility>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Element.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodTci.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/SimpleWenoImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Weno.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoType.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/CharacteristicHelpers.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/Flattener.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/WenoType.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace {
Limiters::WenoType weno_type_from_newtonian_euler_weno_type(
    const NewtonianEuler::Limiters::WenoType in) noexcept {
  switch (in) {
    case NewtonianEuler::Limiters::WenoType::CharacteristicSimpleWeno:
      return Limiters::WenoType::SimpleWeno;
    case NewtonianEuler::Limiters::WenoType::ConservativeHweno:
      return Limiters::WenoType::Hweno;
    case NewtonianEuler::Limiters::WenoType::ConservativeSimpleWeno:
      return Limiters::WenoType::SimpleWeno;
    default:
      // TODO(FH) clean this up... need an uninitialized value maybe?
      ERROR("bad newtonian euler weno type");
  }
}

bool acts_on_conserved_variables(
    const NewtonianEuler::Limiters::WenoType in) noexcept {
  return (in == NewtonianEuler::Limiters::WenoType::ConservativeHweno or
          in == NewtonianEuler::Limiters::WenoType::ConservativeSimpleWeno);
}

template <size_t VolumeDim, size_t ThermodynamicDim>
bool characteristic_simple_weno_impl(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const double tvb_constant, const double neighbor_linear_weight,
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        typename NewtonianEuler::Limiters::Weno<VolumeDim>::PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data) noexcept {
  // Cellwise means, used in computing the cons/char transformations
  const auto mean_density =
      Scalar<double>{mean_value(get(*mass_density_cons), mesh)};
  const auto mean_momentum = [&momentum_density, &mesh]() noexcept {
    tnsr::I<double, VolumeDim> result{};
    for (size_t i = 0; i < VolumeDim; ++i) {
      result.get(i) = mean_value(momentum_density->get(i), mesh);
    }
    return result;
  }();
  const auto mean_energy =
      Scalar<double>{mean_value(get(*energy_density), mesh)};

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
  auto& temp_mass_density_cons = get<::Tags::TempScalar<0>>(temp_buffer);
  auto& temp_momentum_density = get<::Tags::TempI<0, VolumeDim>>(temp_buffer);
  auto& temp_energy_density = get<::Tags::TempScalar<1>>(temp_buffer);
  auto& accumulate_mass_density_cons = get<::Tags::TempScalar<2>>(temp_buffer);
  auto& accumulate_momentum_density =
      get<::Tags::TempI<1, VolumeDim>>(temp_buffer);
  auto& accumulate_energy_density = get<::Tags::TempScalar<3>>(temp_buffer);

  // Initialize the accumulating tensors
  get(accumulate_mass_density_cons) = 0.;
  for (size_t i = 0; i < VolumeDim; ++i) {
    accumulate_momentum_density.get(i) = 0.;
  }
  get(accumulate_energy_density) = 0.;

  // Storage for transforming neighbor_data into char variables
  using CharWenoType =
      Limiters::Weno<VolumeDim, tmpl::list<NewtonianEuler::Tags::UMinus,
                                           NewtonianEuler::Tags::U0<VolumeDim>,
                                           NewtonianEuler::Tags::UPlus>>;
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
  Limiters::Minmod_detail::BufferWrapper<VolumeDim> tci_buffer(mesh);
  const auto effective_neighbor_sizes =
      Limiters::Minmod_detail::compute_effective_neighbor_sizes(element,
                                                                neighbor_data);

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

  // Apply limiter to chars, for chars w.r.t. each direction
  for (size_t d = 0; d < VolumeDim; ++d) {
    const auto normal = [&d]() noexcept {
      auto normal_components = make_array<VolumeDim>(0.);
      normal_components[d] = 1.;
      return tnsr::i<double, VolumeDim>(normal_components);
    }();
    const auto right_and_left = NewtonianEuler::Limiters::compute_eigenvectors(
        mean_density, mean_momentum, mean_energy, equation_of_state, normal);
    const auto& right = right_and_left.first;
    const auto& left = right_and_left.second;

    // Transform all field data to char vars:
    NewtonianEuler::Limiters::char_tensors_from_cons_tensors(
        make_not_null(&local_char_uminus), make_not_null(&local_char_u0),
        make_not_null(&local_char_uplus), *mass_density_cons, *momentum_density,
        *energy_density, left);
    for (const auto& kv : neighbor_data) {
      const auto& key = kv.first;
      const auto& data = kv.second;
      NewtonianEuler::Limiters::char_vars_from_cons_vars(
          make_not_null(&(neighbor_char_data[key].volume_data)),
          data.volume_data, left);
      NewtonianEuler::Limiters::char_means_from_cons_means(
          make_not_null(&(neighbor_char_data[key].means)), data.means, left);
    }

    // Begin SimpleWENO logic
    bool some_component_was_limited_with_this_normal = false;
    const auto wrap_minmod_tci_and_simple_weno_impl =
        [&some_component_was_limited,
         &some_component_was_limited_with_this_normal, &tci_buffer,
         &interpolator_buffer, &modified_neighbor_solution_buffer,
         &tvb_constant, &neighbor_linear_weight, &mesh, &element, &element_size,
         &neighbor_char_data,
         &effective_neighbor_sizes](auto tag, const auto tensor) noexcept {
          for (size_t tensor_storage_index = 0;
               tensor_storage_index < tensor->size(); ++tensor_storage_index) {
            // Check TCI
            const auto effective_neighbor_means =
                Limiters::Minmod_detail::compute_effective_neighbor_means<
                    decltype(tag)>(tensor_storage_index, element,
                                   neighbor_char_data);
            const bool component_needs_limiting =
                Limiters::Tci::tvb_minmod_indicator(
                    make_not_null(&tci_buffer), tvb_constant,
                    (*tensor)[tensor_storage_index], mesh, element,
                    element_size, effective_neighbor_means,
                    effective_neighbor_sizes);

            if (component_needs_limiting) {
              if (modified_neighbor_solution_buffer.empty()) {
                // Allocate the neighbor solution buffers only if the limiter is
                // triggered. This reduces allocation when no limiting occurs.
                for (const auto& neighbor_and_data : neighbor_char_data) {
                  const auto& neighbor = neighbor_and_data.first;
                  modified_neighbor_solution_buffer.insert(std::make_pair(
                      neighbor, DataVector(mesh.number_of_grid_points())));
                }
              }
              Limiters::Weno_detail::simple_weno_impl<decltype(tag)>(
                  make_not_null(&interpolator_buffer),
                  make_not_null(&modified_neighbor_solution_buffer), tensor,
                  neighbor_linear_weight, tensor_storage_index, mesh, element,
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
      NewtonianEuler::Limiters::cons_tensors_from_char_tensors(
          make_not_null(&temp_mass_density_cons),
          make_not_null(&temp_momentum_density),
          make_not_null(&temp_energy_density), local_char_uminus, local_char_u0,
          local_char_uplus, right);
    } else {
      temp_mass_density_cons = *mass_density_cons;
      temp_momentum_density = *momentum_density;
      temp_energy_density = *energy_density;
    }

    // Add contribution from this particular choice of left/right matrices to
    // the running sum. Note: can't skip this step, because other normals might
    // contribute to limiting...
    get(accumulate_mass_density_cons) +=
        get(temp_mass_density_cons) / static_cast<double>(VolumeDim);
    for (size_t i = 0; i < VolumeDim; ++i) {
      accumulate_momentum_density.get(i) +=
          temp_momentum_density.get(i) / static_cast<double>(VolumeDim);
    }
    get(accumulate_energy_density) +=
        get(temp_energy_density) / static_cast<double>(VolumeDim);
  }  // for loop over dimensions

  *mass_density_cons = accumulate_mass_density_cons;
  *momentum_density = accumulate_momentum_density;
  *energy_density = accumulate_energy_density;
  return some_component_was_limited;
}

}  // namespace

namespace NewtonianEuler {
namespace Limiters {

template <size_t VolumeDim>
Weno<VolumeDim>::Weno(const WenoType weno_type,
                      const double neighbor_linear_weight,
                      const double tvb_constant, const bool apply_flattener,
                      const bool disable_for_debugging) noexcept
    : weno_type_(weno_type),
      neighbor_linear_weight_(neighbor_linear_weight),
      tvb_constant_(tvb_constant),
      apply_flattener_(apply_flattener),
      disable_for_debugging_(disable_for_debugging) {
  ASSERT(tvb_constant >= 0.0, "The TVB constant must be non-negative.");
}

template <size_t VolumeDim>
// NOLINTNEXTLINE(google-runtime-references)
void Weno<VolumeDim>::pup(PUP::er& p) noexcept {
  p | weno_type_;
  p | neighbor_linear_weight_;
  p | tvb_constant_;
  p | apply_flattener_;
  p | disable_for_debugging_;
}

template <size_t VolumeDim>
void Weno<VolumeDim>::package_data(
    const gsl::not_null<PackagedData*> packaged_data,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, VolumeDim>& momentum_density,
    const Scalar<DataVector>& energy_density, const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const OrientationMap<VolumeDim>& orientation_map) const noexcept {
  // Convert NewtonianEuler::Limiters::WenoType -> Limiters::WenoType
  const ::Limiters::WenoType generic_weno_type =
      weno_type_from_newtonian_euler_weno_type(weno_type_);
  const ConservativeVarsWeno weno(generic_weno_type, neighbor_linear_weight_,
                                  tvb_constant_, disable_for_debugging_);
  weno.package_data(packaged_data, mass_density_cons, momentum_density,
                    energy_density, mesh, element_size, orientation_map);
}

template <size_t VolumeDim>
template <size_t ThermodynamicDim>
bool Weno<VolumeDim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const gsl::not_null<Scalar<DataVector>*> limiter_diagnostics,
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

  // Checks for the post-timestep, pre-limiter NewtonianEuler state, e.g.:
  const double mean_density = mean_value(get(*mass_density_cons), mesh);
  ASSERT(mean_density > 0.0,
         "Positivity was violated on a cell-average level.");
  if (ThermodynamicDim == 2) {
    const double mean_energy = mean_value(get(*energy_density), mesh);
    ASSERT(mean_energy > 0.0,
           "Positivity was violated on a cell-average level.");
  }
  // End pre-limiter checks

  bool limiter_activated = false;

  // Convert NewtonianEuler::Limiters::WenoType -> Limiters::WenoType
  const ::Limiters::WenoType generic_weno_type =
      weno_type_from_newtonian_euler_weno_type(weno_type_);

  if (acts_on_conserved_variables(weno_type_)) {
    const ConservativeVarsWeno weno(generic_weno_type, neighbor_linear_weight_,
                                    tvb_constant_, disable_for_debugging_);
    limiter_activated =
        weno(mass_density_cons, momentum_density, energy_density, mesh, element,
             element_size, neighbor_data);
  } else if (weno_type_ ==
             NewtonianEuler::Limiters::WenoType::CharacteristicSimpleWeno) {
    limiter_activated = characteristic_simple_weno_impl(
        mass_density_cons, momentum_density, energy_density, tvb_constant_,
        neighbor_linear_weight_, mesh, element, element_size, equation_of_state,
        neighbor_data);
  }

  size_t flattener_status = 0;
  if (apply_flattener_) {
    flattener_status =
        flatten_solution(mass_density_cons, momentum_density, energy_density,
                         mesh, equation_of_state);
  }

  // Checks for the post-limiter NewtonianEuler state, e.g.:
  ASSERT(min(get(*mass_density_cons)) > 0.0, "Bad density after limiting.");
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
    ASSERT(min(get(pressure)) > 0.0, "Bad energy after limiting.");
  }
  // End post-limiter checks

  get(*limiter_diagnostics) =
      (limiter_activated ? 1.0 : 0.0) + (flattener_status > 0 ? 1.0 : 0.0);
  return limiter_activated;
}

template <size_t LocalDim>
bool operator==(const Weno<LocalDim>& lhs, const Weno<LocalDim>& rhs) noexcept {
  return lhs.weno_type_ == rhs.weno_type_ and
         lhs.neighbor_linear_weight_ == rhs.neighbor_linear_weight_ and
         lhs.tvb_constant_ == rhs.tvb_constant_ and
         lhs.apply_flattener_ == rhs.apply_flattener_ and
         lhs.disable_for_debugging_ == rhs.disable_for_debugging_;
}

template <size_t VolumeDim>
bool operator!=(const Weno<VolumeDim>& lhs,
                const Weno<VolumeDim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data) template class Weno<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE

#define INSTANTIATE(_, data)                                                   \
  template bool Weno<DIM(data)>::operator()(                                   \
      const gsl::not_null<Scalar<DataVector>*>,                                \
      const gsl::not_null<tnsr::I<DataVector, DIM(data)>*>,                    \
      const gsl::not_null<Scalar<DataVector>*>,                                \
      const gsl::not_null<Scalar<DataVector>*>, const Mesh<DIM(data)>&,        \
      const Element<DIM(data)>&, const std::array<double, DIM(data)>&,         \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>&,       \
      const std::unordered_map<                                                \
          std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>, PackagedData, \
          boost::hash<                                                         \
              std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>>>&)        \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (1, 2))

#undef INSTANTIATE
#undef DIM
#undef THERMO_DIM

}  // namespace Limiters
}  // namespace NewtonianEuler

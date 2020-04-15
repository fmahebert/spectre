// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/NumericalFluxes/PositivityPreservingLaxFriedrichs.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FaceNormal.hpp"
#include "ErrorHandling/Error.hpp"
#include "Evolution/Systems/NewtonianEuler/Characteristics.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace NewtonianEuler {
namespace NumericalFluxes {

template <size_t Dim>
void PositivityPreservingLaxFriedrichs<Dim>::package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_n_dot_f_mass_density,
    const gsl::not_null<tnsr::I<DataVector, Dim>*>
        packaged_n_dot_f_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> packaged_n_dot_f_energy_density,
    const gsl::not_null<Scalar<DataVector>*> packaged_mass_density,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> packaged_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> packaged_energy_density,
    const gsl::not_null<Scalar<DataVector>*> packaged_max_abs_char_speed,
    const gsl::not_null<Scalar<DataVector>*> packaged_max_n_dot_f_mass_density,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_max_n_dot_f_energy_density,
    const Scalar<DataVector>& normal_dot_flux_mass_density,
    const tnsr::I<DataVector, Dim>& normal_dot_flux_momentum_density,
    const Scalar<DataVector>& normal_dot_flux_energy_density,
    const Scalar<DataVector>& mass_density,
    const tnsr::I<DataVector, Dim>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const db::const_item_type<char_speeds_tag>& characteristic_speeds,
    const Scalar<DataVector>& max_n_dot_f_mass_density,
    const Scalar<DataVector>& max_n_dot_f_energy_density) const noexcept {
  *packaged_n_dot_f_mass_density = normal_dot_flux_mass_density;
  *packaged_n_dot_f_momentum_density = normal_dot_flux_momentum_density;
  *packaged_n_dot_f_energy_density = normal_dot_flux_energy_density;

  *packaged_mass_density = mass_density;
  *packaged_momentum_density = momentum_density;
  *packaged_energy_density = energy_density;

  for (size_t s = 0; s < characteristic_speeds[0].size(); ++s) {
    double local_max_speed = 0.0;
    for (size_t u = 0; u < characteristic_speeds.size(); ++u) {
      local_max_speed = std::max(
          local_max_speed, std::abs(gsl::at(characteristic_speeds, u)[s]));
    }
    get(*packaged_max_abs_char_speed)[s] = local_max_speed;
  }

  *packaged_max_n_dot_f_mass_density = max_n_dot_f_mass_density;
  *packaged_max_n_dot_f_energy_density = max_n_dot_f_energy_density;
}

template <size_t Dim>
void PositivityPreservingLaxFriedrichs<Dim>::operator()(
    const gsl::not_null<Scalar<DataVector>*>
        normal_dot_numerical_flux_mass_density,
    const gsl::not_null<tnsr::I<DataVector, Dim>*>
        normal_dot_numerical_flux_momentum_density,
    const gsl::not_null<Scalar<DataVector>*>
        normal_dot_numerical_flux_energy_density,
    const Scalar<DataVector>& normal_dot_flux_mass_density_int,
    const tnsr::I<DataVector, Dim>& normal_dot_flux_momentum_density_int,
    const Scalar<DataVector>& normal_dot_flux_energy_density_int,
    const Scalar<DataVector>& mass_density_int,
    const tnsr::I<DataVector, Dim>& momentum_density_int,
    const Scalar<DataVector>& energy_density_int,
    const Scalar<DataVector>& max_abs_char_speed_int,
    const Scalar<DataVector>& max_n_dot_f_mass_density_int,
    const Scalar<DataVector>& /*max_n_dot_f_energy_density_int*/,
    const Scalar<DataVector>& minus_normal_dot_flux_mass_density_ext,
    const tnsr::I<DataVector, Dim>& minus_normal_dot_flux_momentum_density_ext,
    const Scalar<DataVector>& minus_normal_dot_flux_energy_density_ext,
    const Scalar<DataVector>& mass_density_ext,
    const tnsr::I<DataVector, Dim>& momentum_density_ext,
    const Scalar<DataVector>& energy_density_ext,
    const Scalar<DataVector>& max_abs_char_speed_ext,
    const Scalar<DataVector>& max_n_dot_f_mass_density_ext,
    const Scalar<DataVector>& /*max_n_dot_f_energy_density_ext*/) const
    noexcept {
  auto corrected_minus_normal_dot_flux_mass_density_ext =
      minus_normal_dot_flux_mass_density_ext;
  auto corrected_minus_normal_dot_flux_momentum_density_ext =
      minus_normal_dot_flux_momentum_density_ext;
  auto corrected_minus_normal_dot_flux_energy_density_ext =
      minus_normal_dot_flux_energy_density_ext;
  auto corrected_max_abs_char_speed_ext = max_abs_char_speed_ext;
  auto corrected_max_n_dot_f_mass_density_ext = max_n_dot_f_mass_density_ext;

  const bool small_rho_ext = max(abs(get(mass_density_ext))) < 1e-15;
  const bool small_e_ext = max(abs(get(energy_density_ext))) < 1e-15;
  if (small_rho_ext and small_e_ext) {
    get(corrected_minus_normal_dot_flux_mass_density_ext) = 0.;
    for (size_t i = 0; i < Dim; ++i) {
      corrected_minus_normal_dot_flux_momentum_density_ext.get(i) = 0.;
    }
    get(corrected_minus_normal_dot_flux_energy_density_ext) = 0.;
    get(corrected_max_abs_char_speed_ext) = 0.;
    // no need to place a limit on ingoing fluxes
    get(corrected_max_n_dot_f_mass_density_ext) = 1e99;
  }

  const Scalar<DataVector> max_abs_char_speed(DataVector(
      max(get(max_abs_char_speed_int), get(corrected_max_abs_char_speed_ext))));
  const auto assemble_numerical_flux =
      [&max_abs_char_speed](const auto n_dot_num_f, const auto& n_dot_f_in,
                            const auto& u_in, const auto& minus_n_dot_f_ex,
                            const auto& u_ex) noexcept {
        for (size_t i = 0; i < n_dot_num_f->size(); ++i) {
          (*n_dot_num_f)[i] =
              0.5 * (n_dot_f_in[i] - minus_n_dot_f_ex[i] +
                     get(max_abs_char_speed) * (u_in[i] - u_ex[i]));
        }
        return nullptr;
      };
  assemble_numerical_flux(normal_dot_numerical_flux_mass_density,
                          normal_dot_flux_mass_density_int, mass_density_int,
                          corrected_minus_normal_dot_flux_mass_density_ext,
                          mass_density_ext);
  assemble_numerical_flux(normal_dot_numerical_flux_momentum_density,
                          normal_dot_flux_momentum_density_int,
                          momentum_density_int,
                          corrected_minus_normal_dot_flux_momentum_density_ext,
                          momentum_density_ext);
  assemble_numerical_flux(
      normal_dot_numerical_flux_energy_density,
      normal_dot_flux_energy_density_int, energy_density_int,
      corrected_minus_normal_dot_flux_energy_density_ext, energy_density_ext);

  if (not enable_pp_checks_) {
    // For "vanilla" LF flux, exit here. For PP checks, continue...
    return;
  }

  ASSERT(Dim == 1 or Dim == 2 or Dim == 3, "no");
  size_t pts_per_dim = 0;
  if (Dim == 1) {
    const size_t mesh_num_pts = get(mass_density_int).size();
    ASSERT(mesh_num_pts == 1, "no");
    pts_per_dim = 1;
  } else if (Dim == 2) {
    const size_t mesh_num_pts = get(mass_density_int).size();
    ASSERT(mesh_num_pts > 1, "no");
    pts_per_dim = mesh_num_pts;
  } else if (Dim == 3) {
    const size_t mesh_num_pts = get(mass_density_int).size();
    switch (mesh_num_pts) {
      case 4:
        pts_per_dim = 2;
        break;
      case 9:
        pts_per_dim = 3;
        break;
      case 16:
        pts_per_dim = 4;
        break;
      default:
        ASSERT(false, "no");
    }
  }
  Mesh<Dim - 1> mesh(pts_per_dim, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);

  // now constrain fluxes
  const double mean_int =
      mean_value(get(normal_dot_flux_mass_density_int), mesh);
  const double mean_ext =
      -mean_value(get(corrected_minus_normal_dot_flux_mass_density_ext), mesh);
  const double mean_num =
      mean_value(get(*normal_dot_numerical_flux_mass_density), mesh);
  if (mean_num >= 0.0) {
    // outgoing, so check F* isn't greater than allowed from internal side
    const double max = get(max_n_dot_f_mass_density_int)[0];
    if (mean_num > max) {
      if (mean_int < max) {
        *normal_dot_numerical_flux_mass_density =
            normal_dot_flux_mass_density_int;
        *normal_dot_numerical_flux_momentum_density =
            normal_dot_flux_momentum_density_int;
        *normal_dot_numerical_flux_energy_density =
            normal_dot_flux_energy_density_int;
      } else if (mean_ext < max) {
        *normal_dot_numerical_flux_mass_density =
            corrected_minus_normal_dot_flux_mass_density_ext;
        *normal_dot_numerical_flux_momentum_density =
            corrected_minus_normal_dot_flux_momentum_density_ext;
        *normal_dot_numerical_flux_energy_density =
            corrected_minus_normal_dot_flux_energy_density_ext;
        get(*normal_dot_numerical_flux_mass_density) *= -1.0;
        for (size_t i = 0; i < Dim; ++i) {
          normal_dot_numerical_flux_momentum_density->get(i) *= -1.0;
        }
        get(*normal_dot_numerical_flux_energy_density) *= -1.0;
        //} else {
        //  const double scale = 0.1 * max / mean_num;
        //  get(*normal_dot_numerical_flux_mass_density) *= scale;
        //  for (size_t i = 0; i < Dim; ++i) {
        //    normal_dot_numerical_flux_momentum_density->get(i) *= scale;
        //  }
        //  get(*normal_dot_numerical_flux_energy_density) *= scale;
        //  std::cout << "bad outgoing data at timestep, with\n"
        //    "  mean F_int = " << mean_int << "\n"
        //    "  mean F_ext = " << mean_ext << "\n"
        //    "  mean u_int = " << mean_value(get(mass_density_int), mesh) <<
        //    "\n" "  mean u_ext = " << mean_value(get(mass_density_ext), mesh)
        //    << "\n" "  mean F_num = " << mean_num << "\n" "  mean F_max = " <<
        //    max << "\n" "  => RESCALED FLUXES\n";
      }
    }
  } else {
    // ingoing, so check F* isn't greater than allowed by neighbor
    const double max = get(max_n_dot_f_mass_density_ext)[0];
    // where "greather than allowed" = more negative than the max
    if (mean_num < -max) {
      if (mean_ext > -max) {
        *normal_dot_numerical_flux_mass_density =
            corrected_minus_normal_dot_flux_mass_density_ext;
        *normal_dot_numerical_flux_momentum_density =
            corrected_minus_normal_dot_flux_momentum_density_ext;
        *normal_dot_numerical_flux_energy_density =
            corrected_minus_normal_dot_flux_energy_density_ext;
        get(*normal_dot_numerical_flux_mass_density) *= -1.0;
        for (size_t i = 0; i < Dim; ++i) {
          normal_dot_numerical_flux_momentum_density->get(i) *= -1.0;
        }
        get(*normal_dot_numerical_flux_energy_density) *= -1.0;
      } else if (mean_int > -max) {
        *normal_dot_numerical_flux_mass_density =
            normal_dot_flux_mass_density_int;
        *normal_dot_numerical_flux_momentum_density =
            normal_dot_flux_momentum_density_int;
        *normal_dot_numerical_flux_energy_density =
            normal_dot_flux_energy_density_int;
        //} else {
        //  const double scale = 0.1 * fabs(max / mean_num);
        //  get(*normal_dot_numerical_flux_mass_density) *= scale;
        //  for (size_t i = 0; i < Dim; ++i) {
        //    normal_dot_numerical_flux_momentum_density->get(i) *= scale;
        //  }
        //  get(*normal_dot_numerical_flux_energy_density) *= scale;
        //  std::cout << "bad ingoing data at timestep, with\n"
        //    "  mean F_int = " << mean_int << "\n"
        //    "  mean F_ext = " << mean_ext << "\n"
        //    "  mean u_int = " << mean_value(get(mass_density_int), mesh) <<
        //    "\n" "  mean u_ext = " << mean_value(get(mass_density_ext), mesh)
        //    << "\n" "  mean F_num = " << mean_num << "\n" "  mean F_max = " <<
        //    max << "\n" "  => RESCALED FLUXES\n";
      }
    }
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) \
  template struct PositivityPreservingLaxFriedrichs<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace NumericalFluxes
}  // namespace NewtonianEuler

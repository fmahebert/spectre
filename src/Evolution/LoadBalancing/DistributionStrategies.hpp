// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/GlobalCache.hpp"

#include "Parallel/Printf.hpp"

namespace Lb {
namespace Distribution {

struct RoundRobin;
struct KnownEvenOptimal;
struct KnownMaxFragmented;

// create a parent abstract class for this, template the class on metavariables
// so we don't need it in the processor decision function.

// TODO move to cpp

struct DistributionStrategy : public PUP::able {
  using creatable_classes =
      tmpl::list<RoundRobin, KnownEvenOptimal, KnownMaxFragmented>;

  WRAPPED_PUPable_abstract(DistributionStrategy);

  virtual std::unique_ptr<DistributionStrategy> get_clone() const noexcept = 0;

  virtual int which_proc(
      const Domain<1>& domain,
      const std::vector<std::array<size_t, 1>>& initial_refinement_levels,
      const size_t number_of_procs, const Element<1>& element,
      const ElementId<1>& element_id,
      const ElementMap<1, Frame::Inertial>& element_map) noexcept = 0;

  virtual int which_proc(
      const Domain<2>& domain,
      const std::vector<std::array<size_t, 2>>& initial_refinement_levels,
      const size_t number_of_procs, const Element<2>& element,
      const ElementId<2>& element_id,
      const ElementMap<2, Frame::Inertial>& element_map) noexcept = 0;

  virtual int which_proc(
      const Domain<3>& domain,
      const std::vector<std::array<size_t, 3>>& initial_refinement_levels,
      const size_t number_of_procs, const Element<3>& element,
      const ElementId<3>& element_id,
      const ElementMap<3, Frame::Inertial>& element_map) noexcept = 0;

};

// we will need to do a bit of tricking around to figure out where each of the
// elements are in the domain

struct RoundRobin : public DistributionStrategy {
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "Default standard load-balancing; just assigns the components to "
      "processors evenly without considering communication patterns or load."};

  WRAPPED_PUPable_decl_template(RoundRobin);  // NOLINT
  RoundRobin() = default;
  explicit RoundRobin(CkMigrateMessage* /*unused*/) noexcept {}

  std::unique_ptr<DistributionStrategy> get_clone() const noexcept override {
    return std::make_unique<RoundRobin>(*this);
  }

  int which_proc(
      const Domain<1>& domain,
      const std::vector<std::array<size_t, 1>>& initial_refinement_levels,
      const size_t number_of_procs, const Element<1>& element,
      const ElementId<1>& element_id,
      const ElementMap<1, Frame::Inertial>& element_map) noexcept override {
    return which_proc_impl(domain, initial_refinement_levels, number_of_procs,
                           element, element_id, element_map);
  }

  int which_proc(
      const Domain<2>& domain,
      const std::vector<std::array<size_t, 2>>& initial_refinement_levels,
      const size_t number_of_procs, const Element<2>& element,
      const ElementId<2>& element_id,
      const ElementMap<2, Frame::Inertial>& element_map) noexcept override {
    return which_proc_impl(domain, initial_refinement_levels, number_of_procs,
                           element, element_id, element_map);
  }

  int which_proc(
      const Domain<3>& domain,
      const std::vector<std::array<size_t, 3>>& initial_refinement_levels,
      const size_t number_of_procs, const Element<3>& element,
      const ElementId<3>& element_id,
      const ElementMap<3, Frame::Inertial>& element_map) noexcept override {
    return which_proc_impl(domain, initial_refinement_levels, number_of_procs,
                           element, element_id, element_map);
  }

  template <size_t Dim>
  int which_proc_impl(
      const Domain<Dim>& /*domain*/,
      const std::vector<std::array<size_t, Dim>>& /*initial_refinement_levels*/,
      const size_t number_of_procs, const Element<Dim>& /*element*/,
      const ElementId<Dim>& /*element_id*/,
      const ElementMap<Dim, Frame::Inertial>& /*element_map*/) noexcept {
    return counter_++ % static_cast<int>(number_of_procs);
  }

  void pup(PUP::er& p) noexcept override { p | counter_; }

 private:
  int counter_ = 0;
};

struct KnownEvenOptimal : public DistributionStrategy {
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "A deliberately 'best-case' distribution strategy to help test the "
      "extent to which load balancing can optimize"};

  WRAPPED_PUPable_decl_template(KnownEvenOptimal);  // NOLINT
  KnownEvenOptimal() = default;
  explicit KnownEvenOptimal(CkMigrateMessage* /*unused*/) noexcept {}

  std::unique_ptr<DistributionStrategy> get_clone() const noexcept override {
    return std::make_unique<KnownEvenOptimal>(*this);
  }

  int which_proc(
      const Domain<1>& domain,
      const std::vector<std::array<size_t, 1>>& /*initial_refinement_levels*/,
      const size_t number_of_procs, const Element<1>& /*element*/,
      const ElementId<1>& /*element_id*/,
      const ElementMap<1, Frame::Inertial>& element_map) noexcept override {
    const auto& blocks = domain.blocks();
    if (blocks.size() > 1) {
      ERROR(
          "optimal even distribution is only supported for a single 1D block");
    }
    double x_coord =
        element_map(tnsr::I<double, 1, Frame::Logical>{{{0.0}}})[0];
    Parallel::printf("sending %f to %d\n", x_coord,
                     static_cast<int>(0.5 * (x_coord + 1.0) * number_of_procs));
    return static_cast<int>(0.5 * (x_coord + 1.0) * number_of_procs);
  }

  int which_proc(
      const Domain<2>& /*domain*/,
      const std::vector<std::array<size_t, 2>>& /*initial_refinement_levels*/,
      const size_t /*number_of_procs*/, const Element<2>& /*element*/,
      const ElementId<2>& /*element_id*/,
      const ElementMap<2, Frame::Inertial>& /*element_map*/) noexcept override {
    ERROR("optimal even distribution is only supported for a single 1D block");
    return 0;
  }

  int which_proc(
      const Domain<3>& /*domain*/,
      const std::vector<std::array<size_t, 3>>& /*initial_refinement_levels*/,
      const size_t /*number_of_procs*/, const Element<3>& /*element*/,
      const ElementId<3>& /*element_id*/,
      const ElementMap<3, Frame::Inertial>& /*element_map*/) noexcept override {
    ERROR("optimal even distribution is only supported for a single 1D block");
    return 0;
  }
  void pup(PUP::er& /*p*/) noexcept override {}
};

struct KnownMaxFragmented : public DistributionStrategy {
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "A deliberately 'worst-case' distribution strategy to help test the "
      "extent to which load balancing can optimize"};

  WRAPPED_PUPable_decl_template(KnownMaxFragmented);  // NOLINT
  KnownMaxFragmented() = default;
  explicit KnownMaxFragmented(CkMigrateMessage* /*unused*/) noexcept {}

  std::unique_ptr<DistributionStrategy> get_clone() const noexcept override {
    return std::make_unique<KnownMaxFragmented>(*this);
  }


  int which_proc(
      const Domain<1>& domain,
      const std::vector<std::array<size_t, 1>>& initial_refinement_levels,
      const size_t number_of_procs, const Element<1>& /*element*/,
      const ElementId<1>& /*element_id*/,
      const ElementMap<1, Frame::Inertial>& element_map) noexcept override {
    const auto& blocks = domain.blocks();
    if (blocks.size() > 1) {
      ERROR(
          "optimally fragmented distribution is only supported for a single 1D "
          "block");
    }
    double x_coord = element_map(tnsr::I<double, 1, Frame::Logical>{{0.0}})[0];
    Parallel::printf(
        "sending %f to %d\n", x_coord,
        static_cast<int>(0.5 * (x_coord + 1.0) *
                         pow(2.0, initial_refinement_levels[0][0])) %
            static_cast<int>(number_of_procs));
    return static_cast<int>(0.5 * (x_coord + 1.0) *
                            pow(2.0, initial_refinement_levels[0][0])) %
           static_cast<int>(number_of_procs);
  }

  int which_proc(
      const Domain<2>& /*domain*/,
      const std::vector<std::array<size_t, 2>>& /*initial_refinement_levels*/,
      const size_t /*number_of_procs*/, const Element<2>& /*element*/,
      const ElementId<2>& /*element_id*/,
      const ElementMap<2, Frame::Inertial>& /*element_map*/) noexcept override {
    ERROR(
        "optimally fragmented distribution is only supported for a single 1D "
        "block");
    return 0;
  }

  int which_proc(
      const Domain<3>& /*domain*/,
      const std::vector<std::array<size_t, 3>>& /*initial_refinement_levels*/,
      const size_t /*number_of_procs*/, const Element<3>& /*element*/,
      const ElementId<3>& /*element_id*/,
      const ElementMap<3, Frame::Inertial>& /*element_map*/) noexcept override {
    ERROR(
        "optimally fragmented distribution is only supported for a single 1D "
        "block");
    return 0;
  }

  void pup(PUP::er& /*p*/) noexcept override {}
};

}  // namespace Distribution
}  // namespace Lb

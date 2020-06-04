// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <pup.h>
#include <utility>
#include <vector>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

namespace Lb {
template <typename TriggerRegistrars>
class SpecifiedStepTrigger;

template <typename TriggerRegistrars>
class EveryNStepsTrigger;

template <typename TriggerRegistrars>
class SpecifiedWallTimeTrigger;
}

namespace Triggers::Registrars {
using SpecifiedStepTrigger = Registration::Registrar<Lb::SpecifiedStepTrigger>;
using EveryNStepsTrigger = Registration::Registrar<Lb::EveryNStepsTrigger>;
using SpecifiedWallTimeTrigger =
    Registration::Registrar<Lb::SpecifiedWallTimeTrigger>;
}

namespace Lb {

template <typename TriggerRegistrars =
              tmpl::list<Triggers::Registrars::SpecifiedStepTrigger>>
class SpecifiedStepTrigger : public Trigger<TriggerRegistrars> {
 public:
  SpecifiedStepTrigger() = default;
  explicit SpecifiedStepTrigger(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(SpecifiedStepTrigger);  // NOLINT

  struct Steps{
    using type = std::vector<size_t>;
    static constexpr Options::String help{"steps to trigger at"};
  };

  static constexpr Options::String help{"Trigger at specific steps."};
  using options = tmpl::list<Steps>;

  explicit SpecifiedStepTrigger(const std::vector<size_t>& steps) noexcept {
    for(const auto step: steps) {
      steps_.insert(step);
    }
  }

  using argument_tags = tmpl::list<Tags::StepNumber>;

  bool operator()(const size_t current_step) const noexcept {
    return steps_.count(current_step) == 1;
  }

  void pup(PUP::er& p) noexcept override { p | steps_; }  // NOLINT

 private:
  std::set<size_t> steps_;
};

template <typename TriggerRegistrars =
              tmpl::list<Triggers::Registrars::SpecifiedStepTrigger>>
class EveryNStepsTrigger : public Trigger<TriggerRegistrars> {
 public:
  EveryNStepsTrigger() = default;
  explicit EveryNStepsTrigger(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(EveryNStepsTrigger);  // NOLINT

  struct Period{
    using type = size_t;
    static constexpr Options::String help{"Number of steps between triggers"};
  };

  struct Offset{
    using type = size_t;
    static constexpr Options::String help{"The first step to trigger on"};
  };

  static constexpr Options::String help{"Trigger periodically on steps."};
  using options = tmpl::list<Period, Offset>;

  explicit EveryNStepsTrigger(const size_t period, const size_t offset) noexcept
      : period_{period}, offset_{offset} {}

  using argument_tags = tmpl::list<Tags::StepNumber>;

  bool operator()(const size_t current_step) const noexcept {
    return current_step > offset_ and (current_step - offset_) % period_ == 0;
  }

  void pup(PUP::er& p) noexcept override { // NOLINT
    p | period_;
    p | offset_;
  }

 private:
  size_t period_;
  size_t offset_;
};


template <typename TriggerRegistrars =
              tmpl::list<Triggers::Registrars::SpecifiedWallTimeTrigger>>
class SpecifiedWallTimeTrigger : public Trigger<TriggerRegistrars> {
 public:
  SpecifiedWallTimeTrigger() = default;
  explicit SpecifiedWallTimeTrigger(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(SpecifiedWallTimeTrigger);  // NOLINT

  struct WallTimes {
    using type = std::vector<double>;
    static constexpr Options::String help{"times to trigger at"};
  };

  static constexpr Options::String help{"Trigger at specified wall times."};
  using options = tmpl::list<WallTimes>;

  explicit SpecifiedWallTimeTrigger(const std::vector<double>& times) noexcept
      : times_{std::move(times)} {
    std::sort(times_.begin(), times_.end());
  }

  using argument_tags = tmpl::list<Tags::GraphDumpLabel>;

  // note this feature means that this trigger is currently not interoperable
  // with other triggers and can only be used with the charm projections graph
  // dump functionality. However, because charm specifically uses wall time to
  // initiate load-balancing in certain run modes, it is useful to be able to
  // dump the graph either just before or just after the load-balancing takes
  // place
  bool operator()(const size_t current_graph_index) const noexcept {
    return current_graph_index < times_.size() and
           Parallel::wall_time() > times_[current_graph_index];
  }

  void pup(PUP::er& p) noexcept override { p | times_; }

 private:
  std::vector<double> times_;
};

template <typename TriggerRegistrars>
PUP::able::PUP_ID SpecifiedStepTrigger<TriggerRegistrars>::my_PUP_ID = 0;  // NOLINT
template <typename TriggerRegistrars>
PUP::able::PUP_ID EveryNStepsTrigger<TriggerRegistrars>::my_PUP_ID =
    0;  // NOLINT
template <typename TriggerRegistrars>
PUP::able::PUP_ID SpecifiedWallTimeTrigger<TriggerRegistrars>::my_PUP_ID =
    0;  // NOLINT
}  // namespace Lb

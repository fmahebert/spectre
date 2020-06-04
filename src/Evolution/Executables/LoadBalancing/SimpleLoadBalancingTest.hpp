// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Evolution/LoadBalancing/DistributionStrategies.hpp"
#include "Evolution/LoadBalancing/LoadBalancingTestArray.hpp"
#include "Evolution/LoadBalancing/StepTriggers.hpp"
#include "Evolution/LoadBalancing/Tags.hpp"
#include "Parallel/Actions/ManagePhaseControl.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

template <size_t Dim, bool UseAtSync>
struct EvolutionMetavars {
  static constexpr size_t volume_dim = Dim;
  using temporal_id = Lb::Tags::StepNumber;
  static constexpr bool use_at_sync = UseAtSync;

  using triggers = tmpl::list<Triggers::Registrars::SpecifiedStepTrigger,
                              Triggers::Registrars::EveryNStepsTrigger>;

  using graph_dump_triggers =
      tmpl::list<Triggers::Registrars::SpecifiedStepTrigger,
                 Triggers::Registrars::EveryNStepsTrigger,
                 Triggers::Registrars::SpecifiedWallTimeTrigger>;

  using global_sync_triggers =
      tmpl::list<Triggers::Registrars::SpecifiedStepTrigger,
                 Triggers::Registrars::EveryNStepsTrigger>;

  enum class Phase {
    Initialization,
    LoadBalancing,
    Evolve,
    Exit
  };

  static std::string phase_name(const Phase phase) noexcept {
    switch (phase) {
      case Phase::LoadBalancing:
        return "LoadBalancing";
      case Phase::Evolve:
        return "Evolve";
      default:
        ERROR("No name for given phase");
    }
  }

  using global_sync_phases =
      tmpl::list<std::integral_constant<Phase, Phase::LoadBalancing>>;

  static void global_startup_routines() noexcept { return; }



  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<
          EvolutionMetavars>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::Evolve;
      case Phase::Evolve:
        return Phase::Exit;
      case Phase::Exit:
        ERROR(
            "Should never call determine_next_phase with the current phase "
            "being 'Exit'");
      default:
        ERROR(
            "Unknown type of phase. Did you static_cast<Phase> an integral "
            "value?");
    }
  }

  static constexpr Options::String help{
      "Run a fast strawman version of a domain evolution for testing "
      "load-balancing strategies according to parameters of the elements"};

  // Probably want a triggers interface to specify when/where to graph
  // dump. That will be the desired interface for the main code I think.

  // the idea is that the diagnostic dump will occur only when a trigger is
  // satisfied (so the trigger will be retrieved from the const global cache and
  // evaluated, but there will be no associated event.)
  using component_list =
      tmpl::list<Lb::LoadBalancingTestArray<EvolutionMetavars>>;
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &domain::creators::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<
        Lb::Distribution::DistributionStrategy>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<metavariables::triggers>>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<metavariables::graph_dump_triggers>>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<metavariables::global_sync_triggers>>};

static const std::vector<void (*)()> charm_init_proc_funcs{
  &enable_floating_point_exceptions};


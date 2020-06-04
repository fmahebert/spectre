// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/LoadBalancing/DistributionStrategies.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "Options/Options.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Serialize.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"

namespace Lb {
namespace OptionTags {

struct TestLoadBalancing {
  static constexpr Options::String help{
      "Options for the load-balancer test array"};
};

struct ExecutionLoad {
  using type = size_t;
  static constexpr Options::String help{
      "Number of times to iterate the test load function"};
  using group = TestLoadBalancing;
};

struct InternalStorageSize {
  using type = size_t;
  static constexpr Options::String help{"Multiplier for internal storage cost"};
  using group = TestLoadBalancing;
};

struct CommunicationSize {
  using type = size_t;
  static constexpr Options::String help{
      "Multiplier for amount of data to send between elements"};
  using group = TestLoadBalancing;
};

struct NumberOfSteps {
  using type = size_t;
  static constexpr Options::String help{"number of steps to run"};
  using group = TestLoadBalancing;
};

struct DistributionStrategy {
  using type = std::unique_ptr<Distribution::DistributionStrategy>;
  static constexpr Options::String help{
      "Strategy for initial element distribution"};
  using group = TestLoadBalancing;
};

template <typename TriggerRegistrars>
struct GraphDumpTrigger {
  using type = std::unique_ptr<Trigger<TriggerRegistrars>>;
  static constexpr Options::String help =
      "condition for which to dump graph to projections";
};
}  // namespace OptionTags

namespace Tags {

struct ExecutionLoad : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::ExecutionLoad>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(const size_t execution_load) noexcept {
    return execution_load;
  }
};

struct InternalStorageSize : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::InternalStorageSize>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(
      const size_t internal_storage_size) noexcept {
    return internal_storage_size;
  }
};

struct CommunicationSize : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::CommunicationSize>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(
      const size_t communication_size) noexcept {
    return communication_size;
  }
};

struct NumberOfSteps : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::NumberOfSteps>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(const size_t number_of_steps) noexcept {
    return number_of_steps;
  }
};

struct StepNumber :db::SimpleTag {
  using type = size_t;
};

struct GraphDumpLabel : db::SimpleTag {
  using type = size_t;
};

template <size_t Dim>
struct NeighborData : db::SimpleTag {
  using type =
      FixedHashMap<maximum_number_of_neighbors(Dim), dg::MortarId<Dim>,
                   std::vector<DataVector>, boost::hash<dg::MortarId<Dim>>>;
};

struct InternalStorage : db::SimpleTag {
  using type = std::vector<DataVector>;
};

struct DistributionStrategy : db::SimpleTag {
  using type =
      std::unique_ptr<Distribution::DistributionStrategy>;
  using option_tags = tmpl::list<OptionTags::DistributionStrategy>;

  static constexpr bool pass_metavariables = false;
  static std::unique_ptr<Distribution::DistributionStrategy>
  create_from_options(const std::unique_ptr<Distribution::DistributionStrategy>&
                          distribution_strategy) {
    return distribution_strategy->get_clone();
  }
};

template <typename TriggerRegistrars>
struct GraphDumpTrigger : db::SimpleTag {
  using type = std::unique_ptr<Trigger<TriggerRegistrars>>;
  using option_tags =
      tmpl::list<OptionTags::GraphDumpTrigger<TriggerRegistrars>>;

  static constexpr bool pass_metavariables = false;
  static std::unique_ptr<Trigger<TriggerRegistrars>> create_from_options(
      const std::unique_ptr<Trigger<TriggerRegistrars>>& trigger) noexcept {
    // :(
    return deserialize<type>(serialize<type>(trigger).data());
  }
};

}  // namespace Tags

namespace ReceiveTags {

template <size_t Dim>
struct LoadBalancingCommunication
    : Parallel::InboxInserters::Map<LoadBalancingCommunication<Dim>> {
  using temporal_id = size_t;
  using type = std::unordered_map<
      temporal_id,
      FixedHashMap<maximum_number_of_neighbors(Dim), dg::MortarId<Dim>,
                   std::vector<DataVector>, boost::hash<dg::MortarId<Dim>>>>;
};

}  // namespace ReceiveTags
}  // namespace Lb

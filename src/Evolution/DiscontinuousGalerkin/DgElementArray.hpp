// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "AlgorithmArray.hpp"
#include "Domain/Block.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/OptionTags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Protocols.hpp"
#include "Evolution/Tags.hpp"
#include "IO/Importers/VolumeDataReader.hpp"
#include "IO/Importers/VolumeDataReaderActions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "ParallelAlgorithms/Actions/SetData.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

#include "Utilities/MakeString.hpp"

// Control the initial distribution of DgElementArray elements across the
// processors available
namespace OptionTags {
template <size_t VolumeDim>
struct DomainPartitionChunkSize {
  static std::string name() noexcept { return "DomainPartitionChunkSize"; }
  static constexpr Options::String help = "Domain partition chunk size";
  using type = std::array<size_t, VolumeDim>;
  using group = evolution::OptionTags::Group;
};
}  // namespace OptionTags
namespace Tags {
template <size_t VolumeDim>
struct DomainPartitionChunkSize : db::SimpleTag {
  using type = std::array<size_t, VolumeDim>;
  using option_tags =
      tmpl::list<::OptionTags::DomainPartitionChunkSize<VolumeDim>>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& partition) noexcept {
    return partition;
  }
};
}  // namespace Tags

/*!
 * \brief Don't try to import initial data from a data file
 *
 * \see `DgElementArray`
 */
struct ImportNoInitialData {};

/*!
 * \brief Import numeric initial data in the `ImportPhase`
 *
 * Requires the `InitialData` conforms to
 * `evolution::protocols::NumericInitialData`. The `InitialData::import_fields`
 * will be imported in the `ImportPhase` from the data file that is specified by
 * the options in the `evolution::OptionTags::NumericInitialData` group.
 *
 * \see `DgElementArray`
 */
template <typename PhaseType, PhaseType ImportPhase, typename InitialData>
struct ImportNumericInitialData {
  static_assert(tt::assert_conforms_to<
                InitialData, evolution::protocols::NumericInitialData>);
  using phase_type = PhaseType;
  static constexpr PhaseType import_phase = ImportPhase;
  using initial_data = InitialData;
};

namespace DgElementArray_detail {

template <typename DgElementArray, typename InitialData,
          typename ImportFields = typename InitialData::import_fields>
using read_element_data_action = importers::ThreadedActions::ReadVolumeData<
    evolution::OptionTags::NumericInitialData, ImportFields,
    ::Actions::SetData<ImportFields>, DgElementArray>;

template <typename DgElementArray, typename ImportInitialData>
struct import_numeric_data_cache_tags {
  using type = tmpl::list<>;
};

template <typename DgElementArray, typename PhaseType, PhaseType ImportPhase,
          typename InitialData>
struct import_numeric_data_cache_tags<
    DgElementArray,
    ImportNumericInitialData<PhaseType, ImportPhase, InitialData>> {
  using type =
      typename read_element_data_action<DgElementArray,
                                        InitialData>::const_global_cache_tags;
};

template <typename Metavariables, typename Component,
          typename check = std::void_t<>>
struct has_registration_list_impl : std::false_type {};

template <typename Metavariables, typename Component>
struct has_registration_list_impl<
    Metavariables, Component,
    std::void_t<
        typename Metavariables::template registration_list<Component>::type>>
    : std::true_type {};

template <typename Metavariables, typename Component>
constexpr bool has_registration_list =
    has_registration_list_impl<Metavariables, Component>::value;
}  // namespace DgElementArray_detail

/*!
 * \brief The parallel component responsible for managing the DG elements that
 * compose the computational domain
 *
 * This parallel component will perform the actions specified by the
 * `PhaseDepActionList`.
 *
 * This component also supports loading initial for its elements from a file.
 * To do so, set the `ImportInitialData` template parameter to
 * `ImportNumericInitialData`. See the documentation of the
 * `ImportNumericInitialData` for details.
 */
template <class Metavariables, class PhaseDepActionList,
          class ImportInitialData = ImportNoInitialData>
struct DgElementArray {
  static constexpr size_t volume_dim = Metavariables::volume_dim;

  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using phase_dependent_action_list = PhaseDepActionList;
  using array_index = ElementId<volume_dim>;

  using const_global_cache_tags =
      tmpl::list<domain::Tags::Domain<volume_dim>,
                 typename DgElementArray_detail::import_numeric_data_cache_tags<
                     DgElementArray, ImportInitialData>::type>;

  using array_allocation_tags =
      tmpl::list<domain::Tags::InitialRefinementLevels<volume_dim>>;

  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>,
      array_allocation_tags>;

  template <typename DbTagList, typename ArrayIndex>
  static void pup(PUP::er& p, db::DataBox<DbTagList>& box,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index) noexcept {
    // this does not actually insert anything into the PUP::er stream, so
    // nothing is done on a sizing pup.
    if constexpr (DgElementArray_detail::has_registration_list<
                      Metavariables, DgElementArray>) {
      using registration_list =
          typename Metavariables::template registration_list<
              DgElementArray>::type;
      if (p.isPacking()) {
        tmpl::for_each<registration_list>(
            [&box, &cache, &array_index](auto registration_v) noexcept {
              using registration = typename decltype(registration_v)::type;
              registration::template perform_deregistration<DgElementArray>(
                  box, cache, array_index);
            });
      }
      if (p.isUnpacking()) {
        tmpl::for_each<registration_list>(
            [&box, &cache, &array_index](auto registration_v) noexcept {
              using registration = typename decltype(registration_v)::type;
              registration::template perform_registration<DgElementArray>(
                  box, cache, array_index);
            });
      }
    }
  }

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<initialization_tags>&
          initialization_items) noexcept;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<DgElementArray>(local_cache)
        .start_phase(next_phase);

    if constexpr (not std::is_same_v<ImportInitialData, ImportNoInitialData>) {
      static_assert(
          std::is_same_v<typename ImportInitialData::phase_type,
                         typename Metavariables::Phase>,
          "Make sure the 'ImportNumericInitialData' type uses a 'Phase' "
          "that is defined in the Metavariables.");
      if (next_phase == ImportInitialData::import_phase) {
        Parallel::threaded_action<
            DgElementArray_detail::read_element_data_action<
                DgElementArray, typename ImportInitialData::initial_data>>(
            Parallel::get_parallel_component<
                importers::VolumeDataReader<Metavariables>>(local_cache));
      }
    }
  }
};
template <class Metavariables, class PhaseDepActionList,
          class ImportInitialData>
void DgElementArray<Metavariables, PhaseDepActionList, ImportInitialData>::
    allocate_array(
        Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
        const tuples::tagged_tuple_from_typelist<initialization_tags>&
            initialization_items) noexcept {
  auto& local_cache = *(global_cache.ckLocalBranch());
  auto& dg_element_array =
      Parallel::get_parallel_component<DgElementArray>(local_cache);
  const auto& domain =
      Parallel::get<domain::Tags::Domain<volume_dim>>(local_cache);
  const auto& initial_refinement_levels =
      get<domain::Tags::InitialRefinementLevels<volume_dim>>(
          initialization_items);

  // Get desired partition
  const auto partition = get<Tags::DomainPartitionChunkSize<volume_dim>,
                             Metavariables>(local_cache);

  // If "zero" partition, use behavior from spectre:develop
  if (partition == make_array<volume_dim>(0_st)) {

  int which_proc = 0;
  for (const auto& block : domain.blocks()) {
    const auto initial_ref_levs = initial_refinement_levels[block.id()];
    const std::vector<ElementId<volume_dim>> element_ids =
        initial_element_ids(block.id(), initial_ref_levs);
    const int number_of_procs = Parallel::number_of_procs();
    for (size_t i = 0; i < element_ids.size(); ++i) {
      dg_element_array(ElementId<volume_dim>(element_ids[i]))
          .insert(global_cache, initialization_items, which_proc);
      which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
    }
  }

  } else {

  // Check partition is compatible with the domain
  ASSERT(domain.blocks().size() == 1, "assuming single block");
  for (size_t i = 0; i < volume_dim; ++i) {
    if (two_to_the(initial_refinement_levels[0][i]) % partition[i] != 0) {
      ERROR("input partition did not match extents");
    }
  }

  // Relate procs to nodes
  // NOTE: currently hardcoded for 2 nodes (of any number of procs/node)
  constexpr size_t number_of_nodes = 2;
  const int number_of_procs = Parallel::number_of_procs();
  ASSERT(number_of_procs % number_of_nodes == 0, "oops");
  const size_t number_of_procs_per_node =
      static_cast<size_t>(number_of_procs) / number_of_nodes;
  auto which_proc_on_this_node = make_array<number_of_nodes>(0_st);

  for (const auto& block : domain.blocks()) {
    const auto initial_ref_levs = initial_refinement_levels[block.id()];
    const std::vector<ElementId<volume_dim>> element_ids =
        initial_element_ids(block.id(), initial_ref_levs);
    for (size_t i = 0; i < element_ids.size(); ++i) {
      size_t which_node = 0;
      for (size_t d = 0; d < volume_dim; ++d) {
        which_node += element_ids[i].segment_ids()[d].index() / partition[d];
      }
      which_node = which_node % number_of_nodes;
      auto& which_proc = which_proc_on_this_node[which_node];
      dg_element_array(ElementId<volume_dim>(element_ids[i]))
          .insert(global_cache, initialization_items,
              which_proc + which_node * number_of_procs_per_node);
      which_proc = (which_proc + 1 == number_of_procs_per_node ? 0 :
                                      which_proc + 1);
    }
  }

  }
  dg_element_array.doneInserting();
}

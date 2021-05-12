// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"

namespace OptionTags {
struct WallClockHoursForCheckpointAndExit {
  using type = double;
  static constexpr Options::String help =
      "WallClock time in hours from which to try to checkpoint and exit";
};
}  // namespace OptionTags

namespace Parallel::Tags {
// WallClock time in hours from which to try to checkpoint and exit
struct WallClockHoursForCheckpointAndExit : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<
      ::OptionTags::WallClockHoursForCheckpointAndExit>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double hours) noexcept {
    return hours;
  }
};
}  // namespace Parallel::Tags

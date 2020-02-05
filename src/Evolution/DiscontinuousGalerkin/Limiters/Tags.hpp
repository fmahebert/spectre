// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"

namespace OptionTags {
/*!
 * \ingroup OptionGroupsGroup
 * \brief Holds the `OptionTags::Limiter` option in the input file
 */
struct LimiterGroup {
  static std::string name() noexcept { return "Limiter"; }
  static constexpr OptionString help = "Options for limiting troubled cells";
};

/*!
 * \ingroup OptionTagsGroup
 * \brief The global cache tag that retrieves the parameters for the limiter
 * from the input file
 */
template <typename LimiterType>
struct Limiter {
  static std::string name() noexcept { return option_name<LimiterType>(); }
  static constexpr OptionString help = "Options for the limiter";
  using type = LimiterType;
  using group = LimiterGroup;
};
}  // namespace OptionTags

namespace Tags {
/*!
 * \brief The global cache tag for the limiter
 */
template <typename LimiterType>
struct Limiter : db::SimpleTag {
  using type = LimiterType;
  using option_tags = tmpl::list<::OptionTags::Limiter<LimiterType>>;

  static constexpr bool pass_metavariables = false;
  static LimiterType create_from_options(const LimiterType& limiter) noexcept {
    return limiter;
  }
};

/*!
 * \brief Buffer to hold diagnostics on the limiter's activation
 *
 * It can be useful to see when and where the limiters activate to modify the
 * solution. Although this information could be most compactly encoded in an
 * integer or enum, we opt to write it to a DataVector so that we can easily
 * save the diagnostic to H5 output.
 *
 * As an example, the limiter could set the DataVector to 0. each time it does
 * not activate, and to 1. each time it does. More sophisticated limiters could
 * encode additional information.
 *
 * This Tag should be added to the DgElementArray DataBox during initialization.
 * Then, at each timestep, the limiter can modify the values in the DataVector.
 */
struct LimiterDiagnostics : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace Tags

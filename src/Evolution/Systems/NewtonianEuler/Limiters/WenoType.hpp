// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <ostream>

/// \cond
class Option;
template <typename T>
struct create_from_yaml;
/// \endcond

namespace NewtonianEuler {
namespace Limiters {
/// \brief Possible types of the NewtonianEuler-specialized WENO limiter
enum class WenoType {
  CharacteristicHweno,
  CharacteristicSimpleWeno,
  ConservativeHweno,
  ConservativeSimpleWeno
};

std::ostream& operator<<(std::ostream& os,
                         Limiters::WenoType weno_type) noexcept;
}  // namespace Limiters
}  // namespace NewtonianEuler

template <>
struct create_from_yaml<NewtonianEuler::Limiters::WenoType> {
  template <typename Metavariables>
  static NewtonianEuler::Limiters::WenoType create(const Option& options) {
    return create<void>(options);
  }
};
template <>
NewtonianEuler::Limiters::WenoType
create_from_yaml<NewtonianEuler::Limiters::WenoType>::create<void>(
    const Option& options);

// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Limiters/WenoType.hpp"

#include <ostream>
#include <string>

#include "ErrorHandling/Error.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"

std::ostream& NewtonianEuler::Limiters::operator<<(
    std::ostream& os,
    const NewtonianEuler::Limiters::WenoType weno_type) noexcept {
  switch (weno_type) {
    case Limiters::WenoType::CharacteristicSimpleWeno:
      return os << "CharacteristicSimpleWeno";
    case Limiters::WenoType::ConservativeHweno:
      return os << "ConservativeHweno";
    case Limiters::WenoType::ConservativeSimpleWeno:
      return os << "ConservativeSimpleWeno";
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR(
          "Missing a case for operator<<(NewtonianEuler::Limiters::WenoType)");
      // LCOV_EXCL_STOP
  }
}

template <>
NewtonianEuler::Limiters::WenoType
create_from_yaml<NewtonianEuler::Limiters::WenoType>::create<void>(
    const Option& options) {
  const std::string weno_type_read = options.parse_as<std::string>();
  if (weno_type_read == "CharacteristicSimpleWeno") {
    return NewtonianEuler::Limiters::WenoType::CharacteristicSimpleWeno;
  } else if (weno_type_read == "ConservativeHweno") {
    return NewtonianEuler::Limiters::WenoType::ConservativeHweno;
  } else if (weno_type_read == "ConservativeSimpleWeno") {
    return NewtonianEuler::Limiters::WenoType::ConservativeSimpleWeno;
  }
  PARSE_ERROR(
      options.context(),
      "Failed to convert \""
          << weno_type_read
          << "\" to NewtonianEuler::Limiters::WenoType. Expected one of: "
             "{CharacteristicSimpleWeno, ConservativeHweno, "
             "ConservativeSimpleWeno}.");
}

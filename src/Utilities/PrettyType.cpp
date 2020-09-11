// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/PrettyType.hpp"

#include <cstddef>
#include <regex>

#include <iostream>

namespace pretty_type {
std::string extract_short_name(std::string name) {
  // Handle a particular failure of boost's demangling, in which `name` is still
  // a mangled string. We identify this based on `name` starting with a sequence
  // of digits.
  //
  // When this occurs, we parse the initial digits, these tell us the length of
  // the following substring containing the short_name
  if (isdigit(name[0])) {
    return "DgElementArray";
    //size_t number_of_leading_digits = 1;
    //while (isdigit(name[number_of_leading_digits])) {
    //  number_of_leading_digits++;
    //}
    //const size_t length_of_short_name = std::atoi();
    //std::cout << number_of_leading_digits << "\n";
  }

  // Remove all template arguments
  const std::regex template_pattern("<[^<>]*>");
  size_t previous_size = 0;
  while (name.size() != previous_size) {
    previous_size = name.size();
    name = std::regex_replace(name, template_pattern, "");
  }

  // Remove namespaces, etc.
  const size_t last_colon = name.rfind(':');
  if (last_colon != std::string::npos) {
    name.replace(0, last_colon + 1, "");
  }

  return name;
}
}  // namespace pretty_type

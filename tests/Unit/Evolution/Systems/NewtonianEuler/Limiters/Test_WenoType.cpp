// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/NewtonianEuler/Limiters/WenoType.hpp"
#include "Framework/TestCreation.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Limiters.WenoType",
                  "[Limiters][Unit]") {
  CHECK(NewtonianEuler::Limiters::WenoType::ConservativeHweno ==
        TestHelpers::test_creation<NewtonianEuler::Limiters::WenoType>(
            "ConservativeHweno"));
  CHECK(NewtonianEuler::Limiters::WenoType::ConservativeSimpleWeno ==
        TestHelpers::test_creation<NewtonianEuler::Limiters::WenoType>(
            "ConservativeSimpleWeno"));

  CHECK(get_output(NewtonianEuler::Limiters::WenoType::ConservativeHweno) ==
        "ConservativeHweno");
  CHECK(
      get_output(NewtonianEuler::Limiters::WenoType::ConservativeSimpleWeno) ==
      "ConservativeSimpleWeno");
}

// [[OutputRegex, Failed to convert "BadType" to
// NewtonianEuler::Limiters::WenoType]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.Limiters.WenoType.OptionParseError",
    "[Limiters][Unit]") {
  ERROR_TEST();
  TestHelpers::test_creation<NewtonianEuler::Limiters::WenoType>("BadType");
}

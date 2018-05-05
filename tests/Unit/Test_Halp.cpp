// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "DataStructures/Index.hpp"
#include "Utilities/TMPL.hpp"

template <typename TagsToLimit>
struct Foo;

template <typename... Tags>
struct Foo<tmpl::list<Tags...>> {
  void apply(const Index<1>& mesh) const {
    expand_pack((
        (void)[=](auto /*tag*/) {
          // gcc 7.3 warns, at the "[=]" capture point above:
          // warning: converting to ‘Index<1>’ from initializer list would use
          // explicit constructor ‘Index<Dim>::Index(size_t) [with long unsigned
          // int Dim = 1; size_t = long unsigned int]’
          // note: in C++11 and above a default constructor can be explicit
          CHECK(mesh[0] == 3);
          // CHECK fails with mesh[0] = garbage, looks like uninitialized memory
        }(Tags{}),
        '0')...);
  }
};

struct tag1 {};
struct tag2 {};

SPECTRE_TEST_CASE("Unit.Foo.Halp", "[Unit]") {
  const Foo<tmpl::list<tag1, tag2>> foo;
  foo.apply(Index<1>(3));
}

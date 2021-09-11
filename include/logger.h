// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>

namespace grann {
#if defined(GRANN_DLL)
  extern std::basic_ostream<char> cout;
  extern std::basic_ostream<char> cerr;
#else
  extern std::basic_ostream<char> cout;
  extern std::basic_ostream<char> cerr;
#endif

}  // namespace grann

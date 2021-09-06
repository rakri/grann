// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "windows_customizations.h"

namespace grann {
#if defined(GRANN_DLL)
  extern std::basic_ostream<char> cout;
  extern std::basic_ostream<char> cerr;
#else
  GRANN_DLLIMPORT extern std::basic_ostream<char> cout;
  GRANN_DLLIMPORT extern std::basic_ostream<char> cerr;
#endif

}  // namespace grann

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <string>
#include "windows_customizations.h"

#ifndef _WINDOWS
#define __FUNCSIG__ __PRETTY_FUNCTION__
#endif

namespace grann {
  class ANNException {
   public:
    GRANN_DLLEXPORT ANNException(const std::string& message, int errorCode);
    GRANN_DLLEXPORT ANNException(const std::string& message, int errorCode,
                                   const std::string& funcSig,
                                   const std::string& fileName,
                                   unsigned int       lineNum);

    GRANN_DLLEXPORT std::string message() const;

   private:
    int          _errorCode;
    std::string  _message;
    std::string  _funcSig;
    std::string  _fileName;
    unsigned int _lineNum;
  };
}  // namespace grann

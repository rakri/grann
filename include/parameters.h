// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <sstream>
#include <typeinfo>
#include <unordered_map>

namespace grann {

class Parameters {
 public:
  Parameters() {
    Set<uint32_t>("num_threads", 0);
    Set<float>("alpha",1);
    Set<uint32_t>("pruning_rule", 0);
  }

  template<typename ParamType>
  inline void Set(const std::string &name, const ParamType &value) {
    std::stringstream sstream;
    sstream << value;
    params[name] = sstream.str();
  }

  template<typename ParamType>
  inline ParamType Get(const std::string &name) const {
    auto item = params.find(name);
    if (item == params.end()) {
      throw std::invalid_argument(std::string("Invalid parameter name ") + name);
    } else {
      return ConvertStrToValue<ParamType>(item->second);
    }
  }

  template<typename ParamType>
  inline ParamType Get(const std::string &name, const ParamType &default_value) {
    try {
      return Get<ParamType>(name);
    } catch (std::invalid_argument e) {
      return default_value;
    }
  }
 private:
  std::unordered_map<std::string, std::string> params;

  template<typename ParamType>
  inline ParamType ConvertStrToValue(const std::string &str) const {
    std::stringstream sstream(str);
    ParamType value;
    if (!(sstream >> value) || !sstream.eof()) {
      std::stringstream err;
      err << "Failed to convert value '" << str << "' to type: " << typeid(value).name();
      throw std::runtime_error(err.str());
    }
    return value;
  }

};

}
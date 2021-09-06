// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#ifdef _WINDOWS
#define GRANN_DLLEXPORT __declspec(dllexport)
#define GRANN_DLLIMPORT __declspec(dllimport)
#else
#define GRANN_DLLEXPORT
#define GRANN_DLLIMPORT
#endif

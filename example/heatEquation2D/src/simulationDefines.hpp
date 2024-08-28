/* Copyright 2024 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

// simulation defines
// {Y, X}
constexpr alpaka::Vec<alpaka::DimInt<2u>, uint32_t> numNodes{64, 64};
constexpr alpaka::Vec<alpaka::DimInt<2u>, uint32_t> haloSize{2, 2};
constexpr alpaka::Vec<alpaka::DimInt<2u>, uint32_t> extent = numNodes + haloSize;

constexpr uint32_t numTimeSteps = 100;
constexpr double tMax = 0.001;
// x, y in [0, 1], t in [0, tMax]
constexpr double dx = 1.0 / static_cast<double>(extent[1] - 1);
constexpr double dy = 1.0 / static_cast<double>(extent[0] - 1);
constexpr double dt = tMax / static_cast<double>(numTimeSteps);

// Define a workdiv for the given problem
constexpr alpaka::Vec<alpaka::DimInt<2u>, uint32_t> elemPerThread{1, 1};
// Appropriate chunk size to split your problem for your Acc
constexpr alpaka::Vec<alpaka::DimInt<2u>, uint32_t> chunkSize{16u, 16u};
constexpr auto chunkSizeWithHalo = chunkSize + haloSize;
constexpr alpaka::Vec<alpaka::DimInt<2u>, uint32_t> numChunks{
    alpaka::core::divCeil(numNodes[0], chunkSize[0]),
    alpaka::core::divCeil(numNodes[1], chunkSize[1]),
};

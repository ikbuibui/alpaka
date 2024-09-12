/* Copyright 2024 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "analyticalSolution.hpp"
#include "helpers.hpp"

#include <alpaka/alpaka.hpp>

//! alpaka version of explicit finite-difference 1d heat equation solver
//!
//! Applies boundary conditions
//! forward difference in t and second-order central difference in x
//!
//! \param uBuf grid values of u for each x, y and the current value of t:
//!                 u(x, y, t)  | t = t_current
//! \param pitch
//! \param step simulation timestep
//! \param dx step in x
//! \param dy step in y
//! \param dt step in t
struct BoundaryKernel
{
    template<typename TAcc, typename TChunk>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        double* const uBuf,
        TChunk const pitch,
        uint32_t step,
        double const dx,
        double const dy,
        double const dt) const -> void
    {
        // Get extents(dimensions)
        auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        // Get indexes
        auto const gridBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        auto const idx2D = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);

        if(idx2D[0] == 0 || idx2D[0] == gridThreadExtent[0] - 1 || idx2D[1] == 0
           || idx2D[1] == gridThreadExtent[1] - 1)
        {
            auto elem = getElementPtr(uBuf, idx2D, pitch);
            *elem = exactSolution(idx2D[1] * dx, idx2D[0] * dy, step * dt);
        }
    }
};

template<typename TAcc, typename TDev, typename TQueue, typename... TArgs>
auto applyBoundaries(
    TDev const& devAcc,
    alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& extent,
    alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& elemPerThread,
    TQueue& queue,
    TArgs&&... args) -> void
{
    static alpaka::KernelCfg<TAcc> const boundaryCfg = {extent, elemPerThread};
    static BoundaryKernel boundaryKernel{};
    static auto const workDivBoundary = alpaka::getValidWorkDiv(boundaryCfg, devAcc, boundaryKernel, args...);

    alpaka::exec<TAcc>(queue, workDivBoundary, boundaryKernel, args...);
}

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
        auto const gridThreadsExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        for(auto index : alpaka::uniformElementsND(acc))
        {
            if(index.x() == 0 || index.x() == gridThreadsExtent[1] - 1 || index.y() == 0
               || index.y() == gridThreadsExtent[0] - 1)
            {
                auto elem = getElementPtr(uBuf, index, pitch);
                *elem = exactSolution(index.x() * dx, index.y() * dy, step * dt);
            }
        }
    }
};

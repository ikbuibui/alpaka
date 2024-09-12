/* Copyright 2024 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

//! alpaka version of explicit finite-difference 2D heat equation solver
//!
//! Solving equation u_t(x, t) = u_xx(x, t) + u_yy(y, t) using a simple explicit scheme with
//! forward difference in t and second-order central difference in x and y
//!
//! \param uCurrBuf Current buffer with grid values of u for each x, y pair and the current value of t:
//!                 u(x, y, t) | t = t_current
//! \param uNextBuf resulting grid values of u for each x, y pair and the next value of t:
//!              u(x, y, t) | t = t_current + dt
//! \param chunkSize The size of the chunk or tile that the user divides the problem into. This defines the size of the
//!                  workload handled by each thread block.
//! \param dx step in x
//! \param dy step in y
//! \param dt step in t
struct StencilKernel
{
    template<typename TAcc, typename TDim, typename TIdx, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TMdSpan uCurrBuf,
        TMdSpan uNextBuf,
        alpaka::Vec<TDim, TIdx> const chunkSize,
        double const dx,
        double const dy,
        double const dt) const -> void
    {
        // Get indexes
        auto const gridBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const blockStartThreadIdx = gridBlockIdx * chunkSize;

        // Each kernel executes one element
        double const rX = dt / (dx * dx);
        double const rY = dt / (dy * dy);

        auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        auto const numThreadsPerBlock = blockThreadExtent.prod();

        auto const threadIdx1D = alpaka::mapIdx<1>(blockThreadIdx, blockThreadExtent)[0u];

        // go over only core cells
        for(auto i = threadIdx1D; i < chunkSize.prod(); i += numThreadsPerBlock)
        {
            auto idx2D = alpaka::mapIdx<2>(alpaka::Vec<alpaka::DimInt<1u>, TIdx>(i), chunkSize);
            idx2D = idx2D + alpaka::Vec<TDim, TIdx>{1, 1}; // offset for halo above and to the left

            auto const bufIdx = idx2D + blockStartThreadIdx;

            uNextBuf(bufIdx[0], bufIdx[1])
                = uCurrBuf(bufIdx[0], bufIdx[1]) * (1.0 - 2.0 * rX - 2.0 * rY)
                  + uCurrBuf(bufIdx[0], bufIdx[1] + 1) * rX + uCurrBuf(bufIdx[0], bufIdx[1] - 1) * rX
                  + uCurrBuf(bufIdx[0] + 1, bufIdx[1]) * rY + uCurrBuf(bufIdx[0] - 1, bufIdx[1]) * rY;
        }
    }
};

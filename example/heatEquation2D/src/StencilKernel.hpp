/* Copyright 2024 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "helpers.hpp"

#include <alpaka/alpaka.hpp>

//! alpaka version of explicit finite-difference 2D heat equation solver
//!
//! \tparam T_BlockSize1D size of the shared memory box
//!
//! Solving equation u_t(x, t) = u_xx(x, t) + u_yy(y, t) using a simple explicit scheme with
//! forward difference in t and second-order central difference in x and y
//!
//! \param uCurrBuf grid values of u for each x and the current value of t:
//!                 u(x, y, t) | t = t_current
//! \param uNext resulting grid values of u for each x and the next value of t:
//!              u(x, y, t) | t = t_current + dt
//! \param dx step in x
//! \param dt step in t

template<size_t T_ChunkSize1D>
struct StencilKernel
{
    template<typename TAcc, typename TDim, typename TIdx>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        double const* const uCurrBuf,
        double* const uNextBuf,
        alpaka::Vec<TDim, TIdx> const chunkSize,
        alpaka::Vec<TDim, TIdx> const pitch,
        double const dx,
        double const dy,
        double const dt) const -> void
    {
        auto& sdata(alpaka::declareSharedVar<double[T_ChunkSize1D], __COUNTER__>(acc));

        // Get extents(dimensions)
        auto const gridBlockExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
        auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        auto const numThreadsPerBlock = blockThreadExtent.prod();

        // Get indexes
        auto const gridBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const threadIdx1D = alpaka::mapIdx<1>(blockThreadIdx, blockThreadExtent)[0u];
        auto const blockStartIdx = gridBlockIdx * chunkSize;

        constexpr alpaka::Vec<TDim, TIdx> guard{2, 2};

        for(auto i = threadIdx1D; i < T_ChunkSize1D; i += numThreadsPerBlock)
        {
            auto idx2d = alpaka::mapIdx<2>(alpaka::Vec(i), chunkSize + guard);
            idx2d = idx2d + blockStartIdx;
            auto elem = getElementPtr(uCurrBuf, idx2d, pitch);
            sdata[i] = *elem;
        }

        alpaka::syncBlockThreads(acc);

        // Each kernel executes one element
        double const r_x = dt / (dx * dx);
        double const r_y = dt / (dy * dy);

        // go over only core cells
        for(auto i = threadIdx1D; i < chunkSize.prod(); i += numThreadsPerBlock)
        {
            auto idx2d = alpaka::mapIdx<2>(alpaka::Vec(i), chunkSize);
            idx2d = idx2d + alpaka::Vec<TDim, TIdx>{1, 1}; // offset for halo above and to the left
            auto localIdx1D = alpaka::mapIdx<1>(idx2d, chunkSize + guard)[0u];


            auto bufIdx = idx2d + blockStartIdx;
            auto elem = getElementPtr(uNextBuf, bufIdx, pitch);

            *elem = sdata[localIdx1D] * (1.0 - 2.0 * r_x - 2.0 * r_y) + sdata[localIdx1D - 1] * r_x
                    + sdata[localIdx1D + 1] * r_x + sdata[localIdx1D - chunkSize[1] - 2] * r_y
                    + sdata[localIdx1D + chunkSize[1] + 2] * r_y;
        }
    }
};

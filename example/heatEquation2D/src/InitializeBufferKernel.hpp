/* Copyright 2024 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "analyticalSolution.hpp"
#include "helpers.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdio>

//! Exact solution to the test problem
//! u_t(x, y, t) = u_xx(x, t) + u_yy(y, t), x in [0, 1], y in [0, 1], t in [0, T]
//!
//! \param x value of x
//! \param x value of y
//! \param t value of t
template<typename TAcc>
ALPAKA_FN_ACC auto analyticalSolution(TAcc const& acc, double const x, double const y, double const t) -> double
{
    constexpr double pi = alpaka::math::constants::pi;
    return alpaka::math::exp(acc, -pi * pi * t) * (alpaka::math::sin(acc, pi * x) + alpaka::math::sin(acc, pi * y));
}

//! alpaka version of explicit finite-difference 2D heat equation solver
//!
//! Solving equation u_t(x, t) = u_xx(x, t) + u_yy(y, t) using a simple explicit scheme with
//! forward difference in t and second-order central difference in x and y
//!
//! \param bufData Current buffer data with grid values of u for each x, y pair and the current value of t:
//!                 u(x, y, t) | t = t_current
//! \param chunkSize The size of the chunk or tile that the user divides the problem into. This defines the size of the
//!                  workload handled by each thread block.
//! \param pitch The pitch (or stride) in memory corresponding to the TDim grid in the accelerator's memory.
//!              This is used to calculate memory offsets when accessing elements in the buffers.
struct InitializeBufferKernel
{
    template<typename TAcc, typename TDim, typename TIdx>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        double* const bufData,
        alpaka::Vec<TDim, TIdx> const pitch,
        double dx,
        double dy) const -> void
    {
        // Get extents(dimensions)
        auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);

        // Get indexes
        auto const gridBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);

        printf(
            "Block index [y,x]: %u %u \t Thread index in block [y,x] : %u %u \t Thread index in grid [y,x] : "
            "%u %u \n",
            gridBlockIdx[0] * blockThreadExtent[0] + blockThreadIdx[0],
            gridBlockIdx[1] * blockThreadExtent[1] + blockThreadIdx[1],
            blockThreadIdx[0],
            blockThreadIdx[1],
            gridThreadIdx[0],
            gridThreadIdx[1]);

        *getElementPtr(bufData, gridThreadIdx, pitch)
            = analyticalSolution(acc, gridThreadIdx[1] * dx, gridThreadIdx[0] * dy, 0.0);
    }
};

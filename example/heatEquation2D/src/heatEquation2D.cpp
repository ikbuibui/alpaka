/* Copyright 2020 Benjamin Worpitz, Matthias Werner, Jakob Krude, Sergei Bastrakov, Bernhard Manfred Gruber
 * SPDX-License-Identifier: ISC
 */

// #include "pngCreator.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>
#include <alpaka/idx/MapIdx.hpp>
#include <alpaka/mem/global/DeviceGlobalCpu.hpp>
#include <alpaka/vec/Vec.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <utility>

//! Exact solution to the test problem
//! u_t(x, y, t) = u_xx(x, t) + u_yy(y, t), x in [0, 1], y in [0, 1], t in [0, T]
//! u(0, t) = u(1, t) = 0
//! u(x, 0) = sin(pi * x)
//! u(0, y) = sin(pi * y)
//!
//! \param x value of x
//! \param x value of y
//! \param t value of t
ALPAKA_FN_HOST_ACC auto exactSolution(double const x, double const y, double const t) -> double
{
    constexpr double pi = 3.141592653589793238462643383279502884;
    return std::exp(-pi * pi * t) * std::sin(pi * x) + std::exp(-pi * pi * t) * std::sin(pi * y);
}

template<typename T, typename TDim, typename TIdx>
ALPAKA_FN_ACC T* getElementPtr(T* ptr, alpaka::Vec<TDim, TIdx> idx, alpaka::Vec<TDim, TIdx> pitch)
{
    return reinterpret_cast<T*>(reinterpret_cast<std::byte*>(ptr) + idx[0] * pitch[0] + idx[1] * pitch[1]);
}

template<typename T, typename TDim, typename TIdx>
ALPAKA_FN_ACC T const* getElementPtr(T const* ptr, alpaka::Vec<TDim, TIdx> idx, alpaka::Vec<TDim, TIdx> pitch)
{
    return reinterpret_cast<T const*>(reinterpret_cast<std::byte const*>(ptr) + idx[0] * pitch[0] + idx[1] * pitch[1]);
}

//! alpaka version of explicit finite-difference 1d heat equation solver
//!
//! \tparam T_BlockSize1D size of the shared memory box
//!
//! Solving equation u_t(x, t) = u_xx(x, t) using a simple explicit scheme with
//! forward difference in t and second-order central difference in x
//!
//! \param uCurrBuf grid values of u for each x and the current value of t:
//!                 u(x, t) | t = t_current
//! \param uNext resulting grid values of u for each x and the next value of t:
//!              u(x, t) | t = t_current + dt
//! \param dx step in x
//! \param dt step in t

template<size_t T_ChunkSize1D>
struct HeatEquationKernel
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

//! alpaka version of explicit finite-difference 1d heat equation solver
//!
//! Applies boundary conditions
//! forward difference in t and second-order central difference in x
//!
//! \param uNextBuf grid values of u for each x and the current value of t:
//!                 u(x, t) | t = t_current
//! \param uNext resulting grid values of u for each x and the next value of t:
//!              u(x, t) | t = t_current + dt
//! \param extent number of grid nodes in x (eq. to numNodesX)
//! \param dx step in x
//! \param dt step in t

struct ApplyBoundariesKernel
{
    template<typename TAcc, typename T_Chunk>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        double* const uBuf,
        T_Chunk const chunkSize,
        T_Chunk const pitch,
        uint32_t step,
        double const dx,
        double const dy,
        double const dt) const -> void
    {
        using Dim = alpaka::DimInt<2u>;
        using Idx = uint32_t;

        // Get extents(dimensions)
        auto const gridBlockExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
        auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        auto const numThreadsPerBlock = blockThreadExtent.prod();

        // Get indexes
        auto const gridBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const threadIdx1D = alpaka::mapIdx<1>(blockThreadIdx, blockThreadExtent)[0u];
        auto const blockStartIdx = gridBlockIdx * chunkSize;

        // apply boundary conditions
        // top row
        if(gridBlockIdx[0] == 0)
        {
            auto const globalIdx = blockStartIdx + alpaka::Vec<Dim, Idx>(0, 1);
            for(auto i = threadIdx1D; i < chunkSize[1]; i += numThreadsPerBlock)
            {
                auto idx2D = globalIdx + alpaka::Vec<Dim, Idx>(0, i);
                auto elem = getElementPtr(uBuf, idx2D, pitch);
                *elem = exactSolution(idx2D[1] * dx, idx2D[0] * dy, step * dt);
            }
        }
        if(gridBlockIdx[0] == gridBlockExtent[0] - 1)
        {
            auto const globalIdx = blockStartIdx + alpaka::Vec<Dim, Idx>{chunkSize[0] + 1, 1};

            for(auto i = threadIdx1D; i < chunkSize[1]; i += numThreadsPerBlock)
            {
                auto idx2D = globalIdx + alpaka::Vec<Dim, Idx>(0, i);
                auto elem = getElementPtr(uBuf, idx2D, pitch);
                *elem = exactSolution(idx2D[1] * dx, idx2D[0] * dy, step * dt);
            }
        }
        if(gridBlockIdx[1] == 0)
        {
            auto const globalIdx = blockStartIdx + alpaka::Vec<Dim, Idx>(1, 0);

            for(auto i = threadIdx1D; i < chunkSize[0]; i += numThreadsPerBlock)
            {
                auto idx2D = globalIdx + alpaka::Vec<Dim, Idx>(i, 0);
                auto elem = getElementPtr(uBuf, idx2D, pitch);
                *elem = exactSolution(idx2D[1] * dx, idx2D[0] * dy, step * dt);
            }
        }
        if(gridBlockIdx[1] == gridBlockExtent[1] - 1)
        {
            auto const globalIdx = blockStartIdx + alpaka::Vec<Dim, Idx>{1, chunkSize[1] + 1};

            for(auto i = threadIdx1D; i < chunkSize[0]; i += numThreadsPerBlock)
            {
                auto idx2D = globalIdx + alpaka::Vec<Dim, Idx>(i, 0);
                auto elem = getElementPtr(uBuf, idx2D, pitch);
                *elem = exactSolution(idx2D[1] * dx, idx2D[0] * dy, step * dt);
            }
        }
    }
};

//! Each kernel computes the next step for one point.
//! Therefore the number of threads should be equal to numNodesX.
//! Every time step the kernel will be executed numNodesX-times
//! After every step the curr-buffer will be set to the calculated values
//! from the next-buffer.
//!
//! In standard projects, you typically do not execute the code with any available accelerator.
//! Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
//! selected accelerator only. If you use the example as the starting point for your project, you can rename the
//! example() function to main() and move the accelerator tag to the function body.
template<typename TAccTag>
auto example(TAccTag const&) -> int
{
    if constexpr(std::is_same_v<TAccTag, alpaka::TagCpuSerial>)
    {
        return EXIT_SUCCESS;
    }
    // Set Dim and Idx type
    using Dim = alpaka::DimInt<2u>;
    using Idx = uint32_t;

    // Define the accelerator
    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    // Select specific devices
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    // simulation defines
    // Parameters (a user is supposed to change numNodesX, numTimeSteps)
    constexpr alpaka::Vec<Dim, Idx> numNodes{512, 1024}; // {Y, X}
    constexpr alpaka::Vec<Dim, Idx> haloSize{2, 2};


    constexpr uint32_t numTimeSteps = 10000;
    constexpr double tMax = 0.001;
    // x in [0, 1], t in [0, tMax]
    constexpr double dx = 1.0 / static_cast<double>(numNodes[1] + haloSize[1] - 1);
    constexpr double dy = 1.0 / static_cast<double>(numNodes[0] + haloSize[0] - 1);
    constexpr double dt = tMax / static_cast<double>(numTimeSteps);

    // Check the stability condition
    constexpr double r = dt / std ::min(dx * dx, dy * dy);
    if constexpr(r > 0.5)
    {
        std::cerr << "Stability condition check failed: dt/min(dx^2,dy^2) = " << r
                  << ", it is required to be <= 0.5\n";
        return EXIT_FAILURE;
    }

    constexpr alpaka::Vec<Dim, Idx> extent = numNodes + haloSize;

    // Initialize host-buffer
    // This buffer holds the calculated values
    auto uNextBufHost = alpaka::allocBuf<double, Idx>(devHost, extent);
    // This buffer will hold the current values (used for the next step)
    auto uCurrBufHost = alpaka::allocBuf<double, Idx>(devHost, extent);

    double* const pCurrHost = uCurrBufHost.data();

    // Accelerator buffer
    using BufAcc = alpaka::Buf<Acc, double, Dim, Idx>;
    BufAcc uNextBufAcc{alpaka::allocBuf<double, Idx>(devAcc, extent)};
    BufAcc uCurrBufAcc{alpaka::allocBuf<double, Idx>(devAcc, extent)};


    double* pCurrAcc = uCurrBufAcc.data();
    double* pNextAcc = uNextBufAcc.data();

    // Apply initial conditions for the test problem
    for(uint32_t j = 0; j < extent[0]; j++)
    {
        for(uint32_t i = 0; i < extent[1]; i++)
        {
            pCurrHost[j * extent[1] + i] = exactSolution(i * dx, j * dy, 0.0);
        }
    }

    // Select queue
    using QueueProperty = alpaka::NonBlocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    QueueAcc queue1{devAcc};
    QueueAcc queue2{devAcc};

    // Define a workdiv for the given problem
    constexpr alpaka::Vec<Dim, Idx> elemPerThread{1, 1};

    // Appropriate chunk size for your Acc
    constexpr alpaka::Vec<Dim, Idx> chunkSize{16u, 16u};
    constexpr alpaka::Vec<Dim, Idx> chunkSizeWithHalo{chunkSize[0] + haloSize[0], chunkSize[1] + haloSize[1]};
    // TODO clean this
    auto const maxThreadsPerBlock = alpaka::getAccDevProps<Acc>(devAcc).m_blockThreadExtentMax;
    auto const threadsPerBlock = maxThreadsPerBlock.prod() < chunkSize.prod() ? maxThreadsPerBlock : chunkSize;

    constexpr alpaka::Vec<Dim, Idx> numChunks{
        (extent[0] - haloSize[0] - 1) / chunkSize[0] + 1,
        (extent[1] - haloSize[1] - 1) / chunkSize[1] + 1};

    static_assert(
        (extent[0] - haloSize[0]) % chunkSize[0] == 0 && (extent[1] - haloSize[1]) % chunkSize[1] == 0,
        "Domain must be divisible by chunk size");

    auto const pitchCurrAcc{alpaka::getPitchesInBytes(uCurrBufAcc)};

    alpaka::WorkDivMembers<Dim, Idx> workDiv_manual{numChunks, threadsPerBlock, elemPerThread};

    HeatEquationKernel<chunkSizeWithHalo.prod()> heatEqKernel;
    ApplyBoundariesKernel boundariesKernel;

    // Copy host -> device
    alpaka::memcpy(queue1, uCurrBufAcc, uCurrBufHost);
    alpaka::wait(queue1);

    // PngCreator createPng;

    for(uint32_t step = 1; step <= numTimeSteps; step++)
    {
        // Compute next values
        alpaka::exec<
            Acc>(queue1, workDiv_manual, heatEqKernel, pCurrAcc, pNextAcc, chunkSize, pitchCurrAcc, dx, dy, dt);
        alpaka::exec<
            Acc>(queue1, workDiv_manual, boundariesKernel, pNextAcc, chunkSize, pitchCurrAcc, step, dx, dy, dt);

        if(step % 100 == 0) // even steps will have currBufHost and PCurr pointing to same buffer
        {
            alpaka::memcpy(queue2, uCurrBufHost, uCurrBufAcc);
            alpaka::wait(queue2);
            // createPng(step - 1, pCurrHost, extent);
        }

        // So we just swap next to curr (shallow copy)
        alpaka::wait(queue1);
        std::swap(pCurrAcc, pNextAcc);
        std::swap(uNextBufAcc, uCurrBufAcc);
    }

    // Copy device -> host
    alpaka::memcpy(queue1, uCurrBufHost, uCurrBufAcc);
    alpaka::wait(queue1);

    // Calculate error
    double maxError = 0.0;
    for(uint32_t j = 1; j < extent[0] - 1; j++)
    {
        for(uint32_t i = 0; i < extent[1]; i++)
        {
            auto const error = std::abs(uCurrBufHost.data()[j * extent[1] + i] - exactSolution(i * dx, j * dy, tMax));
            maxError = std::max(maxError, error);
        }
    }

    double const errorThreshold = 1e-5;
    bool resultCorrect = (maxError < errorThreshold);
    if(resultCorrect)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Execution results incorrect: Max error = " << maxError << " (the grid resolution may be too low)"
                  << std::endl;
        return EXIT_FAILURE;
    }
}

auto main() -> int
{
    // Execute the example once for each enabled accelerator.
    // If you would like to execute it for a single accelerator only you can use the following code.
    //  \code{.cpp}
    //  auto tag = TagCpuSerial;
    //  return example(tag);
    //  \endcode
    //
    // valid tags:
    //   TagCpuSerial, TagGpuHipRt, TagGpuCudaRt, TagCpuOmp2Blocks, TagCpuTbbBlocks,
    //   TagCpuOmp2Threads, TagCpuSycl, TagCpuTbbBlocks, TagCpuThreads,
    //   TagFpgaSyclIntel, TagGenericSycl, TagGpuSyclIntel
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}

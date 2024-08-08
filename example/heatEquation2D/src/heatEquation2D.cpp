/* Copyright 2020 Benjamin Worpitz, Matthias Werner, Jakob Krude, Sergei Bastrakov, Bernhard Manfred Gruber
 * SPDX-License-Identifier: ISC
 */

// #include "pngCreator.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <utility>

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
//! \param extent number of grid nodes in x (eq. to numNodesX)
//! \param dx step in x
//! \param dt step in t

// template<size_t T_BlockSize1D>
// struct HeatEquationKernel
// {
//     template<typename TAcc, typename T_Extent>
//     ALPAKA_FN_ACC auto operator()(
//         TAcc const& acc,
//         double const* const uCurrBuf,
//         double* const uNextBuf,
//         T_Extent const extent,
//         double const dx,
//         double const dy,
//         double const dt) const -> void
//     {
//         auto& sdata(alpaka::declareSharedVar<double[T_BlockSize1D], __COUNTER__>(acc));


//         // Get extents(dimensions)
//         auto const gridBlockExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
//         auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
//         // Get indexes
//         auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
//         auto const blockThreadIdx1D = alpaka::mapIdx<1u>(blockThreadIdx, blockThreadExtent)[0u];


//         auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
//         auto idx1D = alpaka::mapIdx<1u>(idx, extent)[0u];

//         sdata[blockThreadIdx1D] = uCurrBuf[idx1D];


//         // Each kernel executes one element
//         double const r_x = dt / (dx * dx);
//         double const r_y = dt / (dy * dy);

//         if(blockThreadIdx[0] > 0 && blockThreadIdx[0] < blockThreadExtent[0] - 1u && blockThreadIdx[1] > 0
//            && blockThreadIdx[1] < blockThreadExtent[1] - 1u)
//         {
//             uNextBuf[idx1D] = sdata[blockThreadIdx1D] * (1.0 - 2.0 * r_x - 2.0 * r_y)
//                               + sdata[blockThreadIdx1D - 1] * r_x + sdata[blockThreadIdx1D + 1] * r_x
//                               + sdata[blockThreadIdx1D - blockThreadExtent[1]] * r_y
//                               + sdata[blockThreadIdx1D + blockThreadExtent[1]] * r_y;
//         }
//         else if(idx[0] > 0 && idx[0] < extent[0] - 1u && idx[1] > 0 && idx[1] < extent[1] - 1u)
//         {
//             uNextBuf[idx1D] = uCurrBuf[idx1D] * (1.0 - 2.0 * r_x - 2.0 * r_y) + uCurrBuf[idx1D - 1] * r_x
//                               + uCurrBuf[idx1D + 1] * r_x + uCurrBuf[idx1D - extent[1]] * r_y
//                               + uCurrBuf[idx1D + extent[1]] * r_y;
//         }
//     }
// };
template<size_t T_ChunkSize1D>
struct HeatEquationKernel
{
    template<typename TAcc, typename T_Halo>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        double const* const uCurrBuf,
        double* const uNextBuf,
        T_Halo const halo,
        double const dx,
        double const dt) const -> void
    {
        auto& sdata(alpaka::declareSharedVar<double[T_ChunkSize1D], __COUNTER__>(acc));

        // Get extents(dimensions)
        auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
        auto const chunkSize = T_ChunkSize1D - halo;
        auto const numElementsThread = (chunkSize - 1) / blockThreadExtent + 1;

        // Get indexes
        auto const gridBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
        auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
        auto const blockStartIdx = gridBlockIdx * chunkSize;
        auto const threadStartIdx = blockThreadIdx * numElementsThread;


        for(auto i = blockThreadIdx; i < T_ChunkSize1D; i += blockThreadExtent)
        {
            sdata[i] = uCurrBuf[blockStartIdx + i];
        }

        alpaka::syncBlockThreads(acc);

        // Each kernel executes one element
        double const r_x = dt / (dx * dx);

        for(uint i = 0; i < numElementsThread; i++)
        {
            uNextBuf[blockStartIdx + threadStartIdx + i + 1] = sdata[1 + threadStartIdx + i] * (1.0 - 2.0 * r_x)
                                                               + sdata[1 + threadStartIdx + i - 1] * r_x
                                                               + sdata[1 + threadStartIdx + i + 1] * r_x;
        }
    }
};

//! Exact solution to the test problem
//! u_t(x, t) = u_xx(x, t), x in [0, 1], t in [0, T]
//! u(0, t) = u(1, t) = 0
//! u(x, 0) = sin(pi * x)
//! u(0, y) = sin(pi * y)
//!
//! \param x value of x
//! \param x value of y
//! \param t value of t
// auto exactSolution(double const x, double const y, double const t) -> double
// {
//     constexpr double pi = 3.14159265358979323846;
//     return std::exp(-pi * pi * t) * std::sin(pi * x) + std::exp(-pi * pi * t) * std::sin(pi * y);
// }
auto exactSolution(double const x, double const t) -> double
{
    constexpr double pi = 3.14159265358979323846;
    return std::exp(-pi * pi * t) * std::sin(pi * x);
}

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
    using Dim = alpaka::DimInt<1u>;
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
    constexpr uint32_t numNodesX = 1000;
    constexpr uint32_t haloSize = 2;

    // constexpr uint32_t numNodesY = 500;

    constexpr uint32_t numTimeSteps = 10000;
    constexpr double tMax = 0.001;
    // x in [0, 1], t in [0, tMax]
    constexpr double dx = 1.0 / static_cast<double>(numNodesX + haloSize - 1);
    // constexpr double dy = 1.0 / static_cast<double>(numNodesY - 1);
    constexpr double dt = tMax / static_cast<double>(numTimeSteps - 1);

    // Check the stability condition
    constexpr double r = dt / (dx * dx);
    if constexpr(r > 0.5)
    {
        std::cerr << "Stability condition check failed: dt/dx^2 = " << r << ", it is required to be <= 0.5\n";
        return EXIT_FAILURE;
    }

    // constexpr alpaka::Vec<Dim, Idx> extent{numNodesY, numNodesX};
    constexpr Idx extent{numNodesX + haloSize};

    // Initialize host-buffer
    // This buffer holds the calculated values
    auto uNextBufHost = alpaka::allocBuf<double, Idx>(devHost, extent);
    // This buffer will hold the current values (used for the next step)
    auto uCurrBufHost = alpaka::allocBuf<double, Idx>(devHost, extent);

    double* const pCurrHost = std::data(uCurrBufHost);
    double* const pNextHost = std::data(uNextBufHost);

    // Accelerator buffer
    using BufAcc = alpaka::Buf<Acc, double, Dim, Idx>;
    BufAcc uNextBufAcc{alpaka::allocBuf<double, Idx>(devAcc, extent)};
    BufAcc uCurrBufAcc{alpaka::allocBuf<double, Idx>(devAcc, extent)};


    double* pCurrAcc = std::data(uCurrBufAcc);
    double* pNextAcc = std::data(uNextBufAcc);

    // Apply initial conditions for the test problem
    // for(uint32_t j = 0; j < numNodesY; j++)
    // {
    //     for(uint32_t i = 0; i < numNodesX; i++)
    //     {
    //         pCurrHost[j * extent[1] + i] = exactSolution(i * dx, j * dy, 0.0);
    //     }
    // }
    for(uint32_t i = 0; i < extent; i++)
    {
        pCurrHost[i] = exactSolution(i * dx, 0.0);
    }

    // Get valid workdiv for the given problem
    // static constexpr alpaka::Vec<Dim, Idx> elemPerThread{1, 1};
    static constexpr Idx elemPerThread{1};

    // Appropriate block size for your Acc
    // TODO ask can be auto inferred from alpaka config
    // static constexpr alpaka::Vec<Dim, Idx> blockSize{1, 1};
    // static constexpr Idx blockSize{1};

    // static constexpr alpaka::Vec<Dim, Idx> blockCount{
    //     (extent[0] - 1) / blockSize[0] + 1,
    //     (extent[1] - 1) / blockSize[1] + 1};
    // static constexpr Idx blockCount{(numNodesX - 1) / blockSize + 1};
    // todo check extent is divisble by chunk size
    // static_assert
    constexpr Idx chunkSize{50};
    constexpr Idx numChunks{(numNodesX - 1) / chunkSize + 1};

    auto const deviceProperties = alpaka::getAccDevProps<Acc>(devAcc);
    auto const maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];

    auto const threadsPerBlock = std::min({maxThreadsPerBlock, chunkSize});

    alpaka::WorkDivMembers<Dim, Idx> workDiv_manual{numChunks, threadsPerBlock, elemPerThread};


    // HeatEquationKernel<chunkSize.prod()> heatEqKernel;
    HeatEquationKernel<chunkSize + haloSize> heatEqKernel;


    // auto const& bundeledKernel = alpaka::KernelBundle(heatEqKernel, pCurrAcc, pNextAcc, extent, dx, dy, dt);
    // auto const& bundeledKernel = alpaka::KernelBundle(heatEqKernel, pCurrAcc, pNextAcc, extent, dx, dt);

    // Let alpaka calculate good block and grid sizes given our full problem extent
    // auto const workDiv = alpaka::getValidWorkDivForKernel<Acc>(devAcc, bundeledKernel, extent, elemPerThread);

    // Select queue
    using QueueProperty = alpaka::NonBlocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    QueueAcc queue1{devAcc};
    QueueAcc queue2{devAcc};

    // Copy host -> device
    alpaka::memcpy(queue1, uCurrBufAcc, uCurrBufHost);
    // Copy to the buffer for next as well to have boundary values set
    alpaka::memcpy(queue1, uNextBufAcc, uCurrBufAcc);
    alpaka::wait(queue1);

    // PngCreator createPng;

    for(uint32_t step = 0; step < numTimeSteps; step++)
    {
        // Compute next values
        // alpaka::exec<Acc>(queue1, workDiv, heatEqKernel, pCurrAcc, pNextAcc, extent, dx, dy, dt);
        alpaka::exec<Acc>(queue1, workDiv_manual, heatEqKernel, pCurrAcc, pNextAcc, haloSize, dx, dt);

        if(step % 100 == 0)
        {
            alpaka::wait(queue2);
            // createPng(step, pCurrHost, extent);
        }

        // We assume the boundary conditions are constant and so these values
        // do not need to be updated.
        // So we just swap next to curr (shallow copy)
        alpaka::wait(queue1);
        std::swap(pCurrAcc, pNextAcc);
        if(step % 100 == 0)
        {
            alpaka::memcpy(queue2, uCurrBufHost, uCurrBufAcc);
        }
    }

    // Copy device -> host
    alpaka::memcpy(queue1, uNextBufHost, uNextBufAcc);
    alpaka::wait(queue1);

    // Calculate error
    double maxError = 0.0;
    // for(uint32_t j = 0; j < numNodesY; j++)
    // {
    //     for(uint32_t i = 0; i < numNodesX; i++)
    //     {
    //         auto const error = std::abs(pNextHost[j * extent[1] + i] - exactSolution(i * dx, j * dy, tMax));
    //         maxError = std::max(maxError, error);
    //     }
    // }


    for(uint32_t i = 0; i < extent; i++)
    {
        auto const error = std::abs(pNextHost[i] - exactSolution(i * dx, tMax));
        maxError = std::max(maxError, error);
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
        std::cout << "Execution results incorrect: error = " << maxError << " (the grid resolution may be too low)"
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

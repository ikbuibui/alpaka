/* Copyright 2020 Benjamin Worpitz, Matthias Werner, Jakob Krude, Sergei Bastrakov, Bernhard Manfred Gruber,
 * Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#include "BoundaryKernel.hpp"
#include "StencilKernel.hpp"
#include "analyticalSolution.hpp"

#ifdef PNGWRITER_ENABLED
#    include "writeImage.hpp"
#endif

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <utility>

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
    // get suitable device for this Acc
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    // simulation defines
    // Parameters
    constexpr alpaka::Vec<Dim, Idx> numNodes{64, 64}; // {Y, X}
    constexpr alpaka::Vec<Dim, Idx> haloSize{2, 2};
    constexpr alpaka::Vec<Dim, Idx> extent = numNodes + haloSize;

    constexpr uint32_t numTimeSteps = 100;
    constexpr double tMax = 0.001;
    // x, y in [0, 1], t in [0, tMax]
    constexpr double dx = 1.0 / static_cast<double>(extent[1] - 1);
    constexpr double dy = 1.0 / static_cast<double>(extent[0] - 1);
    constexpr double dt = tMax / static_cast<double>(numTimeSteps);

    // Check the stability condition
    constexpr double r = dt / std::min(dx * dx, dy * dy);
    if constexpr(r > 0.5)
    {
        std::cerr << "Stability condition check failed: dt/min(dx^2,dy^2) = " << r
                  << ", it is required to be <= 0.5\n";
        return EXIT_FAILURE;
    }

    // Initialize host-buffer
    // This buffer will hold the current values (used for the next step)
    auto uBufHost = alpaka::allocBuf<double, Idx>(devHost, extent);

    // Accelerator buffer
    auto uCurrBufAcc = alpaka::allocBuf<double, Idx>(devAcc, extent);
    auto uNextBufAcc = alpaka::allocBuf<double, Idx>(devAcc, extent);

    // Set buffer to initial conditions
    initalizeBuffer(uBufHost, dx, dy);

    // Select queue
    using QueueProperty = alpaka::NonBlocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    QueueAcc dumpQueue{devAcc};
    QueueAcc computeQueue{devAcc};

    // Copy host -> device
    alpaka::memcpy(computeQueue, uCurrBufAcc, uBufHost);

    // Define a workdiv for the given problem
    constexpr alpaka::Vec<Dim, Idx> elemPerThread{1, 1};
    // Appropriate chunk size to split your problem for your Acc
    constexpr alpaka::Vec<Dim, Idx> chunkSize{16u, 16u};
    constexpr auto chunkSizeWithHalo = chunkSize + haloSize;

    auto const maxThreadsPerBlock = alpaka::getAccDevProps<Acc>(devAcc).m_blockThreadExtentMax; //@TODO
    auto const threadsPerBlock = maxThreadsPerBlock.prod() < chunkSize.prod() ? maxThreadsPerBlock : chunkSize;

    constexpr alpaka::Vec<Dim, Idx> numChunks{
        (extent[0] - haloSize[0] - 1) / chunkSize[0] + 1,
        (extent[1] - haloSize[1] - 1) / chunkSize[1] + 1};

    static_assert(
        (extent[0] - haloSize[0]) % chunkSize[0] == 0 && (extent[1] - haloSize[1]) % chunkSize[1] == 0,
        "Domain must be divisible by chunk size");

    auto const pitchCurrAcc{alpaka::getPitchesInBytes(uCurrBufAcc)};

    alpaka::WorkDivMembers<Dim, Idx> workDiv_manual{numChunks, threadsPerBlock, elemPerThread};

    StencilKernel<chunkSizeWithHalo.prod()> stencilKernel;
    BoundaryKernel boundaryKernel;

    for(uint32_t step = 1; step <= numTimeSteps; ++step)
    {
        // Compute next values
        alpaka::exec<Acc>(
            computeQueue,
            workDiv_manual,
            stencilKernel,
            uCurrBufAcc.data(),
            uNextBufAcc.data(),
            chunkSize,
            pitchCurrAcc,
            dx,
            dy,
            dt);

        // apply boundaries
        alpaka::exec<Acc>(
            computeQueue,
            workDiv_manual,
            boundaryKernel,
            uNextBufAcc.data(),
            chunkSize,
            pitchCurrAcc,
            step,
            dx,
            dy,
            dt);

#ifdef PNGWRITER_ENABLED
        if((step - 1) % 100 == 0) // even steps will have currBufHost and PCurr pointing to same buffer
        {
            alpaka::memcpy(dumpQueue, uBufHost, uCurrBufAcc);
            alpaka::wait(dumpQueue);
            writeImage(step - 1, uBufHost);
        }
#endif

        // So we just swap next and curr (shallow copy)
        alpaka::wait(computeQueue);
        std::swap(uNextBufAcc, uCurrBufAcc);
    }

    // Copy device -> host
    alpaka::memcpy(dumpQueue, uBufHost, uCurrBufAcc);
    alpaka::wait(dumpQueue);

    // Validate
    auto const [resultIsCorrect, maxError] = validateSolution(uBufHost, extent, dx, dy, tMax);

    if(resultIsCorrect)
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

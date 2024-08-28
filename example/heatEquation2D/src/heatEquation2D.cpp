/* Copyright 2020 Benjamin Worpitz, Matthias Werner, Jakob Krude, Sergei Bastrakov, Bernhard Manfred Gruber,
 * Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#include "BoundaryKernel.hpp"
#include "StencilKernel.hpp"
#include "analyticalSolution.hpp"
#include "helpers.hpp"
#include "simulationDefines.hpp"

#ifdef PNGWRITER_ENABLED
#    include "writeImage.hpp"
#endif

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <algorithm>
#include <cassert>
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

    // check validity of simulation setup
    if(!isStable(dx, dy, dt) || !isValidChunking(numNodes, chunkSize))
    {
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
    alpaka::Queue<Acc, alpaka::NonBlocking> queue{devAcc};

    // Copy host -> device
    alpaka::memcpy(queue, uCurrBufAcc, uBufHost);
    alpaka::wait(queue);

    StencilKernel stencilKernel;
    BoundaryKernel boundaryKernel;

    // Get max threads that can be run in a block for this kernel
    auto const kernelFunctionAttributes = alpaka::getFunctionAttributes<Acc>(
        devAcc,
        stencilKernel,
        alpaka::experimental::getMdSpan(uCurrBufAcc),
        alpaka::experimental::getMdSpan(uNextBufAcc),
        chunkSize,
        dx,
        dy,
        dt);
    auto const maxThreadsPerBlock = kernelFunctionAttributes.maxThreadsPerBlock;

    auto const threadsPerBlock
        = maxThreadsPerBlock < chunkSize.prod() ? alpaka::Vec<Dim, Idx>{maxThreadsPerBlock, 1} : chunkSize;

    alpaka::WorkDivMembers<Dim, Idx> workDiv_manual{numChunks, threadsPerBlock, elemPerThread};

    // Simulate
    for(uint32_t step = 1; step <= numTimeSteps; ++step)
    {
        // Compute next values
        alpaka::exec<Acc>(
            queue,
            workDiv_manual,
            stencilKernel,
            alpaka::experimental::getMdSpan(uCurrBufAcc),
            alpaka::experimental::getMdSpan(uNextBufAcc),
            chunkSize,
            dx,
            dy,
            dt);

        // apply boundaries
        alpaka::exec<Acc>(
            queue,
            workDiv_manual,
            boundaryKernel,
            alpaka::experimental::getMdSpan(uNextBufAcc),
            chunkSize,
            step,
            dx,
            dy,
            dt);

#ifdef PNGWRITER_ENABLED
        if((step - 1) % 100 == 0)
        {
            alpaka::memcpy(queue, uBufHost, uCurrBufAcc);
            alpaka::wait(queue);
            writeImage(step - 1, uBufHost);
        }
#endif

        // So we just swap next and curr (shallow copy)
        alpaka::wait(queue);
        std::swap(uNextBufAcc, uCurrBufAcc);
    }

    // Copy device -> host
    alpaka::memcpy(queue, uBufHost, uCurrBufAcc);
    alpaka::wait(queue);

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

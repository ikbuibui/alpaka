/* Copyright 2024 Benjamin Worpitz, Matthias Werner, Jakob Krude, Sergei
 * Bastrakov, Bernhard Manfred Gruber, Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#include "InitializeBufferKernel.hpp"
#include "analyticalSolution.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>

//! Each kernel computes the next step for one point.
//! Therefore the number of threads should be equal to numNodesX.
//! Every time step the kernel will be executed numNodesX-times
//! After every step the curr-buffer will be set to the calculated values
//! from the next-buffer.
//!
//! In standard projects, you typically do not execute the code with any
//! available accelerator. Instead, a single accelerator is selected once from
//! the active accelerators and the kernels are executed with the selected
//! accelerator only. If you use the example as the starting point for your
//! project, you can rename the example() function to main() and move the
//! accelerator tag to the function body.
template<typename TAccTag>
auto example(TAccTag const&) -> int
{
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

    constexpr alpaka::Vec<Dim, Idx> numNodes{64, 64};
    constexpr alpaka::Vec<Dim, Idx> haloSize{2, 2};
    constexpr alpaka::Vec<Dim, Idx> extent = numNodes + haloSize;

    // x, y in [0, 1], t in [0, tMax]
    constexpr double dx = 1.0 / static_cast<double>(extent[1] - 1);
    constexpr double dy = 1.0 / static_cast<double>(extent[0] - 1);

    // Initialize host-buffer
    auto uBufHost = alpaka::allocBuf<double, Idx>(devHost, extent);

    // // Accelerator buffer
    auto uBufAcc = alpaka::allocBuf<double, Idx>(devAcc, extent);

    // Create queue
    alpaka::Queue<Acc, alpaka::NonBlocking> queue{devAcc};

    // Set buffer to initial conditions
    InitializeBufferKernel initBufferKernel;
    // Define a workdiv for the given problem
    constexpr alpaka::Vec<Dim, Idx> elemPerThread{1, 1};

    alpaka::KernelCfg<Acc> const kernelCfg = {extent, elemPerThread};
    auto const pitchCurrAcc{alpaka::getPitchesInBytes(uBufAcc)};

    auto workDiv = alpaka::getValidWorkDiv(kernelCfg, devAcc, initBufferKernel, uBufAcc.data(), pitchCurrAcc, dx, dy);

    alpaka::exec<Acc>(queue, workDiv, initBufferKernel, uBufAcc.data(), pitchCurrAcc, dx, dy);

    // Copy device -> host
    alpaka::memcpy(queue, uBufHost, uBufAcc);
    alpaka::wait(queue);

    // Validate
    auto const [resultIsCorrect, maxError] = validateSolution(uBufHost, dx, dy, 0.0);

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
    // If you would like to execute it for a single accelerator only you can use
    // the following code.
    //  \code{.cpp}
    //  auto tag = TagCpuSerial;
    //  return example(tag);
    //  \endcode
    //
    // valid tags:
    //   TagCpuSerial, TagGpuHipRt, TagGpuCudaRt, TagCpuOmp2Blocks,
    //   TagCpuTbbBlocks, TagCpuOmp2Threads, TagCpuSycl, TagCpuTbbBlocks,
    //   TagCpuThreads, TagFpgaSyclIntel, TagGenericSycl, TagGpuSyclIntel
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}

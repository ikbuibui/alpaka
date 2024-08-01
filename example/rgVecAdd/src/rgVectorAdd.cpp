/* Copyright 2023 Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber, Jan Stephan, Luca Ferragina,
 *                Aurora Perego
 * SPDX-License-Identifier: ISC
 */

#include "./rgMemCpy.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <redGrapes/redGrapes.hpp>
#include <redGrapes/resource/access/io.hpp>
#include <redGrapes/resource/ioresource.hpp>
#include <redGrapes/resource/resource.hpp>

#include <iostream>
#include <random>

//! A vector addition kernel.
class VectorAddKernel
{
public:
    //! The kernel entry point.
    //!
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator to be executed on.
    //! \param A The first source vector.
    //! \param B The second source vector.
    //! \param C The destination vector.
    //! \param numElements The number of elements.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TElem, typename TIdx>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TElem const* const A,
        TElem const* const B,
        TElem* const C,
        TIdx const& numElements) const -> void
    {
        static_assert(alpaka::Dim<TAcc>::value == 1, "The VectorAddKernel expects 1-dimensional indices!");

        TIdx const gridThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        TIdx const threadElemExtent(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
        TIdx const threadFirstElemIdx(gridThreadIdx * threadElemExtent);

        if(threadFirstElemIdx < numElements)
        {
            // Calculate the number of elements to compute in this thread.
            // The result is uniform for all but the last thread.
            TIdx const threadLastElemIdx(threadFirstElemIdx + threadElemExtent);
            TIdx const threadLastElemIdxClipped((numElements > threadLastElemIdx) ? threadLastElemIdx : numElements);

            for(TIdx i(threadFirstElemIdx); i < threadLastElemIdxClipped; ++i)
            {
                C[i] = A[i] + B[i];
            }
        }
    }
};

auto main() -> int
{
    auto rg = redGrapes::init();

    using TTask = decltype(rg)::RGTask;

    // Define the index domain
    using Dim = alpaka::DimInt<1u>;
    using Idx = std::size_t;

    // Define the accelerator
    //
    // It is possible to choose from a set of accelerators:
    // - AccGpuCudaRt
    // - AccGpuHipRt
    // - AccCpuThreads
    // - AccCpuOmp2Threads
    // - AccCpuOmp2Blocks
    // - AccCpuTbbBlocks
    // - AccCpuSerial
    // using Acc = alpaka::AccCpuSerial<Dim, Idx>;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    using DevAcc = alpaka::Dev<Acc>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    // Defines the synchronization behavior of a queue
    //
    // choose between Blocking and NonBlocking
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;

    // Select a device
    auto const platform = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platform, 0);

    // Create a queue on the device
    QueueAcc queue0(devAcc);
    QueueAcc queue1(devAcc);
    QueueAcc queue2(devAcc);


    // Define the work division
    Idx const numElements(123456);
    Idx const elementsPerThread(8u);
    alpaka::Vec<Dim, Idx> const extent(numElements);

    // Let alpaka calculate good block and grid sizes given our full problem extent
    alpaka::WorkDivMembers<Dim, Idx> const workDiv(alpaka::getValidWorkDiv<Acc>(
        devAcc,
        extent,
        elementsPerThread,
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted));

    // Define the buffer element type
    using Data = std::uint32_t;

    // Get the host device for allocating memory on the host.
    using DevHost = alpaka::DevCpu;
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    // Allocate 3 host memory buffers
    using BufHost = alpaka::Buf<DevHost, Data, Dim, Idx>;


    auto rg_bufHostA = redGrapes::IOResource<BufHost, TTask>(alpaka::allocBuf<Data, Idx>(devHost, extent));
    auto rg_bufHostB = redGrapes::IOResource<BufHost, TTask>(alpaka::allocBuf<Data, Idx>(devHost, extent));
    auto rg_bufHostC = redGrapes::IOResource<BufHost, TTask>(alpaka::allocBuf<Data, Idx>(devHost, extent));


    // Initialize the host vectors

    rg.emplace_task(
        [](auto rg_bufHostA, auto rg_bufHostB, auto rg_bufHostC)
        {
            Data* const pBufHostA(alpaka::getPtrNative(*rg_bufHostA.obj));
            Data* const pBufHostB(alpaka::getPtrNative(*rg_bufHostB.obj));
            Data* const pBufHostC(alpaka::getPtrNative(*rg_bufHostC.obj));
            // C++14 random generator for uniformly distributed numbers in {1,..,42}

            std::random_device rd{};
            std::default_random_engine eng{rd()};
            std::uniform_int_distribution<Data> dist(1, 42);

            for(Idx i(0); i < numElements; ++i)
            {
                pBufHostA[i] = dist(eng);
                pBufHostB[i] = dist(eng);
                pBufHostC[i] = 0;
            }
        },
        rg_bufHostA.write(),
        rg_bufHostB.write(),
        rg_bufHostC.write());


    // Allocate 3 buffers on the accelerator
    using BufAcc = alpaka::Buf<DevAcc, Data, Dim, Idx>;


    auto rg_bufAccA = redGrapes::IOResource<BufAcc, TTask>(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    auto rg_bufAccB = redGrapes::IOResource<BufAcc, TTask>(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    auto rg_bufAccC = redGrapes::IOResource<BufAcc, TTask>(alpaka::allocBuf<Data, Idx>(devAcc, extent));

    // Copy Host -> Acc

    auto helper = RGHelper(rg);
    helper.rgMemCpy(queue0, rg_bufAccA, rg_bufHostA);

    rg.emplace_task(
        [queue0](auto rg_bufHostA, auto rg_bufAccA) mutable
        { alpaka::memcpy(queue0, *rg_bufAccA.obj, *rg_bufHostA.obj); },
        rg_bufHostA.read(),
        rg_bufAccA.write());

    rg.emplace_task(
        [queue1](auto rg_bufHostB, auto rg_bufAccB) mutable
        { alpaka::memcpy(queue1, *rg_bufAccB.obj, *rg_bufHostB.obj); },
        rg_bufHostB.read(),
        rg_bufAccB.write());

    rg.emplace_task(
        [queue2](auto rg_bufHostC, auto rg_bufAccC) mutable
        { alpaka::memcpy(queue2, *rg_bufAccC.obj, *rg_bufHostC.obj); },
        rg_bufHostC.read(),
        rg_bufAccC.write());


    // Enqueue the kernel execution task
    rg.emplace_task(
        [queue0, workDiv, numElements](auto rg_bufAccA, auto rg_bufAccB, auto rg_bufAccC) mutable
        {
            // Instantiate the kernel function object
            VectorAddKernel kernel;

            // Create the kernel execution task.
            auto const taskKernel = alpaka::createTaskKernel<Acc>(
                workDiv,
                kernel,
                alpaka::getPtrNative(*rg_bufAccA.obj),
                alpaka::getPtrNative(*rg_bufAccB.obj),
                alpaka::getPtrNative(*rg_bufAccC.obj),
                numElements);
            alpaka::enqueue(queue0, taskKernel);
        },
        rg_bufAccA.read(),
        rg_bufAccB.read(),
        rg_bufAccC.write());

    // Copy back the result
    rg.emplace_task(
        [queue0](auto rg_bufHostC, auto rg_bufAccC) mutable
        { alpaka::memcpy(queue0, *rg_bufHostC.obj, *rg_bufAccC.obj); },
        rg_bufHostC.write(),
        rg_bufAccC.read());


    int falseResults = 0;
    static constexpr int MAX_PRINT_FALSE_RESULTS = 20;

    rg.emplace_task(
        [&falseResults](auto rg_bufHostA, auto rg_bufHostB, auto rg_bufHostC)
        {
            Data* const pBufHostA(alpaka::getPtrNative(*rg_bufHostA.obj));
            Data* const pBufHostB(alpaka::getPtrNative(*rg_bufHostB.obj));
            Data* const pBufHostC(alpaka::getPtrNative(*rg_bufHostC.obj));

            for(Idx i(0u); i < numElements; ++i)
            {
                Data const& val(pBufHostC[i]);
                Data const correctResult(pBufHostA[i] + pBufHostB[i]);
                if(val != correctResult)
                {
                    if(falseResults < MAX_PRINT_FALSE_RESULTS)
                        std::cerr << "C[" << i << "] == " << val << " != " << correctResult << std::endl;
                    ++falseResults;
                }
            }
        },
        rg_bufHostA.read(),
        rg_bufHostB.read(),
        rg_bufHostC.read());


    if(falseResults == 0)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Found " << falseResults << " false results, printed no more than " << MAX_PRINT_FALSE_RESULTS
                  << "\n"
                  << "Execution results incorrect!" << std::endl;
        return EXIT_FAILURE;
    }
}

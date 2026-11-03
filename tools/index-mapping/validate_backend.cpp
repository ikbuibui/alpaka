/* SPDX-License-Identifier: MPL-2.0 */
// Real kernel -> device trace buffers -> host buffers -> explorer comparison.
// build.py --single-header exports this as a Godbolt program with an HTTP header include.
#ifndef ALPAKA_MAPPING_EMBEDDED
#    ifdef ALPAKA_MAPPING_SINGLE_HEADER
#        include "alpaka.hpp"
#    else
#        include <alpaka/alpaka.hpp>
#    endif
#    define ALPAKA_MAPPING_NO_MAIN
#    include "engine.cpp"
#endif

#include <type_traits>

#if defined(ALPAKA_MAPPING_REQUIRE_CUDA) && !ALPAKA_LANG_CUDA
#    error "CUDA validator must be compiled in CUDA mode (nvcc -x cu)"
#endif

namespace backendValidation
{
    using namespace alpaka;
    constexpr uint32_t capacity = 256; // Plus an unwritten guard record per worker.
    constexpr uint32_t sentinel = 0xffff'ffffu;

    template<uint32_t D>
    using V = Vec<uint32_t, D>;

    struct DefaultTraversal
    {
    };

    template<typename Traversal, typename Layout>
    ALPAKA_FN_ACC auto indices(auto const& acc, auto group, auto range)
    {
        if constexpr(std::is_same_v<Traversal, DefaultTraversal>)
        {
            if constexpr(std::is_same_v<Layout, onAcc::layout::Optimized>)
                return onAcc::makeIdxMap(acc, group, range);
            else
            {
                using Default =
                    typename explorer::ContainerInfo<decltype(onAcc::makeIdxMap(acc, group, range))>::Traversal;
                return onAcc::makeIdxMap(acc, group, range, Default{}, Layout{});
            }
        }
        else
            return onAcc::makeIdxMap(acc, group, range, Traversal{}, Layout{});
    }

    template<int Worker>
    ALPAKA_FN_ACC constexpr auto group()
    {
        if constexpr(Worker == 0)
            return onAcc::worker::threadsInGrid;
        else if constexpr(Worker == 1)
            return onAcc::worker::linearThreadsInGrid;
        else if constexpr(Worker == 2)
            return onAcc::worker::blocksInGrid;
        else if constexpr(Worker == 3)
            return onAcc::worker::linearBlocksInGrid;
        else if constexpr(Worker == 4)
            return onAcc::worker::threadsInBlock;
        else if constexpr(Worker == 5)
            return onAcc::worker::linearThreadsInBlock;
        else
            return onAcc::worker::allThreads;
    }

    template<uint32_t D>
    struct Domain
    {
        V<D> begin, end, stride, tile;
        V<D> outerBegin, outerEnd, outerStride, innerBegin, innerEnd, innerStride, directBegin, directEnd;
    };

    template<uint32_t D>
    struct WorkerInfo
    {
        uint32_t uid, gridUid;
        uint32_t block[D], thread[D], gridThread[D], blocks[D], threads[D];
    };

    template<uint32_t D>
    struct Record
    {
        uint32_t uid, outerIteration, innerIteration, inBounds, linear;
        uint32_t origin[D], local[D], index[D];
    };

    template<uint32_t D>
    ALPAKA_FN_ACC void copy(uint32_t (&out)[D], alpaka::concepts::Vector auto const& in)
    {
        for(uint32_t d = 0; d < D; ++d)
            out[d] = in[d];
    }

    template<uint32_t D, typename Buffer>
    struct Recorder
    {
        Buffer records;
        Domain<D> domain;
        uint32_t uid, count = 0;

        ALPAKA_FN_ACC void write(V<D> const& origin, V<D> const& local, uint32_t outer, uint32_t inner)
        {
            auto const slot = count++;
            if(slot >= capacity)
                return; // Count still records overflow; never write out of bounds.
            auto const global = origin + local;
            Record<D> record{};
            record.uid = uid;
            record.outerIteration = outer;
            record.innerIteration = inner;
            record.inBounds = 1;
            for(uint32_t d = 0; d < D; ++d)
                if(global[d] < domain.begin[d] || global[d] >= domain.end[d])
                    record.inBounds = 0;
            record.linear = record.inBounds ? linearize(domain.end, global) : sentinel;
            copy(record.origin, origin);
            copy(record.local, local);
            copy(record.index, global);
            records[Vec{uid, slot}] = record;
        }
    };

    template<
        uint32_t D,
        bool Hierarchy,
        int Worker,
        typename Traversal,
        typename Layout,
        typename OuterTraversal,
        typename OuterLayout>
    struct RecordKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            auto records,
            auto counts,
            auto workers,
            Domain<D> domain,
            uint32_t numWorkers) const
        {
            auto const block = acc.getIdxWithin(onAcc::origin::grid, onAcc::unit::blocks);
            auto const thread = acc.getIdxWithin(onAcc::origin::block, onAcc::unit::threads);
            auto const blocks = acc.getExtentsOf(onAcc::origin::grid, onAcc::unit::blocks);
            auto const threads = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::threads);
            // Block-major UID, NOT linearize(gridExtent, globalThreadIdx) in multidimensional launches.
            uint32_t const uid = linearize(blocks, block) * threads.product() + linearize(threads, thread);
            if(uid >= numWorkers)
                return; // The missing worker sentinel makes this a host validation failure.
            WorkerInfo<D> info{};
            info.uid = uid;
            auto const gridThread = acc.getIdxWithin(onAcc::origin::grid, onAcc::unit::threads);
            info.gridUid = linearize(acc.getExtentsOf(onAcc::origin::grid, onAcc::unit::threads), gridThread);
            copy(info.block, block);
            copy(info.thread, thread);
            copy(info.gridThread, gridThread);
            copy(info.blocks, blocks);
            copy(info.threads, threads);
            workers[Vec{uid}] = info;
            Recorder<D, ALPAKA_TYPEOF(records)> recorder{records, domain, uid};
            if constexpr(Hierarchy)
            {
                uint32_t outer = 0;
                for(auto origin : indices<OuterTraversal, OuterLayout>(
                        acc,
                        onAcc::worker::blocksInGrid,
                        IdxRange{domain.outerBegin, domain.outerEnd, domain.outerStride}))
                {
                    uint32_t inner = 0;
                    for(auto local : indices<Traversal, Layout>(
                            acc,
                            group<Worker>(),
                            IdxRange{domain.innerBegin, domain.innerEnd, domain.innerStride}))
                        recorder.write(origin, local, outer, inner++);
                    ++outer;
                }
            }
            else
            {
                uint32_t inner = 0;
                for(auto index : indices<Traversal, Layout>(
                        acc,
                        group<Worker>(),
                        IdxRange{domain.directBegin, domain.directEnd, domain.stride}))
                    recorder.write(V<D>::fill(0), index, 0, inner++);
            }
            counts[Vec{uid}] = recorder.count;
        }
    };

    template<uint32_t D>
    V<D> asVec(uint32_t const (&values)[D])
    {
        auto result = V<D>::fill(0);
        for(uint32_t d = 0; d < D; ++d)
            result[d] = values[d];
        return result;
    }

    void check(bool valid, std::string const& context)
    {
        if(!valid)
            throw std::runtime_error(context);
    }

    template<typename Traversal>
    constexpr int traversalId()
    {
        return std::is_same_v<Traversal, DefaultTraversal>        ? 0
               : std::is_same_v<Traversal, onAcc::traverse::Flat> ? 1
                                                                  : 2;
    }

    template<typename Layout>
    constexpr int layoutId()
    {
        return std::is_same_v<Layout, onAcc::layout::Optimized> ? 0
               : std::is_same_v<Layout, onAcc::layout::Strided> ? 1
                                                                : 2;
    }

    std::string request(explorer::Config const& c)
    {
        std::ostringstream out;
        out << "6 " << c.dim << ' ' << c.spec << ' ' << c.api << ' ' << c.executor << ' ' << c.hierarchy << ' '
            << c.worker << ' ' << c.traversal << ' ' << c.layout << ' ' << c.outerTraversal << ' ' << c.outerLayout
            << ' ' << c.outerWorker << ' ' << c.maxThreadsPerBlock << ' ' << c.frameType << ' ' << c.innerRangeMode
            << ' ' << c.outerDomainBounds << ' ' << c.directDomainBounds << ' ' << c.tileOuterStride;
        for(auto const* values :
            {&c.blocks,
             &c.threads,
             &c.frames,
             &c.tile,
             &c.begin,
             &c.end,
             &c.stride,
             &c.frameExtent,
             &c.outerBegin,
             &c.outerEnd,
             &c.outerStride,
             &c.innerBegin,
             &c.innerEnd,
             &c.innerStride,
             &c.directBegin,
             &c.directEnd})
            for(auto value : *values)
                out << ' ' << value;
        return out.str();
    }

    template<
        uint32_t D,
        bool Hierarchy,
        int Worker = Hierarchy ? 4 : 0,
        typename Traversal = DefaultTraversal,
        typename Layout = onAcc::layout::Optimized,
        typename OuterTraversal = Traversal,
        typename OuterLayout = Layout>
    void runCase(
        auto device,
        auto executor,
        bool json,
        char const* label,
        int variation = 0,
        bool corrupt = false,
        int frameLaunch = 0)
    {
        // Edit these small dimensions to reproduce a configuration in Godbolt.
        explorer::Config c{};
        c.version = 6;
        c.spec = frameLaunch != 0;
        c.frameType = frameLaunch == 2;
        c.maxThreadsPerBlock = device.getDeviceProperties().maxThreadsPerBlock;
        c.dim = D;
        c.hierarchy = Hierarchy;
        c.worker = Worker;
        c.api = std::is_same_v<ALPAKA_TYPEOF(executor), exec::GpuCuda> ? 1 : 0;
        c.executor = c.api ? 4 : 1;
        c.traversal = traversalId<Traversal>();
        c.layout = layoutId<Layout>();
        c.outerTraversal = traversalId<OuterTraversal>();
        c.outerLayout = layoutId<OuterLayout>();
        if constexpr(D == 1)
        {
            c.blocks = {2};
            c.threads = {c.api ? 4 : 1};
            c.tile = {8};
            c.begin = {2};
            c.end = {17};
            c.stride = {3};
        }
        else
        {
            c.blocks = {2, 2};
            c.threads = {c.api ? 2 : 1, c.api ? 2 : 1};
            c.tile = {2, 4};
            c.begin = {1, 2};
            c.end = {4, 9};
            c.stride = {2, 3};
        }
        c.frames = c.blocks;
        c.frameExtent = D == 1 ? std::vector<int>{4} : std::vector<int>{4, 2};
        if(frameLaunch)
        {
            // Independent expected launch, deliberately using frame extents different from mapping tiles.
            c.blocks = c.api ? c.frames : std::vector<int>(D, 1);
            c.threads = c.api ? c.frameExtent : std::vector<int>(D, 1);
        }
        if(variation == 1)
            c.begin = c.end; // Empty domain: workers must still report zero.
        if(variation == 2)
            c.end = D == 1 ? std::vector<int>{3} : std::vector<int>{2, 3}; // Idle workers.
        if(variation == 3)
            c.blocks.assign(D, 1); // Multiple outer iterations per block.
        c.directBegin = c.begin;
        c.directEnd = c.end;
        c.outerBegin = c.begin;
        c.outerEnd = c.end;
        c.outerStride = c.tile;
        c.innerBegin.assign(D, 0);
        c.innerEnd = c.tile;
        c.innerStride.assign(D, 1);
        if(variation == 4 || variation == 5)
        {
            c.innerRangeMode = 1;
            c.outerBegin.assign(D, 1);
            c.outerEnd.assign(D, 10);
            c.outerStride.assign(D, 4);
            c.innerBegin.assign(D, 1);
            c.innerEnd.assign(D, 6);
            c.innerStride.assign(D, 2);
            if(variation == 5)
                c.innerBegin = c.innerEnd;
        }
        c = explorer::parse(request(c).c_str());
        auto const blocks = explorer::vec<D>(c.blocks), threads = explorer::vec<D>(c.threads);
        auto const numWorkers = blocks.product() * threads.product();
        Domain<D> const domain{
            explorer::vec<D>(c.begin),
            explorer::vec<D>(c.end),
            explorer::vec<D>(c.stride),
            explorer::vec<D>(c.tile),
            explorer::vec<D>(c.outerBegin),
            explorer::vec<D>(c.outerEnd),
            explorer::vec<D>(c.outerStride),
            c.innerRangeMode ? explorer::vec<D>(c.innerBegin) : V<D>::fill(0),
            c.innerRangeMode ? explorer::vec<D>(c.innerEnd) : explorer::vec<D>(c.tile),
            c.innerRangeMode ? explorer::vec<D>(c.innerStride) : V<D>::fill(1),
            explorer::vec<D>(c.directBegin),
            explorer::vec<D>(c.directEnd)};
        static_assert(std::is_trivially_copyable_v<Record<D>> && std::is_trivially_copyable_v<WorkerInfo<D>>);
        auto recordsH = onHost::allocHost<Record<D>>(Vec{numWorkers, capacity + 1});
        auto countsH = onHost::allocHost<uint32_t>(Vec{numWorkers});
        auto workersH = onHost::allocHost<WorkerInfo<D>>(Vec{numWorkers});
        auto recordsD = onHost::allocLike(device, recordsH);
        auto countsD = onHost::allocLike(device, countsH);
        auto workersD = onHost::allocLike(device, workersH);
        auto queue = device.makeQueue();
        onHost::memset(queue, recordsD, 0xff);
        onHost::memset(queue, countsD, 0xff);
        onHost::memset(queue, workersD, 0xff);
        auto enqueue = [&](auto const& launch)
        {
            queue.enqueue(
                launch,
                RecordKernel<D, Hierarchy, Worker, Traversal, Layout, OuterTraversal, OuterLayout>{},
                recordsD,
                countsD,
                workersD,
                domain,
                numWorkers);
        };
        if(frameLaunch == 2)
        {
            if constexpr(D == 1)
                enqueue(onHost::FrameSpec{explorer::vec<D>(c.frames), CVec<uint32_t, 4u>{}, executor});
            else
                enqueue(onHost::FrameSpec{explorer::vec<D>(c.frames), CVec<uint32_t, 4u, 2u>{}, executor});
        }
        else if(frameLaunch)
            enqueue(onHost::FrameSpec{explorer::vec<D>(c.frames), explorer::vec<D>(c.frameExtent), executor});
        else
            enqueue(onHost::ThreadSpec{blocks, threads, executor});
        // CUDA writes device memory; host buffers are read only after copy-back and wait.
        onHost::memcpy(queue, recordsH, recordsD);
        onHost::memcpy(queue, countsH, countsD);
        onHost::memcpy(queue, workersH, workersD);
        onHost::wait(queue);
        // Test-only fault injection: prove validation reads and checks the returned host buffer.
        if(corrupt)
            recordsH[Vec{0u, 0u}].uid ^= 1u;

        std::vector<uint32_t> coverage(domain.end.product(), 0);
        if(json)
            std::cout << "{\"case\":\"" << label << "\",\"request\":\"" << request(c) << "\",\"workers\":[";
        for(uint32_t uid = 0; uid < numWorkers; ++uid)
        {
            auto const context = std::string{label} + " worker " + std::to_string(uid) + ": ";
            auto const block = mapToND(blocks, uid / threads.product());
            auto const thread = mapToND(threads, uid % threads.product());
            auto const info = workersH[Vec{uid}];
            check(info.uid == uid, context + "worker did not write its UID");
            check(asVec(info.block) == block && asVec(info.thread) == thread, context + "worker coordinates differ");
            check(asVec(info.blocks) == blocks && asVec(info.threads) == threads, context + "launch extents differ");
            check(asVec(info.gridThread) == block * threads + thread, context + "grid thread index differs");
            check(info.gridUid == linearize(blocks * threads, block * threads + thread), context + "grid UID differs");
            auto const count = countsH[Vec{uid}];
            check(count != sentinel && count <= capacity, context + "missing count or trace overflow");
            check(recordsH[Vec{uid, capacity}].uid == sentinel, context + "guard record overwritten");

            auto acc = onAcc::Acc{Dict{
                DictEntry(layer::block, onAcc::cpu::GenericLayer{block, blocks}),
                DictEntry(layer::thread, onAcc::cpu::GenericLayer{thread, threads}),
                DictEntry(object::api, getApi(device)),
                DictEntry(object::deviceKind, getDeviceKind(device))}};
            auto expectedInner = explorer::map<D>(
                acc,
                Worker,
                IdxRange{
                    Hierarchy ? domain.innerBegin : domain.directBegin,
                    Hierarchy ? domain.innerEnd : domain.directEnd,
                    Hierarchy ? domain.innerStride : domain.stride},
                c.traversal,
                c.layout);
            std::vector<V<D>> origins{V<D>::fill(0)};
            if constexpr(Hierarchy)
                origins = explorer::map<D>(
                              acc,
                              2,
                              IdxRange{domain.outerBegin, domain.outerEnd, domain.outerStride},
                              c.outerTraversal,
                              c.outerLayout)
                              .indices;
            check(count == origins.size() * expectedInner.indices.size(), context + "assignment count differs");
            if(json)
            {
                std::cout << (uid ? "," : "") << "{\"id\":" << info.uid << ",\"block\":";
                explorer::jsonVec(std::cout, asVec(info.block));
                std::cout << ",\"thread\":";
                explorer::jsonVec(std::cout, asVec(info.thread));
                std::cout << ",\"assignments\":[";
            }
            uint32_t slot = 0;
            for(uint32_t oi = 0; oi < origins.size(); ++oi)
                for(uint32_t ii = 0; ii < expectedInner.indices.size(); ++ii, ++slot)
                {
                    auto const record = recordsH[Vec{uid, slot}];
                    auto const origin = origins[oi], local = expectedInner.indices[ii], index = origin + local;
                    bool inBounds = true;
                    for(uint32_t d = 0; d < D; ++d)
                        inBounds = inBounds && index[d] >= domain.begin[d] && index[d] < domain.end[d];
                    auto const at = context + "assignment " + std::to_string(slot) + ": ";
                    check(
                        record.uid == uid && record.outerIteration == oi && record.innerIteration == ii,
                        at + "UID or iteration differs");
                    check(
                        asVec(record.origin) == origin && asVec(record.local) == local && asVec(record.index) == index,
                        at + "index sequence differs from explorer");
                    check(record.inBounds == uint32_t(inBounds), at + "bounds guard differs");
                    check(
                        record.linear == (inBounds ? linearize(domain.end, index) : sentinel),
                        at + "linear offset differs");
                    if(inBounds)
                        ++coverage[record.linear];
                    if(json)
                    {
                        std::cout << (slot ? "," : "") << "{\"outerIteration\":" << record.outerIteration
                                  << ",\"innerIteration\":" << record.innerIteration << ",\"origin\":";
                        explorer::jsonVec(std::cout, asVec(record.origin));
                        std::cout << ",\"local\":";
                        explorer::jsonVec(std::cout, asVec(record.local));
                        std::cout << ",\"index\":";
                        explorer::jsonVec(std::cout, asVec(record.index));
                        std::cout << ",\"inBounds\":" << (record.inBounds ? "true" : "false") << ",\"linear\":";
                        if(record.inBounds)
                            std::cout << record.linear;
                        else
                            std::cout << "null";
                        std::cout << '}';
                    }
                }
            if(json)
                std::cout << "]}";
        }
        // Independent completeness check; do not let adapter/backend agreement hide missing work.
        for(uint32_t linear = 0; linear < domain.end.product(); ++linear)
        {
            auto const index = mapToND(domain.end, linear);
            bool eligible = true;
            uint32_t multiplicity = 1;
            for(uint32_t d = 0; d < D; ++d)
            {
                eligible = eligible && index[d] >= domain.begin[d] && index[d] < domain.end[d]
                           && (Hierarchy
                               || (index[d] >= domain.directBegin[d] && index[d] < domain.directEnd[d]
                                   && (index[d] - domain.directBegin[d]) % domain.stride[d] == 0));
                if constexpr(Hierarchy)
                {
                    uint32_t combinations = 0;
                    for(auto outer = domain.outerBegin[d]; outer < domain.outerEnd[d]; outer += domain.outerStride[d])
                        for(auto inner = domain.innerBegin[d]; inner < domain.innerEnd[d];
                            inner += domain.innerStride[d])
                            combinations += outer + inner == index[d];
                    multiplicity *= combinations;
                }
            }
            eligible = eligible && multiplicity > 0;
            check((coverage[linear] > 0) == eligible, std::string{label} + ": domain coverage differs");
            if constexpr((Hierarchy && Worker == 4) || (!Hierarchy && Worker == 0))
                check(
                    coverage[linear] == (eligible ? multiplicity : 0u),
                    std::string{label} + ": duplicate or missing assignment");
        }
        if(json)
            std::cout << "]}\n";
        else
            std::cout << "PASS " << label << " (" << D << "D, " << numWorkers << " workers)\n";
    }

    template<uint32_t D>
    void dimensionSuite(auto device, auto executor, bool json, bool corrupt)
    {
        using namespace onAcc;
        runCase<D, false>(device, executor, json, "direct-default", 0, corrupt);
        runCase<D, true>(device, executor, json, "hierarchy-default");
        runCase<D, true>(device, executor, json, "hierarchy-independent-strided-ranges", 4);
        runCase<D, true>(device, executor, json, "hierarchy-empty-inner-range", 5);
        runCase<D, false>(device, executor, json, "frame-direct", 0, false, true);
        runCase<D, true>(device, executor, json, "frame-hierarchy-independent-tiles", 0, false, true);
        runCase<D, false>(device, executor, json, "cvec-frame-direct", 0, false, 2);
        runCase<D, true>(device, executor, json, "cvec-frame-hierarchy-independent-tiles", 0, false, 2);
        runCase<D, false, 0, DefaultTraversal, layout::Contiguous>(
            device,
            executor,
            json,
            "direct-default-contiguous");
        runCase<D, false, 0, traverse::Flat, layout::Strided>(device, executor, json, "direct-flat-strided");
        runCase<D, false, 0, traverse::Flat, layout::Contiguous>(device, executor, json, "direct-flat-contiguous");
        runCase<D, false, 0, traverse::Tiled, layout::Strided>(device, executor, json, "direct-tiled-strided");
        runCase<D, false, 0, traverse::Tiled, layout::Contiguous>(device, executor, json, "direct-tiled-contiguous");
        runCase<D, true, 4, traverse::Flat, layout::Strided>(device, executor, json, "hierarchy-flat-strided");
        runCase<D, true, 4, traverse::Tiled, layout::Contiguous>(device, executor, json, "hierarchy-tiled-contiguous");
        runCase<D, true, 4, traverse::Flat, layout::Contiguous, traverse::Tiled, layout::Strided>(
            device,
            executor,
            json,
            "hierarchy-mixed-policies",
            3);
        runCase<D, false, 4>(device, executor, json, "repeated-worker-identities");
        runCase<D, false, 6>(device, executor, json, "all-threads-duplicate-work");
        runCase<D, false>(device, executor, json, "empty-range", 1);
        runCase<D, true>(device, executor, json, "empty-hierarchy", 1);
        runCase<D, false>(device, executor, json, "idle-workers", 2);
        if constexpr(D == 1)
            runCase<D, false, 1>(device, executor, json, "linearized-threads");
    }

    int run(auto api, auto kind, auto executor, bool json, bool corrupt)
    {
        auto selector = onHost::makeDeviceSelector(api, kind);
        if(!selector.isAvailable())
        {
            std::cerr << "SKIP: selected backend has no available device; no validation ran.\n";
            return 77;
        }
        auto device = selector.makeDevice(0);
        dimensionSuite<1>(device, executor, json, corrupt);
        dimensionSuite<2>(device, executor, json, corrupt);
        std::cerr << "PASS: 43 real " << ALPAKA_TYPEOF(api)::getName()
                  << " kernel launches; per-worker buffers copied back and matched against explorer.\n";
        return 0;
    }
} // namespace backendValidation

int main(int argc, char** argv)
{
    bool json = false, corrupt = false;
    bool cuda = ALPAKA_LANG_CUDA != 0;
    for(int i = 1; i < argc; ++i)
    {
        std::string option = argv[i];
        if(option == "--json")
            json = true;
        else if(option == "--inject-error")
            corrupt = true;
        else if(option == "--cpu")
            cuda = false;
        else if(option == "--cuda")
            cuda = true;
        else
        {
            std::cerr << "Usage: validator [--cpu|--cuda] [--json] [--inject-error]\n";
            return 2;
        }
    }
    try
    {
        if(cuda)
        {
#if ALPAKA_LANG_CUDA
            return backendValidation::run(
                alpaka::api::cuda,
                alpaka::deviceKind::nvidiaGpu,
                alpaka::exec::gpuCuda,
                json,
                corrupt);
#else
            std::cerr << "SKIP: CUDA backend was not compiled. Compile this program with nvcc.\n";
            return 77;
#endif
        }
        return backendValidation::run(
            alpaka::api::host,
            alpaka::deviceKind::cpu,
            alpaka::exec::cpuSerial,
            json,
            corrupt);
    }
    catch(std::exception const& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

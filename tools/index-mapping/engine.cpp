/* SPDX-License-Identifier: MPL-2.0 */
// The adapter enumerates worker contexts. All worker-index and iteration logic is alpaka's.
#if defined(ALPAKA_MAPPING_EMBEDDED)
// The Godbolt export already included its selected single header.
#elif defined(ALPAKA_MAPPING_SINGLE_HEADER)
#    include "alpaka.hpp"
#else
#    include <alpaka/api/host/IdxLayer.hpp>
#    include <alpaka/onAcc/Acc.hpp>
#    include <alpaka/onAcc/WorkerGroup.hpp>
#    include <alpaka/onAcc/internal/interfaceImpl.hpp>
#    include <alpaka/onHost/FrameSpec.hpp>
#endif

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef ALPAKA_MAPPING_BUILD
#    define ALPAKA_MAPPING_BUILD "unversioned"
#endif

namespace explorer
{
    using namespace alpaka;
    constexpr size_t maxAssignments = 32768;

    struct Config
    {
        int version, dim, spec, api, executor, hierarchy, worker, traversal, layout;
        int outerTraversal, outerLayout, outerWorker;
        std::vector<int> blocks, threads, frames, tile, begin, end, stride;
    };

    void require(bool valid, char const* message)
    {
        if(!valid)
            throw std::runtime_error(message);
    }

    Config parse(char const* input)
    {
        require(input != nullptr && std::char_traits<char>::length(input) < 2048, "Invalid request size");
        Config c{};
        std::istringstream in{input};
        require(
            bool(
                in >> c.version >> c.dim >> c.spec >> c.api >> c.executor >> c.hierarchy >> c.worker >> c.traversal
                >> c.layout >> c.outerTraversal >> c.outerLayout >> c.outerWorker),
            "Missing configuration fields");
        require(c.version == 1, "Unsupported protocol version");
        require(c.dim == 1 || c.dim == 2, "Only 1D and 2D are supported");
        require(c.spec >= 0 && c.spec <= 1 && c.api >= 0 && c.api <= 3, "Unsupported specification or API");
        require(c.executor >= 0 && c.executor <= 6, "Unsupported executor");
        require(
            c.hierarchy >= 0 && c.hierarchy <= 1 && c.worker >= 0 && c.worker <= 6,
            "Unsupported hierarchy or worker");
        require(
            c.traversal >= 0 && c.traversal <= 2 && c.outerTraversal >= 0 && c.outerTraversal <= 2,
            "Unsupported traversal");
        require(c.layout >= 0 && c.layout <= 2 && c.outerLayout >= 0 && c.outerLayout <= 2, "Unsupported layout");
        require(c.outerWorker >= 0 && c.outerWorker <= 1, "Unsupported outer worker");
        for(auto* field : {&c.blocks, &c.threads, &c.frames, &c.tile, &c.begin, &c.end, &c.stride})
            for(int d = 0; d < c.dim; ++d)
            {
                int value;
                require(bool(in >> value), "Missing vector components");
                require(value >= 0 && value <= 64, "Vector components must be between 0 and 64");
                field->push_back(value);
            }
        in >> std::ws;
        require(in.eof(), "Unexpected trailing fields");
        int workers = 1, cells = 1, threads = 1;
        for(int d = 0; d < c.dim; ++d)
        {
            require(
                c.blocks[d] > 0 && c.threads[d] > 0 && c.frames[d] > 0 && c.tile[d] > 0 && c.stride[d] > 0,
                "Launch extents, tile extents, and strides must be positive");
            require(c.begin[d] <= c.end[d] && c.end[d] > 0, "Range must satisfy 0 <= begin <= end and end > 0");
            workers *= c.blocks[d] * c.threads[d];
            threads *= c.threads[d];
            cells *= c.end[d];
        }
        require(workers <= 128 && cells <= 1024, "Prototype limit: 128 workers and 1024 data cells");
        require(c.api != 0 || threads == 1, "Host executors require one thread per block");
        require(
            c.executor == 0 || (c.api == 0 && c.executor <= 3) || c.executor == c.api + 3,
            "Executor is incompatible with the selected API");
        return c;
    }

    template<uint32_t D>
    using V = Vec<uint32_t, D>;

    template<uint32_t D>
    V<D> vec(std::vector<int> const& values)
    {
        auto result = V<D>::fill(0);
        for(uint32_t d = 0; d < D; ++d)
            result[d] = static_cast<uint32_t>(values[d]);
        return result;
    }

    void jsonVec(std::ostream& out, auto const& v)
    {
        out << '[';
        for(uint32_t d = 0; d < ALPAKA_TYPEOF(v)::dim(); ++d)
            out << (d ? "," : "") << v[d];
        out << ']';
    }

    template<typename F>
    void withWorker(int worker, F&& f)
    {
        using namespace onAcc;
        switch(worker)
        {
        case 0:
            f(worker::threadsInGrid);
            break;
        case 1:
            f(worker::linearThreadsInGrid);
            break;
        case 2:
            f(worker::blocksInGrid);
            break;
        case 3:
            f(worker::linearBlocksInGrid);
            break;
        case 4:
            f(worker::threadsInBlock);
            break;
        case 5:
            f(worker::linearThreadsInBlock);
            break;
        case 6:
            f(worker::allThreads);
            break;
        }
    }

    // Introspection is intentionally exhaustive: new alpaka iterator/layout types require review.
    template<typename T>
    struct ContainerInfo;

    template<typename R, typename W, typename L, typename S>
    struct ContainerInfo<onAcc::FlatIdxContainer<R, W, L, S>>
    {
        using Traversal = onAcc::traverse::Flat;
        static constexpr char const* name = "flat";
    };

    template<typename R, typename W, typename L, typename S>
    struct ContainerInfo<onAcc::TiledIdxContainer<R, W, L, S>>
    {
        using Traversal = onAcc::traverse::Tiled;
        static constexpr char const* name = "tiled";
    };

    template<typename L>
    char const* layoutName(L)
    {
        static_assert(
            std::is_same_v<L, onAcc::layout::Strided> || std::is_same_v<L, onAcc::layout::Contiguous>,
            "New alpaka layout type: update and verify the explorer adapter");
        return std::is_same_v<L, onAcc::layout::Strided> ? "strided" : "contiguous";
    }

    template<uint32_t D>
    struct Mapping
    {
        std::vector<V<D>> indices;
        std::string workerIndex, workerCount;
        char const* traversal = "";
        char const* layout = "";
    };

    template<uint32_t D, typename A, typename R>
    Mapping<D> map(A const& acc, int worker, R const& range, int traversal, int layout)
    {
        Mapping<D> result;
        withWorker(
            worker,
            [&](auto group)
            {
                if constexpr(ALPAKA_TYPEOF(group.getThreadIdx(acc))::dim() != D)
                    throw std::runtime_error(
                        "Linear worker groups require a 1D range; choose a multidimensional group for this 2D range");
                else
                {
                    std::ostringstream index, count;
                    jsonVec(index, group.getThreadIdx(acc));
                    jsonVec(count, group.getThreadCount(acc));
                    result.workerIndex = index.str();
                    result.workerCount = count.str();
                    auto collect = [&](auto const& container)
                    {
                        result.traversal = ContainerInfo<ALPAKA_TYPEOF(container)>::name;
                        for(auto idx : container)
                        {
                            require(
                                result.indices.size() < maxAssignments,
                                "Mapping exceeds prototype iteration limit");
                            result.indices.push_back(idx);
                        }
                    };
                    auto withTraversal = [&](auto traverse)
                    {
                        using namespace onAcc;
                        switch(layout)
                        {
                        case 0:
                            collect(makeIdxMap(acc, group, range, traverse, layout::optimized));
                            break;
                        case 1:
                            collect(makeIdxMap(acc, group, range, traverse, layout::strided));
                            break;
                        case 2:
                            collect(makeIdxMap(acc, group, range, traverse, layout::contiguous));
                            break;
                        }
                    };
                    if(traversal == 0 && layout == 0)
                        collect(onAcc::makeIdxMap(acc, group, range)); // Exercise actual overload defaults.
                    else if(traversal == 0)
                    {
                        using DefaultContainer = decltype(onAcc::makeIdxMap(acc, group, range));
                        withTraversal(typename ContainerInfo<DefaultContainer>::Traversal{});
                    }
                    else if(traversal == 1)
                        withTraversal(onAcc::traverse::flat);
                    else
                        withTraversal(onAcc::traverse::tiled);
                }
            });
        result.layout = layout == 1   ? "strided"
                        : layout == 2 ? "contiguous"
                                      : layoutName(onAcc::internal::adjustMapping(acc));
        return result;
    }

    template<uint32_t D>
    void mappingInfo(std::ostream& out, Mapping<D> const& mapping)
    {
        out << "{\"workerIndex\":" << mapping.workerIndex << ",\"workerCount\":" << mapping.workerCount
            << ",\"traversal\":\"" << mapping.traversal << "\",\"layout\":\"" << mapping.layout << "\"}";
    }

    template<uint32_t D, typename Api, typename Device>
    std::string trace(Config const& c, Api api, Device device)
    {
        auto threadSpec = onHost::ThreadSpec{vec<D>(c.blocks), vec<D>(c.threads)};
        auto blocks = threadSpec.getNumBlocks(), threads = threadSpec.getNumThreads(), tile = vec<D>(c.tile);
        auto begin = vec<D>(c.begin), end = vec<D>(c.end), stride = vec<D>(c.stride);
        // A FrameSpec is retained as input, NOT converted using a made-up backend heuristic.
        auto frameSpec = onHost::FrameSpec{vec<D>(c.frames), tile};
        std::ostringstream out;
        out << "{\"build\":\"" ALPAKA_MAPPING_BUILD "\",\"schema\":1,\"dim\":" << D << ",\"launchResolution\":\""
            << (c.spec ? "explicit-assumption" : "explicit-thread-spec") << "\",\"frameCapacity\":";
        jsonVec(out, frameSpec.getNumFrames() * frameSpec.getFrameExtents());
        out << ",\"workers\":[";
        size_t assignments = 0;
        for(uint32_t b = 0; b < blocks.product(); ++b)
            for(uint32_t t = 0; t < threads.product(); ++t)
            {
                auto blockIdx = mapToND(blocks, b), threadIdx = mapToND(threads, t);
                auto acc = onAcc::Acc{Dict{
                    DictEntry(layer::block, onAcc::cpu::GenericLayer{blockIdx, blocks}),
                    DictEntry(layer::thread, onAcc::cpu::GenericLayer{threadIdx, threads}),
                    DictEntry(object::api, api),
                    DictEntry(object::deviceKind, device)}};
                auto inner = map<D>(
                    acc,
                    c.worker,
                    IdxRange{
                        c.hierarchy ? V<D>::fill(0) : begin,
                        c.hierarchy ? tile : end,
                        c.hierarchy ? V<D>::fill(1) : stride},
                    c.traversal,
                    c.layout);
                Mapping<D> outer;
                if(c.hierarchy)
                    outer = map<D>(
                        acc,
                        c.outerWorker ? 3 : 2,
                        IdxRange{begin, end, tile},
                        c.outerTraversal,
                        c.outerLayout);
                else
                    outer.indices.push_back(V<D>::fill(0));
                out << (b || t ? "," : "") << "{\"id\":" << b * threads.product() + t << ",\"block\":";
                jsonVec(out, blockIdx);
                out << ",\"thread\":";
                jsonVec(out, threadIdx);
                out << ",\"inner\":";
                mappingInfo(out, inner);
                if(c.hierarchy)
                {
                    out << ",\"outer\":";
                    mappingInfo(out, outer);
                }
                out << ",\"assignments\":[";
                bool first = true;
                for(size_t oi = 0; oi < outer.indices.size(); ++oi)
                    for(size_t ii = 0; ii < inner.indices.size(); ++ii)
                    {
                        require(++assignments <= maxAssignments, "Trace exceeds prototype assignment limit");
                        auto global = outer.indices[oi] + inner.indices[ii];
                        bool inBounds = true;
                        for(uint32_t d = 0; d < D; ++d)
                            inBounds = inBounds && global[d] >= begin[d] && global[d] < end[d];
                        out << (first ? "" : ",") << "{\"outerIteration\":" << oi << ",\"innerIteration\":" << ii
                            << ",\"origin\":";
                        first = false;
                        jsonVec(out, outer.indices[oi]);
                        out << ",\"local\":";
                        jsonVec(out, inner.indices[ii]);
                        out << ",\"index\":";
                        jsonVec(out, global);
                        out << ",\"inBounds\":" << (inBounds ? "true" : "false") << ",\"linear\":";
                        if(inBounds)
                            out << linearize(end, global);
                        else
                            out << "null";
                        out << '}';
                    }
                out << "]}";
            }
        out << "]}";
        return out.str();
    }

    template<uint32_t D>
    std::string dispatch(Config const& c)
    {
        switch(c.api)
        {
        case 0:
            return trace<D>(c, api::host, deviceKind::cpu);
        case 1:
            return trace<D>(c, api::cuda, deviceKind::nvidiaGpu);
        case 2:
            return trace<D>(c, api::hip, deviceKind::amdGpu);
        default:
            return trace<D>(c, api::oneApi, deviceKind::intelGpu);
        }
    }
} // namespace explorer

extern "C" char const* mapping_trace(char const* input)
{
    static std::string result;
    try
    {
        auto c = explorer::parse(input);
        result = c.dim == 1 ? explorer::dispatch<1>(c) : explorer::dispatch<2>(c);
    }
    catch(std::exception const& error)
    {
        result = std::string{"{\"error\":\""} + error.what() + "\"}";
    }
    return result.c_str();
}

#if !defined(__EMSCRIPTEN__) && !defined(ALPAKA_MAPPING_NO_MAIN)
int main()
{
    std::string line;
    while(std::getline(std::cin, line))
        std::cout << mapping_trace(line.c_str()) << '\n';
}
#endif

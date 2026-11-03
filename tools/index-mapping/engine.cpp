/* SPDX-License-Identifier: MPL-2.0 */
// The adapter enumerates worker contexts. All worker-index and iteration logic is alpaka's.
#if defined(ALPAKA_MAPPING_EMBEDDED)
// The Godbolt export already included its selected single header.
#elif defined(ALPAKA_MAPPING_SINGLE_HEADER)
#    include "alpaka.hpp"
#else
#    include <alpaka/api/host/IdxLayer.hpp>
#    include <alpaka/api/util.hpp>
#    include <alpaka/onAcc/Acc.hpp>
#    include <alpaka/onAcc/WorkerGroup.hpp>
#    include <alpaka/onAcc/internal/interfaceImpl.hpp>
#    include <alpaka/onHost/FrameSpec.hpp>
#endif

#include <array>
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
        int maxThreadsPerBlock = 1024, frameType = 0, innerRangeMode = 0;
        int outerDomainBounds = 0, directDomainBounds = 0, tileOuterStride = 0;
        std::vector<int> blocks, threads, frames, tile, begin, end, stride, frameExtent;
        std::vector<int> outerBegin, outerEnd, outerStride, innerBegin, innerEnd, innerStride, directBegin, directEnd;
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
        require(c.version >= 1 && c.version <= 6, "Unsupported protocol version");
        int legacyCoupling = 1, legacyOuterTileStride = 0;
        if(c.version >= 4)
            require(
                bool(in >> c.maxThreadsPerBlock >> c.frameType >> c.innerRangeMode),
                "Missing device profile or inner range mode");
        else
        {
            if(c.version >= 2)
                require(bool(in >> legacyCoupling >> c.maxThreadsPerBlock), "Missing device profile fields");
            if(c.version >= 3)
                require(bool(in >> c.frameType), "Missing frame extent type");
            require(legacyCoupling == 0 || legacyCoupling == 1, "Invalid legacy tile coupling");
        }
        if(c.version >= 6)
            require(
                bool(in >> c.outerDomainBounds >> c.directDomainBounds >> c.tileOuterStride),
                "Missing mapping range links");
        else if(c.version == 5)
            require(bool(in >> c.outerDomainBounds >> legacyOuterTileStride), "Missing legacy outer range links");
        for(auto link : {c.outerDomainBounds, c.directDomainBounds, c.tileOuterStride, legacyOuterTileStride})
            require(link == 0 || link == 1, "Invalid mapping range link");
        require(c.frameType == 0 || c.frameType == 1, "Unsupported frame extent type");
        require(c.innerRangeMode == 0 || c.innerRangeMode == 1, "Unsupported inner range mode");
        require(c.maxThreadsPerBlock > 0 && c.maxThreadsPerBlock <= 4096, "Device thread limit must be 1 to 4096");
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
        if(c.version >= 2)
            for(int d = 0; d < c.dim; ++d)
            {
                int value;
                require(bool(in >> value) && value > 0 && value <= 64, "Frame extents must be 1 to 64");
                c.frameExtent.push_back(value);
            }
        else
            c.frameExtent = c.tile; // Legacy links used one value for both concepts.
        if(c.version >= 4)
        {
            for(auto* field : {&c.outerBegin, &c.outerEnd, &c.outerStride, &c.innerBegin, &c.innerEnd, &c.innerStride})
                for(int d = 0; d < c.dim; ++d)
                {
                    int value;
                    require(bool(in >> value) && value >= 0 && value <= 64, "Range components must be 0 to 64");
                    field->push_back(value);
                }
        }
        else
        {
            // Flatten old coupling once on import; no live connection to launch frames remains.
            if(c.spec && legacyCoupling)
                c.tile = c.frameExtent;
            c.outerBegin = c.begin;
            c.outerEnd = c.end;
            c.outerStride = c.tile;
            c.innerBegin.assign(c.dim, 0);
            c.innerEnd = c.tile;
            c.innerStride.assign(c.dim, 1);
        }
        if(c.version >= 6)
        {
            for(auto* field : {&c.directBegin, &c.directEnd})
                for(int d = 0; d < c.dim; ++d)
                {
                    int value;
                    require(bool(in >> value) && value >= 0 && value <= 64, "Direct range components must be 0 to 64");
                    field->push_back(value);
                }
        }
        else
        {
            c.directBegin = c.begin;
            c.directEnd = c.end;
            if(c.hierarchy && legacyOuterTileStride)
                c.outerStride = c.tile;
        }
        if(!c.hierarchy && c.directDomainBounds)
        {
            c.directBegin = c.begin;
            c.directEnd = c.end;
        }
        if(c.hierarchy && c.outerDomainBounds)
        {
            c.outerBegin = c.begin;
            c.outerEnd = c.end;
        }
        if(c.hierarchy && c.tileOuterStride)
            c.tile = c.outerStride;
        in >> std::ws;
        require(in.eof(), "Unexpected trailing fields");
        int workers = 1, cells = 1, threads = 1;
        for(int d = 0; d < c.dim; ++d)
        {
            require(c.blocks[d] > 0 && c.threads[d] > 0 && c.frames[d] > 0, "Launch extents must be positive");
            require(c.begin[d] <= c.end[d] && c.end[d] > 0, "Range must satisfy 0 <= begin <= end and end > 0");
            // Inactive mapping settings are retained for mode switches. Their components
            // must still be bounded integers, but only active ranges need to be iterable.
            if(c.hierarchy)
            {
                require(c.outerBegin[d] <= c.outerEnd[d], "Outer range requires begin <= end");
                require(c.outerStride[d] > 0, "Outer range strides must be positive");
                if(c.innerRangeMode)
                {
                    require(c.innerBegin[d] <= c.innerEnd[d], "Inner range requires begin <= end");
                    require(c.innerStride[d] > 0, "Inner range strides must be positive");
                }
                else
                    require(c.tile[d] > 0, "Mapping tile extents must be positive");
            }
            else
            {
                require(c.directBegin[d] <= c.directEnd[d], "Direct range requires begin <= end");
                require(c.stride[d] > 0, "Direct range strides must be positive");
            }
            workers *= c.blocks[d] * c.threads[d];
            threads *= c.threads[d];
            cells *= c.end[d];
        }
        require((c.spec || workers <= 128) && cells <= 1024, "Prototype limit: 128 workers and 1024 data cells");
        require(c.spec || c.api != 0 || threads == 1, "Host executors require one thread per block");
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

    int effectiveExecutor(Config const& c)
    {
        // Browser profile: serial-only CPU default; GPU defaults match the selected API.
        return c.executor ? c.executor : c.api == 0 ? 1 : c.api + 3;
    }

    constexpr uint32_t cudaCompileLimit
        = onHost::getMaxThreadsPerBlock(api::cuda, deviceKind::nvidiaGpu, exec::gpuCuda);
    constexpr uint32_t hipCompileLimit = onHost::getMaxThreadsPerBlock(api::hip, deviceKind::amdGpu, exec::gpuHip);
#if ALPAKA_LANG_ONEAPI
    constexpr uint32_t oneApiCompileLimit
        = onHost::getMaxThreadsPerBlock(api::oneApi, deviceKind::intelGpu, exec::oneApi);
#else
    // The oneAPI trait is gated behind the SYCL compiler. Matches api/oneApi/Api.hpp;
    // this is an explicit browser profile assumption, not a hardware query.
    constexpr uint32_t oneApiCompileLimit = 1024u;
#endif

    uint32_t compileThreadLimit(Config const& c)
    {
        if(c.api == 0)
            return 1u;
        return c.api == 1 ? cudaCompileLimit : c.api == 2 ? hipCompileLimit : oneApiCompileLimit;
    }

    // Browser controls are runtime values, but CVec values must be template arguments.
    // Compile all supported 1..64 shapes into a lookup table using alpaka's actual
    // consteval adjustment. No JS or copied runtime approximation of that algorithm.
    template<uint32_t Limit, uint32_t D, size_t I>
    consteval std::array<uint32_t, D> adjustedCVec()
    {
        if constexpr(D == 1)
        {
            constexpr auto threads = api::util::adjustToLimit<Limit, 0u, 1u>(CVec<uint32_t, I + 1u>{});
            return {threads[0]};
        }
        else
        {
            constexpr auto threads
                = api::util::adjustToLimit<Limit, 0u, 1u>(CVec<uint32_t, I / 64u + 1u, I % 64u + 1u>{});
            return {threads[0], threads[1]};
        }
    }

    template<uint32_t Limit, uint32_t D, size_t... I>
    consteval auto cvecTable(std::index_sequence<I...>)
    {
        return std::array<std::array<uint32_t, D>, sizeof...(I)>{adjustedCVec<Limit, D, I>()...};
    }

    template<uint32_t Limit, uint32_t D>
    V<D> cvecThreads(V<D> const& extents)
    {
        static constexpr auto table = cvecTable<Limit, D>(std::make_index_sequence<(D == 1 ? 64 : 4096)>{});
        auto const index = D == 1 ? extents[0] - 1u : (extents[0] - 1u) * 64u + extents[1] - 1u;
        auto result = V<D>::fill(0);
        for(uint32_t d = 0; d < D; ++d)
            result[d] = table[index][d];
        return result;
    }

    template<uint32_t D>
    auto resolveLaunch(Config const& c)
    {
        if(!c.spec)
            return std::pair{vec<D>(c.blocks), vec<D>(c.threads)};
        auto const one = V<D>::fill(1);
        auto const frames = vec<D>(c.frames), extents = vec<D>(c.frameExtent);
        // Mirror the two simple CPU rules in api/host/Device.hpp. Native device
        // implementations cannot be included in WASM. Real launch tests check parity.
        if(effectiveExecutor(c) == 1)
            return std::pair{one, one};
        if(c.api == 0)
            return std::pair{frames, one};
        if(!c.frameType)
            return std::pair{frames, api::util::adjustToLimit(extents, c.maxThreadsPerBlock)};
        auto const threads = c.api == 1   ? cvecThreads<cudaCompileLimit>(extents)
                             : c.api == 2 ? cvecThreads<hipCompileLimit>(extents)
                                          : cvecThreads<oneApiCompileLimit>(extents);
        return std::pair{frames, threads};
    }

    template<uint32_t D, typename Api, typename Device>
    std::string trace(Config const& c, Api api, Device device)
    {
        auto const [blocks, threads] = resolveLaunch<D>(c);
        auto tile = vec<D>(c.tile);
        auto begin = vec<D>(c.begin), end = vec<D>(c.end), stride = vec<D>(c.stride);
        auto frameSpec = onHost::FrameSpec{vec<D>(c.frames), vec<D>(c.frameExtent)};
        std::ostringstream out;
        out << "{\"build\":\"" ALPAKA_MAPPING_BUILD "\",\"schema\":1,\"dim\":" << D << ",\"launchResolution\":\""
            << (c.spec ? "device-profile" : "explicit-thread-spec")
            << "\",\"effectiveExecutor\":" << effectiveExecutor(c)
            << ",\"maxThreadsPerBlock\":" << c.maxThreadsPerBlock << ",\"frameType\":" << c.frameType
            << ",\"threadLimit\":"
            << (c.frameType ? compileThreadLimit(c)
                : c.api     ? c.maxThreadsPerBlock
                            : 1u)
            << ",\"blocks\":";
        jsonVec(out, blocks);
        out << ",\"threads\":";
        jsonVec(out, threads);
        out << ",\"frameCapacity\":";
        jsonVec(out, frameSpec.getNumFrames() * frameSpec.getFrameExtents());
        if(blocks.product() * threads.product() > 128u)
        {
            out << ",\"error\":\"Resolved launch exceeds prototype limit of 128 workers; launch was not clamped.\"}";
            return out.str();
        }
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
                        c.hierarchy ? (c.innerRangeMode ? vec<D>(c.innerBegin) : V<D>::fill(0))
                                    : vec<D>(c.directBegin),
                        c.hierarchy ? (c.innerRangeMode ? vec<D>(c.innerEnd) : tile) : vec<D>(c.directEnd),
                        c.hierarchy ? (c.innerRangeMode ? vec<D>(c.innerStride) : V<D>::fill(1)) : stride},
                    c.traversal,
                    c.layout);
                Mapping<D> outer;
                if(c.hierarchy)
                    outer = map<D>(
                        acc,
                        c.outerWorker ? 3 : 2,
                        IdxRange{vec<D>(c.outerBegin), vec<D>(c.outerEnd), vec<D>(c.outerStride)},
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

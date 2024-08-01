#include <alpaka/alpaka.hpp>

#include <redGrapes/redGrapes.hpp>

#include <memory>

template<typename RGType>
struct RGHelper
{
    std::shared_ptr<RGType> rg;

    RGHelper(RGType& rg) : rg{std::make_shared<RGType>(rg)}
    {
    }

    template<typename Queue, typename DestView, typename SrcView>
    auto rgMemCpy(Queue& queue, DestView&& dest, SrcView const& src) -> void
    {
        rg->emplace_task(
            [queue](auto dest, auto src) mutable { alpaka::memcpy(queue, *dest.obj, *src.obj); },
            dest.write(),
            src.read());
    }

    template<typename Queue, typename SrcView, typename DestView, typename Extents>
    auto rgMemCpy(Queue& queue, DestView&& dest, SrcView const& src, Extents const& extent) -> void
    {
        rg->emplace_task(
            [queue, extent](auto dest, auto src) mutable { alpaka::memcpy(queue, *dest.obj, *src.obj, extent); },
            dest.write(),
            src.read());
    }
};

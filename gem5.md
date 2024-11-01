* [X] event programming
* [X] new config
* [X] debug flags
* [ ] out of order

**在GEM5退出时，Dump cache中所有的line的地址和内容， 参考[Event-driven programming](https://www.gem5.org/documentation/learning_gem5/part2/events/)， 但是退出的event callback和这里展示的例子不完全一样，需要自己去阅读相关代码。**

首先我使用的CPU类型为TimeSimpleCPU,直接找他与Cache如何交互,也即是下图

![1730213552828](image/gem5/1730213552828.png)

然后找到了tags,这里可以打印所有的blk

在cache/tags/base.cc,可以打印所有的blk

![1730213976910](image/gem5/1730213976910.png)

registerExitCallback注意这个函数

![1730215799144](image/gem5/1730215799144.png)

这个是tag/base.cc

这个如何实现遍历:

```
    void forEachBlk(std::function<void(CacheBlk &)> visitor) override {
        for (CacheBlk& blk : blks) {
            visitor(blk);
        }
    }


```

首先将输入参数变为函数,接受一个blk参数,然后这个blks是blk的集合,存了所有blk,然后进行遍历

这个是base_set_assoc.h

所以可以在结束时调用这个函数,这样就可打印全部内容

```
BaseTags::BaseTags(const Params &p)
    : ClockedObject(p), blkSize(p.block_size), blkMask(blkSize - 1),
      size(p.size), lookupLatency(p.tag_latency),
      system(p.system), indexingPolicy(p.indexing_policy),
      warmupBound((p.warmup_percentage/100.0) * (p.size / p.block_size)),
      warmedUp(false), numBlocks(p.size / p.block_size),
      dataBlks(new uint8_t[p.size]), // Allocate data storage in one big chunk
      stats(*this)
{
    // registerExitCallback([this]() { cleanupRefs(); });
    registerExitCallback([this]() { DPRINTF(DumpCache,"%s",print()); });
}

```

具体就是在这个函数内会注册退出时的回调函数,只需要将里面函数换为print就行

# 1. 官方的adder例子

![1730820232374](image/diplomacy/1730820232374.png)

**首先定义参数**

```
case class UpwardParam(width: Int)
case class DownwardParam(width: Int)
case class EdgeParam(width: Int)

```

也即是INT,

**之后实现节点**

```
object AdderNodeImp extends SimpleNodeImp[DownwardParam, UpwardParam, EdgeParam, UInt] {
  def edge(pd: DownwardParam, pu: UpwardParam, p: Parameters, sourceInfo: SourceInfo) = {
    if (pd.width < pu.width) EdgeParam(pd.width) else EdgeParam(pu.width)
  }
  def bundle(e: EdgeParam) = UInt(e.width.W)
  def render(e: EdgeParam) = RenderedEdge("blue", s"width = ${e.width}")
}
```

这个edge的意思就是去协商向上传的参数与向下传的参数,最终取最小值,然后bundle是根据协商参数创建数据类型,

然后就是节点,节点主要有SourceNode,SinkNode和NexusNode,由于 `SourceNode`只沿向外边生成向下流动的参数，节点实现和之前一样。对 `AdderDriverNode`而言，类型为 `Seq[DownwardParam]`的 `widths`表示初始化该节点（`AdderDriver`）的模块时输出的数据宽度，这里使用 `Seq`是因为每个节点可能驱动多个输出，在这个例子中，每个节点会连接到加法器和monitor。SinkNode同理

最后就是Nexus节点,

加法器节点接收两个 `AdderDriverNode`的输入，并把输出传递给monitor，该节点为 `NexusNode`。`dFn`将向内边传来的向下的参数，映射到向外边的向下的参数，`uFn`将向外边的向上的参数，映射到向内边的向上的参数。

(内边可以理解为传入的参数,外边可以理解为向外传的参数)

```
class AdderDriverNode(widths: Seq[DownwardParam])(implicit valName: ValName)
  extends SourceNode(AdderNodeImp)(widths)

/** node for [[AdderMonitor]] (sink) */
class AdderMonitorNode(width: UpwardParam)(implicit valName: ValName)
  extends SinkNode(AdderNodeImp)(Seq(width))

/** node for [[Adder]] (nexus) */
class AdderNode(dFn: Seq[DownwardParam] => DownwardParam,
                uFn: Seq[UpwardParam] => UpwardParam)(implicit valName: ValName)
  extends NexusNode(AdderNodeImp)(dFn, uFn)
```

这个里面有两个模板匹配,然后最终传入的AdderNode的值为(dps和ups的head),最后将输入累加

```
class Adder(implicit p: Parameters) extends LazyModule {
  val node = new AdderNode (
    { case dps: Seq[DownwardParam] =>
      require(dps.forall(dp => dp.width == dps.head.width), "inward, downward adder widths must be equivalent")
      dps.head
    },
    { case ups: Seq[UpwardParam] =>
      require(ups.forall(up => up.width == ups.head.width), "outward, upward adder widths must be equivalent")
      ups.head
    }
  )
  lazy val module = new LazyModuleImp(this) {
    require(node.in.size >= 2)
    node.out.head._1 := node.in.unzip._1.reduce(_ + _)
  }

  override lazy val desiredName = "Adder"
}
```

主要就是设置numoutputs个驱动节点,然后给每个节点分配随机值

```

/** driver (source)
  * drives one random number on multiple outputs */
class AdderDriver(width: Int, numOutputs: Int)(implicit p: Parameters) extends LazyModule {
  val node = new AdderDriverNode(Seq.fill(numOutputs)(DownwardParam(width)))

  lazy val module = new LazyModuleImp(this) {
    // check that node parameters converge after negotiation
    val negotiatedWidths = node.edges.out.map(_.width)
    require(negotiatedWidths.forall(_ == negotiatedWidths.head), "outputs must all have agreed on same width")
    val finalWidth = negotiatedWidths.head

    // generate random addend (notice the use of the negotiated width)
    val randomAddend = FibonacciLFSR.maxPeriod(finalWidth)

    // drive signals
    node.out.foreach { case (addend, _) => addend := randomAddend }
  }

  override lazy val desiredName = "AdderDriver"
}
```

主要就是设置numoperands个监视节点,和一个adder节点,然后对比nodesum节点和nodeseq节点值的区别,送出error

```
class AdderMonitor(width: Int, numOperands: Int)(implicit p: Parameters) extends LazyModule {
  val nodeSeq = Seq.fill(numOperands) { new AdderMonitorNode(UpwardParam(width)) }
  val nodeSum = new AdderMonitorNode(UpwardParam(width))

  lazy val module = new LazyModuleImp(this) {
    val io = IO(new Bundle {
      val error = Output(Bool())
    })

    // print operation
    printf(nodeSeq.map(node => p"${node.in.head._1}").reduce(_ + p" + " + _) + p" = ${nodeSum.in.head._1}")

    // basic correctness checking
    io.error := nodeSum.in.head._1 =/= nodeSeq.map(_.in.head._1).reduce(_ + _)
  }

  override lazy val desiredName = "AdderMonitor"
}
```

最后就是顶层,顶层就是通过高阶函数将每个节点链接起来

```
class AdderTestHarness()(implicit p: Parameters) extends LazyModule {
  val numOperands = 2
  val adder = LazyModule(new Adder)
  // 8 will be the downward-traveling widths from our drivers
  val drivers = Seq.fill(numOperands) { LazyModule(new AdderDriver(width = 8, numOutputs = 2)) }
  // 4 will be the upward-traveling width from our monitor
  val monitor = LazyModule(new AdderMonitor(width = 4, numOperands = numOperands))

  // create edges via binding operators between nodes in order to define a complete graph
  drivers.foreach{ driver => adder.node := driver.node }

  drivers.zip(monitor.nodeSeq).foreach { case (driver, monitorNode) => monitorNode := driver.node }
  monitor.nodeSum := adder.node

  lazy val module = new LazyModuleImp(this) {
    // when(monitor.module.io.error) {
    //   printf("something went wrong")
    // }
  }

  override lazy val desiredName = "AdderTestHarness"
}
```

# 2. 根据rocketchip 搭建一个简单的SOC框架(基于ysyxSoC)

首先我们需要包含freechip库,有两种方法,1.直接从云端下载,2.直接导入本地的库,本实验选择第二种,基于ysyxSoC的build.sc来创建自己的sc文件,导入成功后就可以进行自己的SoC搭建

我们的SoC框架如下所示

![1730881092351](image/diplomacy/1730881092351.png)

也就是我们CPU需要一个AXI_master节点,clint,SDRAM和MROM各自需要一个AXI_slave节点,然后AXI_XBAR继承于NexusNode,可支持多个输入节点和多个输出节点

然后设备的地址空间安排如下:

| 设备  | 地址空间                |
| ----- | ----------------------- |
| clint | 0x1000_0000-0x1000_ffff |
| SDRAM | 0x8000_0000-0x9fff_ffff |
| MROM  | 0x2000_0000-0x2000_ffff |

首先创建clint的slave节点

```
class AXI4MyCLINT(address: Seq[AddressSet])(implicit p: Parameters) extends LazyModule {
  val beatBytes = 4
  val node = AXI4SlaveNode(Seq(AXI4SlavePortParameters(
    Seq(AXI4SlaveParameters(
        address       = address,
        executable    = true,
        supportsWrite = TransferSizes(1, beatBytes),
        supportsRead  = TransferSizes(1, beatBytes),
        interleavedId = Some(0))
    ),
    beatBytes  = beatBytes)))

  lazy val module = new Impl
  class Impl extends LazyModuleImp(this) {
  }
}
```

可以看到我们首先创建了slvae节点，这个节点里面有一个TransferSizes，来揭示最多可以传多少笔数据，这里是按照四笔来说的，然后在之后的LazyModuleImp有具体的实现，我们可以根据传入的node的信号的地址来读写相应的寄存器，然后SDRAM和MROM比较类似，以SDRAM为主要讲解

```
class AXI4MySDRAM(address: Seq[AddressSet])(implicit p: Parameters) extends LazyModule {
  val beatBytes = 4
  val node = AXI4SlaveNode(Seq(AXI4SlavePortParameters(
    Seq(AXI4SlaveParameters(
        address       = address,
        executable    = true,
        supportsWrite = TransferSizes(1, beatBytes),
        supportsRead  = TransferSizes(1, beatBytes),
        interleavedId = Some(0))
    ),
    beatBytes  = beatBytes)))

  lazy val module = new Impl
  class Impl extends LazyModuleImp(this) {
    val (in, _) = node.in(0)
    val sdram_bundle = IO(new SDRAMIO)

    val msdram = Module(new sdram_top_axi)
    msdram.io.clock := clock
    msdram.io.reset := reset.asBool
    msdram.io.in <> in
    sdram_bundle <> msdram.io.sdram
  }
}
```

SDRAM仍然是先创建slave节点，然后LazyModuleImp中将节点连接到msdram的输入端，这个模块是一个黑盒，这地方的好处就是一般sdram和DDR都使用IP,而现在的IP一般都是verilog,所以包裹一层黑盒

```
class MySoC(implicit p: Parameters) extends LazyModule {
  val xbar = AXI4Xbar()
  val cpu = LazyModule(new CPU(idBits = ChipLinkParam.idBits))
  val lmrom = LazyModule(new AXI4MROM(AddressSet.misaligned(0x20000000, 0x10000)))
  val lclint = LazyModule(new AXI4MyCLINT(AddressSet.misaligned(0x10000000, 0x10000)))
  val sdramAddressSet = AddressSet.misaligned(0x80000000L, 0x2000000)
  val lsdram_axi = Some(LazyModule(new AXI4MySDRAM(sdramAddressSet))) 

  List(lsdram_axi.get.node ,lmrom.node, lclint.node).map(_ := xbar)
  xbar := cpu.masterNode
  
  override lazy val module = new Impl
  class Impl extends LazyModuleImp(this) with DontTouch {

    cpu.module.reset := SynchronizerShiftReg(reset.asBool, 10) || reset.asBool
    cpu.module.slave := DontCare
    val intr_from_chipSlave = IO(Input(Bool()))
    cpu.module.interrupt := intr_from_chipSlave
    val sdramBundle = lsdram_axi.get.module.sdram_bundle
    val sdram = IO(chiselTypeOf(sdramBundle))
    sdram <> sdramBundle
  }
}
```

首先調用xbar创建XBAR,然后为每个设备分配地址空间,最后连线,也就是将slave node和xbar连线,cpu的master node 和xbar连线,之后就是实现部分,主要也是连线逻辑,然后就结束了整个SOC的创建

![1730880825683](image/diplomacy/1730880825683.png)

最后是生成的代码的一部分,可以看到正确链接

# 3.rocketchip 的AXIDelayer解析

```scala
  val node = AXI4AdapterNode()
  require (0.0 <= q && q < 1)
```

首先可以看到他创建了一个AXI4AdapterNode,这个主要就是master原封不动传进来,slave也是原封不动传进来（只可改变参数，但边不可改变）,然后q就是请求延迟的概率

然后在lazymodule定义了一个feed函数

```scala
    def feed[T <: Data](sink: IrrevocableIO[T], source: IrrevocableIO[T], noise: T): Unit = {
      // irrevocable requires that we not lower valid
      val hold = RegInit(false.B)
      when (sink.valid)  { hold := true.B }
      when (sink.fire) { hold := false.B }

      val allow = hold || ((q * 65535.0).toInt).U <= LFSRNoiseMaker(16, source.valid)
      sink.valid := source.valid && allow
      source.ready := sink.ready && allow
      sink.bits := source.bits
      when (!sink.valid) { sink.bits := noise }
    }
```

这个函数就是通过allow来截断sink和source的vaild和ready信号,allow主要有两个信号,一个是hold,另一个是比较电路,假设我们第一次使用这个,那么hold必然为false,只能通过后面的比较电路来决定allow,如果后面的也为false,则会引入噪音,直到后面条件满足,这时控制信号就会通,但是bits仍然是有噪声的

```
    def anoise[T <: AXI4BundleA](bits: T): Unit = {
      bits.id    := LFSRNoiseMaker(bits.params.idBits)
      bits.addr  := LFSRNoiseMaker(bits.params.addrBits)
      bits.len   := LFSRNoiseMaker(bits.params.lenBits)
      bits.size  := LFSRNoiseMaker(bits.params.sizeBits)
      bits.burst := LFSRNoiseMaker(bits.params.burstBits)
      bits.lock  := LFSRNoiseMaker(bits.params.lockBits)
      bits.cache := LFSRNoiseMaker(bits.params.cacheBits)
      bits.prot  := LFSRNoiseMaker(bits.params.protBits)
      bits.qos   := LFSRNoiseMaker(bits.params.qosBits)
    }
```

这个就是给ar和aw通道加noise

```
   (node.in zip node.out) foreach { case ((in, edgeIn), (out, edgeOut)) =>
      val arnoise = Wire(new AXI4BundleAR(edgeIn.bundle))
      val awnoise = Wire(new AXI4BundleAW(edgeIn.bundle))
      val wnoise  = Wire(new  AXI4BundleW(edgeIn.bundle))
      val rnoise  = Wire(new  AXI4BundleR(edgeIn.bundle))
      val bnoise  = Wire(new  AXI4BundleB(edgeIn.bundle))

      arnoise := DontCare
      awnoise := DontCare
      wnoise := DontCare
      rnoise := DontCare
      bnoise := DontCare

      anoise(arnoise)
      anoise(awnoise)

      wnoise.data := LFSRNoiseMaker(wnoise.params.dataBits)
      wnoise.strb := LFSRNoiseMaker(wnoise.params.dataBits/8)
      wnoise.last := LFSRNoiseMaker(1)(0)

      rnoise.id   := LFSRNoiseMaker(rnoise.params.idBits)
      rnoise.data := LFSRNoiseMaker(rnoise.params.dataBits)
      rnoise.resp := LFSRNoiseMaker(rnoise.params.respBits)
      rnoise.last := LFSRNoiseMaker(1)(0)

      bnoise.id   := LFSRNoiseMaker(bnoise.params.idBits)
      bnoise.resp := LFSRNoiseMaker(bnoise.params.respBits)

      feed(out.ar, in.ar, arnoise)
      feed(out.aw, in.aw, awnoise)
      feed(out.w,  in.w,   wnoise)
      feed(in.b,   out.b,  bnoise)
      feed(in.r,   out.r,  rnoise)
```

这一堆主要就是将node in和out的信号和参数分开,然后为w,r,b通道加噪声,最后将这些噪声通过feed传到总线,其实这个模块就是去延迟vaild和ready,在延迟期间bits是noise,在sink为vaild期间就是source的bit

# rocket ICache

一个典型的rocket chip结构

![1731144531780](image/diplomacy/1731144531780.png)

```
  val (tl_out, edge_out) = outer.masterNode.out(0)
```

在看rocket代码中有一个这个语句，masternode的out方法返回了两个变量，一个bundle，另一个是边的参数，这里是outward edge参数

深入挖掘out

```
  def out: Seq[(BO, EO)] = {
    require(
      instantiated,
      s"$name.out should not be called until after instantiation of its parent LazyModule.module has begun"
    )
    bundleOut.zip(edgesOut)
  }
```

发现在diplomacy库中的MixedNode（所有节点都继承了这个类）定义了这个out方法，其注释为将outward的边参数和端口gather起来，只能在LazyModuleImp中使用和访问

```
abstract class MixedNode[DI, UI, EI, BI <: Data, DO, UO, EO, BO <: Data](
  val inner: InwardNodeImp[DI, UI, EI, BI],
  val outer: OutwardNodeImp[DO, UO, EO, BO]
)

```

MixedNode是一个抽象类，只能被继承或作为基类，然后接下来讲解他的参数

DI:从上游传入的Downward-flowing parameters，对于一个InwardNode节点，他的参数由OutwardNode觉得，他可以多个源连接到一起，所以参数是Seq类型

UI:向上传的参数，一般为sink的参数，，对于InwardNode，参数由节点自身决定

EI:描述内边连接的参数，通常是根据协议对sink的一系列例化

BI:连接内边的Bundle type，他是这个sink接口的硬件接口代表真实硬件

DO:向外边传的参数，通常是source的参数对于一个OutwardNode，这个参数由自己决定

UO:外边传入的参数，通常是描述sink节点的参数，对于一个OutwardNode 这个由连接的inwardNode决定，由于这个可以被多个sinks连接，所以他是seq的

EO:描述外边的连接，通常是source节点的特殊的参数

BO:输出IO

接下来回归原题,可以看到有一个edge_out,这个变量有很多tilelink的方法,如检查是否是req等,是否含有data,但AXI的edge就没u,

# rocket ALU

首先ALU继承于下面的抽象类

```
abstract class AbstractALU(implicit p: Parameters) extends CoreModule()(p) {
  val io = IO(new Bundle {
    val dw = Input(UInt(SZ_DW.W))
    val fn = Input(UInt(SZ_ALU_FN.W))
    val in2 = Input(UInt(xLen.W))
    val in1 = Input(UInt(xLen.W))
    val out = Output(UInt(xLen.W))
    val adder_out = Output(UInt(xLen.W))
    val cmp_out = Output(Bool())
  })
}
```

首先dw的含义就是是32位还是64位

重点讲解一下移位操作

```
  // SLL, SRL, SRA
  val (shamt, shin_r) =
    if (xLen == 32) (io.in2(4,0), io.in1)
    else {
      require(xLen == 64)
      val shin_hi_32 = Fill(32, isSub(io.fn) && io.in1(31))
      val shin_hi = Mux(io.dw === DW_64, io.in1(63,32), shin_hi_32)
      val shamt = Cat(io.in2(5) & (io.dw === DW_64), io.in2(4,0))
      (shamt, Cat(shin_hi, io.in1(31,0)))
    }
  val shin = Mux(shiftReverse(io.fn), Reverse(shin_r), shin_r)
  val shout_r = (Cat(isSub(io.fn) & shin(xLen-1), shin).asSInt >> shamt)(xLen-1,0)
  val shout_l = Reverse(shout_r)
  val shout = Mux(io.fn === FN_SR || io.fn === FN_SRA || io.fn === FN_BEXT, shout_r, 0.U) |
              Mux(io.fn === FN_SL,                                          shout_l, 0.U)

```

shamt为移位的位数,我们假设是RV32,shin_r是被移位的数字,shin如果是左移,就将shin_r翻转,如果右移,则不变,shout_r检测是逻辑移位还是算数移位,如果是逻辑移位isSUB为false,也就是最高位符号位为0,,然后转换为有符号数,右移shamt,最后取出低32位,如果是verilog可以设置一个shift_mask(~(32'hffffffff)>>shamt),然后将移位前的符号位和这个mask&,最后或一下移位结果就是算数右移,

为什么逻辑左移可以转换为逻辑右移,将被移位的数字翻转后,最高位变为最低位,我们右移结果,翻转过来就是左移结果

然后就是结果输出模块s

```
  val out = MuxLookup(io.fn, shift_logic_cond)(Seq(
    FN_ADD -> io.adder_out,
    FN_SUB -> io.adder_out
  ) ++ (if (coreParams.useZbb) Seq(
    FN_UNARY -> unary,
    FN_MAX -> maxmin_out,
    FN_MIN -> maxmin_out,
    FN_MAXU -> maxmin_out,
    FN_MINU -> maxmin_out,
    FN_ROL -> rotout,
    FN_ROR -> rotout,
  ) else Nil))
```

这个表默认是shift_logic_cond,然后根据FN类型选择(这里还加入了Zbb扩展)

# rocket DecodeLogic

首先这个逻辑里面定义了两个方法

```
  // TODO This should be a method on BitPat
  private def hasDontCare(bp: BitPat): Boolean = bp.mask.bitCount != bp.width
  // Pads BitPats that are safe to pad (no don't cares), errors otherwise
  private def padBP(bp: BitPat, width: Int): BitPat = {
    if (bp.width == width) bp
    else {
      require(!hasDontCare(bp), s"Cannot pad '$bp' to '$width' bits because it has don't cares")
      val diff = width - bp.width
      require(diff > 0, s"Cannot pad '$bp' to '$width' because it is already '${bp.width}' bits wide!")
      BitPat(0.U(diff.W)) ## bp
    }
  }
```

其中hasDontCare是检查一个Bitpat是否有dontcare位,padBP是将一个bitpat格式的填充到width位

然后定义了好几个apply方法,这里只讲解rocketchip使用的

```
def apply(addr: UInt, default: Seq[BitPat], mappingIn: Iterable[(BitPat, Seq[BitPat])]): Seq[UInt] = {
    val nElts = default.size
    require(mappingIn.forall(_._2.size == nElts),
      s"All Seq[BitPat] must be of the same length, got $nElts vs. ${mappingIn.find(_._2.size != nElts).get}"
    )

    val elementsGrouped = mappingIn.map(_._2).transpose
    val elementWidths = elementsGrouped.zip(default).map { case (elts, default) =>
      (default :: elts.toList).map(_.getWidth).max
    }
    val resultWidth = elementWidths.sum

    val elementIndices = elementWidths.scan(resultWidth - 1) { case (l, r) => l - r }

    // All BitPats that correspond to a given element in the result must have the same width in the
    // chisel3 decoder. We will zero pad any BitPats that are too small so long as they dont have
    // any don't cares. If there are don't cares, it is an error and the user needs to pad the
    // BitPat themselves
    val defaultsPadded = default.zip(elementWidths).map { case (bp, w) => padBP(bp, w) }
    val mappingInPadded = mappingIn.map { case (in, elts) =>
      in -> elts.zip(elementWidths).map { case (bp, w) => padBP(bp, w) }
    }
    val decoded = apply(addr, defaultsPadded.reduce(_ ## _), mappingInPadded.map { case (in, out) => (in, out.reduce(_ ## _)) })

    elementIndices.zip(elementIndices.tail).map { case (msb, lsb) => decoded(msb, lsb + 1) }.toList
  }
```

可以看到他接受三个参数,返回一个seq,addr是要解码的数据,default是解码list的默认格式,rocket的格式为

![1731152261040](image/diplomacy/1731152261040.png)

mappin就是传入的decode表

![1731152309795](image/diplomacy/1731152309795.png)

类似于这种

首先我们nElts就是得出这个列表的元素个数,然后一个assert来确保传入的map和default的元素个数一致,然后elementsGrouped将List的各个控制信号分开,这里使用了map(遍历每个元素)和transpose(将元素转置,这样第一个seq就是所有表的val,以此类推),

然后得出每个bitpat的大小,这个elementWidths也就是将default的元素和elementsGrouped配对,然后将default的元素附加到elementsGrouped上,最后算出每个bitpat的大小

resultWidth就是所有bitpat的大小,然后elementIndices就是每个bitpat大小的索引,也就是假如每个bitpat大小为[4,3,2],这个得出的就是[8,4,1,-1]

然后最后哪一行代码就是将上面这个数组转换为(8,5),(4,2),(1,0)在,这样通过decode生成bool信号,然后将这些信号生成list

总体来说,这个模块使用了很多scala的高阶函数:

map:将给定函数作用于每个元素

transpose:将list转置

scan:扫描元组的每个值,并将其进行之后的函数操作,有累积性,比如这里就是给定初值,然后减去其他元素得到新的数组

zip:将两个元素组成一个元组

reduce:将元组的每个元素做相应操作,具有累积性

最后DecodeLogic实现的就是将输入的addr的每部分解码,然后得到解码的信号

# PMA（Physical Memory Attribute）

PMA是一个SOC系统的固有属性,所以直接将其设为硬件实现,PMA是软件可读的,

在平台支持pma的动态重新配置的地方，将提供一个接口，通过将请求传递给能够正确重新配置平台的机器模式驱动程序来设置属性。

例如，在某些内存区域上切换可缓存性属性可能涉及特定于平台的操作，例如缓存刷新，这些操作仅对机器模式可用。

## 3.6.1. 主内存、I/O和空闲区域

给定内存地址范围最重要的特征是它是否符合规则内存，或I/O设备，或为空闲。常规的主存需要有许多属性，如下所述，而I/O设备可以有更广泛的属性范围。不适合常规主存的内存区域，例如设备刮擦板ram，被归类为I/O区域。空区域也被归类为I/O区域，但具有指定不支持访问的属性。

## 3.6.2. Supported Access Type PMAs

访问类型指定支持哪些访问宽度（从8位字节到长多字突发），以及每个访问宽度是否支持不对齐的访问。

> 注:虽然在RISC-V hart上运行的软件不能直接生成内存突发，但软件可能必须对DMA引擎进行编程以访问I/O设备，因此可能需要知道支持哪种访问大小。

主存区域始终支持连接设备所需的所有访问宽度的读写，并且可以指定是否支持读指令。

> 注:有些平台可能要求所有主存都支持指令读取。其他平台可能会禁止从某些主内存区域获取指令。

在某些情况下，访问主存的处理器或设备的设计可能支持其他宽度，但必须能够与主存支持的类型一起工作。

I/O区域可以指定支持哪些数据宽度的读、写或执行访问组合。

对于具有基于页面的虚拟内存的系统，I/O和内存区域可以指定支持哪些硬件页表读和硬件页表写的组合。

> 注:类unix操作系统通常要求所有可缓存的主内存都支持PTW。

## 3.6.3. Atomicity PMAs

原子性pma描述在此地址区域中支持哪些原子指令。对原子指令的支持分为两类：LR/SC和AMOs。有些平台可能要求所有可缓存的主存支持附加处理器所需的所有原子操作。

在AMOs中，有四个级别的支持：AMONone、amosswap、AMOLogical和AMOArithmetic。

AMONone表示不支持AMO操作。AMOSwap表示该地址范围内只支持AMOSwap指令。AMOLogical表示支持交换指令加上所有逻辑AMOs （amoand、amoor、amoxor）。“AMOArithmetic”表示支持所有的RISC-V AMOs。对于每个级别的支持，如果底层内存区域支持该宽度的读写，则支持给定宽度的自然对齐的AMOs。主存和I/O区域可能只支持处理器支持的原子操作的一个子集，或者不支持处理器支持的原子操作。

![1731160758247](image/diplomacy/1731160758247.png)

对于LR/SC，有三个级别的支持表示可保留性和可能性属性的组合：RsrvNone、RsrvNonEventual和RsrvEventual。RsrvNone不支持LR/SC操作（位置不可预留）。RsrvNonEventual表示支持这些操作（位置是可保留的），但没有非特权ISA规范中描述的最终成功保证。RsrvEventual表示支持这些操作，并提供最终成功的保证。

> 注:我们建议在可能的情况下为主内存区域提供RsrvEventual支持。
>
> 大多数I/O区域将不支持LR/SC访问，因为它们最方便地构建在缓存一致性方案之上，但有些区域可能支持RsrvNonEventual或RsrvEventual。
>
> 当LR/SC用于标记为RsrvNonEventual的内存位置时，软件应该提供在检测到缺乏进度时使用的替代回退机制。

## 3.6.4. Misaligned Atomicity Granule PMA

Misaligned原子性粒子PMA为失调原子性粒子提供了约束支持。这个PMA（如果存在）指定了不对齐原子颗粒的大小，即自然对齐的2次幂字节数。该PMA的特定支持值由MAGNN表示，例如，MAG16表示不对齐的原子性颗粒至少为16字节。

不对齐的原子性颗粒PMA仅适用于基本isa中定义的AMOs、load和store，以及F、D和Q扩展中定义的不超过MXLEN位的load和store。对于该集中的一条指令，如果所有被访问的字节都位于同一个未对齐的原子颗粒中，则该指令不会因为地址对齐而引发异常，并且该指令将仅出于rvwmo的目的而引发一个内存操作。，它将自动执行。

如果一个未对齐的AMO访问的区域没有指定未对齐的原子性颗粒PMA，或者不是所有访问的字节都位于同一个未对齐的原子性颗粒内，则会引发异常。

对于访问这样一个区域的常规加载和存储，或者并非所有访问的字节都位于同一原子性颗粒内，则会引发异常，或者继续访问，但不保证是原子性的。对于一些不对齐的访问，实现可能会引发访问错误异常，而不是地址不对齐异常，这表明trap处理程序不应该模拟该指令。

> LR/SC指令不受此PMA的影响，因此当不对齐时总是引发异常。向量内存访问也不受影响，因此即使包含在未对齐的原子性颗粒中，也可能以非原子方式执行。隐式访问类似

## 3.6.5. Memory-Ordering PMAs

为了按照FENCE指令和原子指令排序位进行排序，地址空间的区域被分类为主存或I/O。

一个hart对主存区域的访问不仅可以被其他hart观察到，还可以被其他能够在主存系统中发起请求的设备（例如，DMA引擎）观察到。

coherence主存区域总是具有RVWMO或RVTSO内存模型。

非coherence的主存区域有一个实现定义的内存模型。

一个hart对一个I/O区域的访问不仅可以被其他hart和总线控制设备观察到，而且可以被目标I/O设备观察到，并且I/O区域可以以宽松或强顺序访问。其他hart和总线主控设备通常以类似于RVWMO内存区域访问顺序的方式来观察对具有宽松顺序的I/O区域的访问，如本规范第1卷a .4.2节所讨论的那样。相比之下，对具有强顺序的I/O区域的访问通常由其他hart和总线控制设备按照程序顺序观察。

每个强有序I/O区域指定一个编号的排序通道，这是一种在不同I/O区域之间提供排序保证的机制。通道0仅用于表示点对点强排序，其中只有hart对单个关联I/O区域的访问是强排序的。

通道1用于跨所有I/O区域提供全局强排序。hart对与通道1相关联的任何I/O区域的任何访问只能被所有其他hart和I/O设备观察到以程序顺序发生，包括相对于hart对宽松I/O区域或具有不同通道号的强顺序I/O区域的访问。换句话说，对通道1中的区域的任何访问都相当于在该指令之前和之后执行一个栅栏io，io指令。

其他更大的通道号为通过该通道号跨具有相同通道号的任何区域的访问提供程序排序。

系统可能支持在每个内存区域上动态配置排序属性。

强排序可用于改进与遗留设备驱动程序代码的兼容性，或者在已知实现不会重新排序访问时，与插入显式排序指令相比，可以提高性能。

本地强排序（通道0）是强排序的默认形式，因为如果hart和I/O设备之间只有一条有序通信路径，则通常可以直接提供它。

通常，如果不同的强排序I/O区域共享相同的互连路径并且路径不重新排序请求，则它们可以共享相同的排序通道，而无需额外的排序硬件

## 3.6.6. Coherence and Cacheability PMAs

内存区域的可缓存性不应该影响该区域的软件视图，除非在其他pma中反映出差异，例如主存与I/O分类、内存排序、支持的访问和原子操作以及一致性。出于这个原因，我们将**可缓存性**视为仅由机器模式软件管理的平台级设置。

如果平台支持内存区域的可配置缓存设置，则特定于平台的机器模式例程将在必要时更改设置并刷新缓存，因此系统仅在可缓存设置之间的转换期间不一致。较低的特权级别不应该看到这个临时状态

一致性很容易提供一个共享内存区域，它不被任何代理缓存。这样一个区域的PMA将简单地表示它不应该缓存在私有或共享缓存中。

对于只读区域，一致性也很简单，可以由多个代理安全地缓存，而不需要缓存一致性方案。该区域的PMA将表明它可以被缓存，但不支持写操作。

一些读写区域可能只由单个代理访问，在这种情况下，它们可以由该代理私下缓存，而不需要一致性方案。这些区域的PMA将表明它们可以被缓存。数据也可以缓存在共享缓存中，因为其他代理不应该访问该区域。

如果代理可以缓存其他代理可以访问的读写区域，无论是缓存还是非缓存，都需要缓存一致性方案来避免使用过时的值。

在缺乏硬件缓存一致性的区域（硬件非一致性区域），缓存一致性可以完全在软件中实现，但众所周知，软件一致性方案难以正确实现，并且由于需要保守的软件定向缓存刷新，通常会对性能产生严重影响。硬件缓存一致性方案需要更复杂的硬件，并且由于缓存一致性探测可能会影响性能，但对软件来说是不可见的。

对于每个硬件缓存相干区域，PMA将指示该区域是相干的，如果系统有多个相干控制器，则指示使用哪个硬件相干控制器。对于某些系统，一致性控制器可能是一个外部共享缓存，它本身可以分层访问其他外部缓存一致性控制器。

平台中的大多数内存区域将与软件一致，因为它们将被固定为非缓存、只读、硬件缓存一致或仅由一个代理访问。

如果PMA表示不可缓存，那么对该区域的访问必须由内存本身满足，而不是由任何缓存满足。

对于具有可缓存性控制机制的实现，可能会出现程序无法访问当前驻留在缓存中的内存位置的情况。在这种情况下，必须忽略缓存的副本。防止这种约束是必要的去阻止高特权模式的推测缓存重新填充不会影响较少特权模式的不可缓存访问行为。

## 3.6.7. Idempotency PMAs

幂等pma描述对地址区域的读写是否幂等。假定主存储器区域是幂等的。对于I/O区域，读和写的幂等性可以分别指定（例如，读是幂等的，而写不是）。如果访问是非幂等的，即对任何读或写访问都有潜在的副作用，则必须避免推测性访问或冗余访问。

为了定义幂等pma，冗余访问对观察到的内存顺序的改变不被认为是副作用。

虽然硬件应始终设计为避免对标记为非幂等的内存区域进行投机或冗余访问，但也有必要确保软件或编译器优化不会生成对非幂等内存区域的虚假访问。

非幂等区域可能不支持不对齐访问。对这些区域的不对齐访问应该引发访问错误异常，而不是地址不对齐异常，这表明软件不应该使用多个较小的访问来模拟不对齐的访问，这可能会导致意想不到的副作用。

对于非幂等区域，隐式读写不能提前或推测地执行，除了以下例外情况。当执行非推测式隐式读操作时，允许实现在包含非推测式隐式读操作地址的自然对齐的2次幂区域内额外读取任何字节。此外，当执行非推测指令获取时，允许实现额外读取下一个自然对齐的相同大小的2次幂区域内的任何字节（该区域的地址取2XLEN模）。这些额外读取的结果可用于满足后续的早期或推测式隐式读取。这些自然对齐的2次幂区域的大小是由实现定义的，但是，对于具有基于页面的虚拟内存的系统，不能超过所支持的最小页面大小

译者注:这里描述的应该跟预取有关,允许预取特定字节的数据,地址得2的幂次对齐

# 3.7. Physical Memory Protection

为了支持安全处理和包含错误，需要限制运行在硬件上的软件可访问的物理地址。一个可选的物理内存保护（PMP）单元提供每台机器模式控制寄存器，允许为每个物理内存区域指定物理内存访问特权（读、写、执行）。PMP值与第3.6节中描述的PMA检查并行检查。

PMP访问控制设置的粒度是特定于平台的，但是标准PMP编码支持小至4字节的区域。某些区域的特权可以是硬连接的，例如，某些区域可能只在机器模式下可见，而在低特权层中不可见。

PMP检查区域:PMP检查应用于有效特权模式为访问S和U模式的指令读取和数据访问,

当mstatus中的MPRV位被设置，并且mstatus中的MPP字段包含S或u时，m模式下的数据访问也被应用于虚拟地址转换的页表访问，其有效特权模式为S。可选地，PMP检查还可以应用于m模式访问，在这种情况下，PMP寄存器本身被锁定，因此即使m模式软件也不能更改它们，直到hart被重置。实际上，PMP可以授予S和U模式权限（默认情况下没有），还可以从Mmode撤销权限（默认情况下具有完全权限）。

PMP违规总是被捕捉到精确异常

## 3.7.1. Physical Memory Protection CSRs

PMP表项由一个8位配置寄存器和一个mxlen位地址寄存器描述。

一些PMP设置还使用与前一个PMP项相关联的地址寄存器。最多支持64个PMP表项。实现可以实现0、16或64个PMP表项；编号最少的PMP表项必须首先实现。所有PMP CSR字段都是WARL，可以是只读零。PMP csr仅在m模式下可访问。

PMP配置寄存器被密集地打包到csr中，以最小化上下文切换时间。对于RV32, 16个csr， pmpcfg0-pmpcfg15，为64个PMP条目保留配置pmp0cfg-pmp63cfg，如图30所示。对于RV64, 8个偶数csr pmpcfg0、pmpcfg2、…、pmpcfg14保存64个PMP条目的配置，如图31所示。对于RV64，奇数配置寄存器pmpcfg1， pmpcfg3，…，pmpcfg15是非法的。

PMP地址寄存器是命名为pmpaddr0-pmpaddr63的csr。每个PMP地址寄存器为RV32编码34位物理地址的第33-2位，如图32所示。对于RV64，每个PMP地址寄存器编码56位物理地址的第55-2位，如图33所示。并非所有的物理地址位都可以实现，因此pmpaddr寄存器是WARL

> 注:章节10.3中描述的基于Sv32页面的虚拟内存方案支持RV32的34位物理地址，因此PMP方案必须支持RV32的大于XLEN的地址。第10.4节和10.5节中描述的Sv39和Sv48基于页面的虚拟内存方案支持56位物理地址空间，因此RV64 PMP地址寄存器施加了相同的限制。

![1731164360120](image/diplomacy/1731164360120.png)

![1731164379398](image/diplomacy/1731164379398.png)

图34显示了PMP配置寄存器的布局。设置R、W和X位时，分别表示PMP项允许读、写和指令执行。当这些位中的一个被清除时，对应的访问类型被拒绝。R、W和X字段形成一个集合的WARL字段，其中保留R=0和W=1的组合。剩下的两个字段A和L将在下面的部分中描述。

尝试从不具有执行权限的PMP区域获取指令将引发指令访问错误异常。试图执行在没有读权限的情况下访问PMP区域内物理地址的加载或负载保留指令会引发加载访问错误异常。试图执行在没有写权限的情况下访问PMP区域内物理地址的存储、存储条件或AMO指令，将引发存储访问错误异常。

### 3.7.1.1. Address Matching

PMP表项配置寄存器中的A字段编码了相关联的PMP地址寄存器的地址匹配模式。这个字段的编码如表18所示。当A=0时，该PMP表项被禁用并且不匹配任何地址。支持另外两种地址匹配模式：自然对齐的2次幂区域（NAPOT），包括自然对齐的四字节区域（NA4）的特殊情况；以及任意范围的上边界（TOR）。这些模式支持四字节粒度

![1731164814689](image/diplomacy/1731164814689.png)

NAPOT范围使用相关地址寄存器的低阶位来编码范围的大小，如表19所示。检测连续1的数目

* 若 `pmpaddr`值为 `yyyy...yy01`，即连续1的个数为1，则该PMP entry所控制的地址空间为从 `yyyy...yy00`开始的16个字节

![1731164947670](image/diplomacy/1731164947670.png)

如果选择TOR，则关联的地址寄存器为地址范围的顶部，前面的PMP地址寄存器为地址范围的底部。如果PMP表项i的A字段设置为TOR，则该表项匹配任何地址y，使pmpaddri-1≤y<pmpaddri（与pmpcfgi-1的值无关）。如果PMP条目0的A字段设置为TOR，则使用0作为下界，因此它匹配任何地址y<pmpaddr0。

如果pmpaddri-1≥pmpaddri和pmpcfgi。A=TOR，则PMP表项i不匹配任何地址。

软件可以通过将0写入pmp0cfg，然后将所有1写入pmpaddr0，然后回读pmpaddr0来确定PMP粒度。如果G是最低有效位集的索引，则PMP粒度为2G+2字节。(NAPOT)

> 注意:这里的G是0在paddr的位置

如果当前的XLEN大于MXLEN，为了地址匹配的目的，PMP地址寄存器从MXLEN到XLEN位进行零扩展。

### 3.7.1.2. Locking and Privilege Mode

L位表示PMP表项被锁定。锁定的PMP表项一直处于锁定状态，直到hart被重置。如果PMP表项i被锁定，对pmppfg和pmpaddri的写入将被忽略。此外，如果PMP表项i被锁定并且PMP icfgA被设置为TOR，对pmpadri -1的写入将被忽略。

设置L位锁定PMP表项，即使A字段被设置为OFF。

除了锁定PMP表项外，L位表示是否对m模式访问强制R/W/X权限。当设置L位时，这些权限对所有特权模式强制执行。

当L位清除时，任何匹配PMP表项的m模式访问都将成功；R/W/X权限只适用于S模式和U模式。

### 3.7.1.3. Priority and Matching Logic

PMP表项的优先级是静态的。与访问的任何字节匹配的编号最低的PMP表项决定该访问是成功还是失败。匹配的PMP表项必须匹配访问的所有字节，否则访问失败，无论L、R、W和X位如何。例如，如果将PMP表项配置为匹配4字节范围0xC-0xF，那么假设PMP表项是匹配这些地址的最高优先级表项，那么对0x8-0xF范围的8字节访问将失败。

如果一个PMP表项匹配一次访问的所有字节，那么L、R、W和X位决定这次访问是成功还是失败。如果L位为空，且访问的特权模式为M，则表示访问成功。否则，如果设置了L位或访问的特权模式为S或U，则只有设置了与访问类型对应的R、W或X位，才能访问成功。

如果没有匹配m模式访问的PMP表项，则访问成功。如果没有匹配s模式或u模式访问的PMP表项，但至少实现了一个PMP表项，则访问失败。如果至少实现了一个PMP表项，但是所有PMP表项的A字段都被设置为OFF，那么所有s模式和u模式内存访问都将失败。

访问失败会产生指令、加载或存储访问错误异常。请注意，一条指令可能产生多个访问，这些访问可能不是相互原子的。如果一条指令产生的至少一次访问失败，则会产生访问错误异常，尽管该指令产生的其他访问可能会成功，但会产生明显的副作用。值得注意的是，引用虚拟内存的指令被分解为多个访问。

在某些实现中，不对齐的加载、存储和指令提取也可以分解为多个访问，其中一些访问可能在访问错误异常发生之前成功。特别是，通过PMP检查的未对齐存储的一部分可能变得可见，即使另一部分未通过PMP检查。即使存储地址是自然对齐的，同样的行为也可能出现在大于XLEN位的存储中（例如，RV32D中的FSD指令）。

## 3.7.2. Physical Memory Protection and Paging

物理内存保护机制被设计成与第10章中描述的基于页面的虚拟内存系统相结合。当启用分页时，访问虚拟内存的指令可能导致多次物理内存访问，包括对页表的隐式引用。**PMP检查应用于所有这些访问**。隐式可分页访问的有效特权模式是S。

使用虚拟内存的实现被允许在显式内存访问要求之前推测性地执行地址转换，并被允许将它们缓存在地址转换缓存结构中——包括可能缓存在Bare转换模式和m模式中使用的从有效地址到物理地址的身份映射。结果物理地址的PMP设置可以在地址转换和显式内存访问之间的任何点进行检查（并可能进行缓存）。因此，当修改PMP设置时，m模式软件必须将PMP设置与虚拟内存系统以及任何PMP或地址转换缓存同步。这是通过执行一个SFENCE来完成的。在PMP csr写入后，rs1=x0和rs2=x0的VMA指令。实现虚拟化管理程序扩展时的其他同步要求，请参见18.5.3节。

如果没有实现基于页面的虚拟内存，内存访问将同步检查PMP设置，因此没有SFENCE.VMA是必需的。

# BOOM IFU

![1731995196160](image/diplomacy&boom/1731995196160.png)

前端将从ICache读出的数据写入fetch buf

## BOOM Front end

前端为5个阶段,f0产生pc,f1进行TLB转换,F2读出数据送入IMem,F3对指令预解码,检查分支预测,(f1,f2,f3每个阶段都可以产生重定向,),然后将指令送入Fetch buffer,将分支预测信息送入FTQ

### F0

这个阶段选择pc,并且向icache和bpd发送请求

```
  when (RegNext(reset.asBool) && !reset.asBool) {
    s0_valid   := true.B
    s0_vpc     := io_reset_vector
    s0_ghist   := (0.U).asTypeOf(new GlobalHistory)
    s0_tsrc    := BSRC_C
  }

  icache.io.req.valid     := s0_valid
  icache.io.req.bits.addr := s0_vpc

  bpd.io.f0_req.valid      := s0_valid
  bpd.io.f0_req.bits.pc    := s0_vpc
  bpd.io.f0_req.bits.ghist := s0_ghist
```

s0的信号来自于其他阶段,这个是f1阶段的信号,如果f1有效,并且没有tlb_miss,就把f1的预测结果送入f0,然后标记结果来自BSRC_1,也就是ubtb,然后把f1的ghist送入f0,

```
  when (s1_valid && !s1_tlb_miss) {
    // Stop fetching on fault
    s0_valid     := !(s1_tlb_resp.ae.inst || s1_tlb_resp.pf.inst)
    s0_tsrc      := BSRC_1
    s0_vpc       := f1_predicted_target
    s0_ghist     := f1_predicted_ghist
    s0_is_replay := false.B
  }
```

f2阶段送入的信号分以下情况

* 如果s2阶段有效,并且icache无回应,或者icache有回应但f3阶段没有准备好接受,此时需要进行重定向,重新发送指令请求,然后清除f1阶段,
* 如果s2阶段有效且f3准备好接受:1. 如果f2阶段预测的和f1的pc一样,就更新f2阶段的ghist,表示预测正确,2.如果f2的预测结果和f1不一样,或者f1本身就是无效的,就清除f1阶段,并且将pc重定向为预测器的pc,将s0的预测结果设置为BSRC_2

```
  when ((s2_valid && !icache.io.resp.valid) ||
        (s2_valid && icache.io.resp.valid && !f3_ready)) {
    s0_valid := (!s2_tlb_resp.ae.inst && !s2_tlb_resp.pf.inst) || s2_is_replay || s2_tlb_miss
    s0_vpc   := s2_vpc
    s0_is_replay := s2_valid && icache.io.resp.valid
    // When this is not a replay (it queried the BPDs, we should use f3 resp in the replaying s1)
    s0_s1_use_f3_bpd_resp := !s2_is_replay
    s0_ghist := s2_ghist
    s0_tsrc  := s2_tsrc
    f1_clear := true.B
  } .elsewhen (s2_valid && f3_ready) {
    when (s1_valid && s1_vpc === f2_predicted_target && !f2_correct_f1_ghist) {
      // We trust our prediction of what the global history for the next branch should be
      s2_ghist := f2_predicted_ghist
    }
    when ((s1_valid && (s1_vpc =/= f2_predicted_target || f2_correct_f1_ghist)) || !s1_valid) {
      f1_clear := true.B

      s0_valid     := !((s2_tlb_resp.ae.inst || s2_tlb_resp.pf.inst) && !s2_is_replay)
      s0_vpc       := f2_predicted_target
      s0_is_replay := false.B
      s0_ghist     := f2_predicted_ghist
      s2_fsrc      := BSRC_2
      s0_tsrc      := BSRC_2
    }
  }
  s0_replay_bpd_resp := f2_bpd_resp
  s0_replay_resp := s2_tlb_resp
  s0_replay_ppc  := s2_ppc
```

如果f3阶段的信号有效,f3重定向有以下情况

* 如果f2阶段信号有效,但f2的pc不为f3的预测pc,或者f2的ghist和f3不一样
* 如果f2阶段无效,f1阶段有效,但f1的pc不为f3的预测pc,或者f1的ghist和f3不一样
* 如果f1,f2均无效

此时,需要清除f2和f1阶段,然后将s0的pc设置为f3预测的pc

```
.elsewhen (( s2_valid &&  (s2_vpc =/= f3_predicted_target || f3_correct_f2_ghist)) ||
          (!s2_valid &&  s1_valid && (s1_vpc =/= f3_predicted_target || f3_correct_f1_ghist)) ||
          (!s2_valid && !s1_valid)) {
      f2_clear := true.B
      f1_clear := true.B

      s0_valid     := !(f3_fetch_bundle.xcpt_pf_if || f3_fetch_bundle.xcpt_ae_if)
      s0_vpc       := f3_predicted_target
      s0_is_replay := false.B
      s0_ghist     := f3_predicted_ghist
      s0_tsrc      := BSRC_3

      f3_fetch_bundle.fsrc := BSRC_3
    }
```

最后就是后端传来信号

* 如果执行了sfence,需要冲刷整个前端,将指令设置为sfence的pc
* 如果后端发来重定向,冲刷整个前端,将pc设置为重定向pc

```
  when (io.cpu.sfence.valid) {
    fb.io.clear := true.B
    f4_clear    := true.B
    f3_clear    := true.B
    f2_clear    := true.B
    f1_clear    := true.B

    s0_valid     := false.B
    s0_vpc       := io.cpu.sfence.bits.addr
    s0_is_replay := false.B
    s0_is_sfence := true.B

  }.elsewhen (io.cpu.redirect_flush) {
    fb.io.clear := true.B
    f4_clear    := true.B
    f3_clear    := true.B
    f2_clear    := true.B
    f1_clear    := true.B

    f3_prev_is_half := false.B

    s0_valid     := io.cpu.redirect_val
    s0_vpc       := io.cpu.redirect_pc
    s0_ghist     := io.cpu.redirect_ghist
    s0_tsrc      := BSRC_C
    s0_is_replay := false.B

    ftq.io.redirect.valid := io.cpu.redirect_val
    ftq.io.redirect.bits  := io.cpu.redirect_ftq_idx
  }
```

**总结**

pc重定向

1、当执行SFENCE.VMA指令时，代表软件可能已经修改了页表，因此此时的TLB里的内容可能是错误的，那么此时正在流水线中执行的指令也有可能是错误的，因此需要刷新TLB和冲刷流水线，也需要重新进行地址翻译和取指，所以此时需要重定向PC值。

2、当执行级发现分支预测失败、后续流水线发生异常或者发生Memory Ordering Failure时（Memory Ordering Failure的相关介绍见参考资料[1])，需要冲刷流水线，将处理器恢复到错误执行前的状态，指令也需要重新进行取指，所以此时也需要重定向PC值。

3、当发生以下三种情况时，需要将PC重定向为F3阶段分支预测器预测的目标跳转地址：

F2阶段的指令有效且F3阶段的分支预测结果与此时处于F2阶段的指令的PC值不相同；

F2阶段的指令无效且F3阶段的分支预测结果与此时处于F1阶段的指令的PC值不相同；

F2阶段和F1阶段的指令均无效。

4、当Icache的响应无效或者F3阶段传来的握手信号没有准备就绪时，需要将PC值重定向为此时处于F2阶段的指令的PC值。

5、当F1阶段的指令有效且F2阶段的分支预测结果与此时处于F1阶段的指令的PC值不相同或者F1阶段的指令无效时，需要将PC重定向为F2阶段分支预测器预测的目标跳转地址。

6、当TLB没有发生miss且F1阶段的分支预测器预测结果为跳转时，需要将PC重定向为预测的目标跳转地址。

### F1

F1阶段进行tlb转换,并且得出ubtb结果,如果tlb miss需要终止icache访存,这个周期ubtb给出预测结果,根据结果对前端重定向

#### TLB访问逻辑

如下面代码,s1_resp的结果来自两部分,如果s1有replay信号,那么结果就是replay的数据(只有f2才会发出replay表示指令准备好了但不能接受),否则就是tlb得出的数据

> 个人感觉这里是降低功耗的一个小方法,如果f2replay,那么他的物理地址一定计算完了,我们就可以减少一次tlb访问

```
tlb.io.req.valid      := (s1_valid && !s1_is_replay && !f1_clear) || s1_is_sfence
...
  val s1_tlb_miss = !s1_is_replay && tlb.io.resp.miss
  val s1_tlb_resp = Mux(s1_is_replay, RegNext(s0_replay_resp), tlb.io.resp)
  val s1_ppc  = Mux(s1_is_replay, RegNext(s0_replay_ppc), tlb.io.resp.paddr)
  val s1_bpd_resp = bpd.io.resp.f1

  icache.io.s1_paddr := s1_ppc
  icache.io.s1_kill  := tlb.io.resp.miss || f1_clear
```

#### 分支信息处理逻辑

f1阶段得出的分支预测结果可能有多个,我们取最旧的一个作为分支目标地址,然后更新ghist(GHR)

> 如何选出最旧的分支呢？这里的做法是首先通过fetchMask得到一个指令包的有效指令位置，然后通过通过查询每个指令是否是分支指令并且taken，生成一个新的f1_redirects，然后通过优先编码器得到最旧指令的idx，之后从bpd的resp取出这个idx对应预测结果，如果确实有分支进行预测，就置target为预测的target，否则为pc+4（or 2）

```
  val f1_mask = fetchMask(s1_vpc)
  val f1_redirects = (0 until fetchWidth) map { i =>
    s1_valid && f1_mask(i) && s1_bpd_resp.preds(i).predicted_pc.valid &&
    (s1_bpd_resp.preds(i).is_jal ||
      (s1_bpd_resp.preds(i).is_br && s1_bpd_resp.preds(i).taken))
  }
  val f1_redirect_idx = PriorityEncoder(f1_redirects)
  val f1_do_redirect = f1_redirects.reduce(_||_) && useBPD.B
  val f1_targs = s1_bpd_resp.preds.map(_.predicted_pc.bits)
  val f1_predicted_target = Mux(f1_do_redirect,
                                f1_targs(f1_redirect_idx),
                                nextFetch(s1_vpc))

  val f1_predicted_ghist = s1_ghist.update(
    s1_bpd_resp.preds.map(p => p.is_br && p.predicted_pc.valid).asUInt & f1_mask,
    s1_bpd_resp.preds(f1_redirect_idx).taken && f1_do_redirect,
    s1_bpd_resp.preds(f1_redirect_idx).is_br,
    f1_redirect_idx,
    f1_do_redirect,
    s1_vpc,
    false.B,
    false.B)
```

#### 详解mask

取指令通过mask来屏蔽无效指令，如下面代码，我们只讲解bank=2的情况，首先算出shamt位移量，然后通过是否在同一个set算出end_mask,最后进行编码

举例：假设fetchWidth=8，coreInstBytes=2，block=16bytes，numChunks=2 banks=2

如果地址为0011 1100，

idx=110

shamt=10

那么这个地址显然需要跨两行，mayNotBeDualBanked显然为1，

故end_mask = 0000 1111

故最终结果为0000 1100，也就是他会屏蔽跨行的指令

如果地址为0011 0100，这个没有跨行，所以最终结果为

1111 1100

也就是说，mask是对取出的指令做一个有效编码

```
  def isLastBankInBlock(addr: UInt) = {
    (nBanks == 2).B && addr(blockOffBits-1, log2Ceil(bankBytes)) === (numChunks-1).U
  }
  def mayNotBeDualBanked(addr: UInt) = {
    require(nBanks == 2)
    isLastBankInBlock(addr)
  }
  def fetchMask(addr: UInt) = {
    val idx = addr.extract(log2Ceil(fetchWidth)+log2Ceil(coreInstBytes)-1, log2Ceil(coreInstBytes))
    if (nBanks == 1) {
      ((1 << fetchWidth)-1).U << idx
    } else {
      val shamt = idx.extract(log2Ceil(fetchWidth)-2, 0)
      val end_mask = Mux(mayNotBeDualBanked(addr), Fill(fetchWidth/2, 1.U), Fill(fetchWidth, 1.U))
      ((1 << fetchWidth)-1).U << shamt & end_mask
    }
  }
```

那么br_mask 就是在有效指令中筛选为BR的指令

#### GHist更新逻辑

以例子来进行讲解

Ghist的更新是采用了update方法,他的输入依次如下：

* branches: UInt,：这个就是上面讲解的br_mask,
* cfi_taken: Bool：指令是否taken，这个信号一般指的是最旧的指令是否taken，这个例子就是先得出f1_redirects(重定向指令的mask)，然后通过优先编码器得出最旧的指令然后得出是否要重定向信号f1_do_redirect，以及预测目标，所以这个信号就是最旧的分支是否taken，并且是否要重定向。
* cfi_is_br: Bool：这个信号得出了最旧的分支指令是否为br，（f1分支预测包含br jalr，jalr，但只有条件分支可以更改ghist）
* cfi_idx: UInt：得出最旧的分支指令的（这个可能包括jal或jalr，而且这个不是oh编码，只是简单的idx）
* cfi_valid: Bool：是否需要重定向
* addr: UInt：pc
* cfi_is_call: Bool
* cfi_is_ret: Bool

```
  val f1_mask = fetchMask(s1_vpc)
  val f1_redirects = (0 until fetchWidth) map { i =>
    s1_valid && f1_mask(i) && s1_bpd_resp.preds(i).predicted_pc.valid &&
    (s1_bpd_resp.preds(i).is_jal ||
      (s1_bpd_resp.preds(i).is_br && s1_bpd_resp.preds(i).taken))
  }
  val f1_redirect_idx = PriorityEncoder(f1_redirects)
  val f1_do_redirect = f1_redirects.reduce(_||_) && useBPD.B
  val f1_targs = s1_bpd_resp.preds.map(_.predicted_pc.bits)
  val f1_predicted_target = Mux(f1_do_redirect,
                                f1_targs(f1_redirect_idx),
                                nextFetch(s1_vpc))
  val f1_predicted_ghist = s1_ghist.update(
    s1_bpd_resp.preds.map(p => p.is_br && p.predicted_pc.valid).asUInt & f1_mask,
    s1_bpd_resp.preds(f1_redirect_idx).taken && f1_do_redirect,
    s1_bpd_resp.preds(f1_redirect_idx).is_br,
    f1_redirect_idx,
    f1_do_redirect,
    s1_vpc,
    false.B,
    false.B)
```

**not_taken_branches**：如果条件分支taken或者不是条件分支，这个就为0，否则就不为0，

然后进入update方法，update方法也是分了bank讨论，首先讨论bank为1的

new_history.old_history更新逻辑：

* 如果这个分支是条件分支并且taken：histories(0) <<1|1.U
* 如果是条件分支但没有taken：histories(0) <<1
* 如果不是条件分支：histories(0)

下面讨论bank为2的情况

他使用的始终histories(1)，也就是更新逻辑,

首先判断cfi指令在bank0或者整个packet是否跨行了（ignore_second_bank），然后得出第一个bank是否有条件分支未taken（first_bank_saw_not_taken）：

如果忽视bank1，根据new_history.new_saw_branch_not_taken ，new_history.new_saw_branch_taken更新old_hist

否则，new_saw_branch_not_taken：bank1是否有没taken的指令

new_saw_branch_taken：bank1是否有taken的指令并且cfi不在bank0

然后更新old_hist:

> ~~感觉这个更新逻辑有问题，ignore_second_bank有两个条件：如果cfi在bank0，otherwise的MUX的cfi_is_br && cfi_in_bank_0必然不会成立，如果mayNotBeDualBanked成立，那么cfi必然在bank1，该条件仍然不会成立，同样first_bank_saw_not_taken也不会成立，所以这个逻辑最后就是得到了histories（1），之前的逻辑都是冗余的(将多余代码去掉仍然可以运行程序)~~
>
> 没什么问题,如果想进入otherwise代码块:
>
> 1. bank0无分支或者分支预测没taken
> 2. 分支指令在bank1
>
> 但cfi_is_br && cfi_in_bank_0是无效的逻辑，进入when代码块必然不会进入otherwise，所以必然不会触发这个MUX条件（理解问题？）

举个例子来说明这两个条件什么意思:

例：假设fetchWidth=8，coreInstBytes=2，block=16bytes，numChunks=2 banks=2

如果地址为0011 1100，cfi_idx_oh为0000 1000

这个地址mayNotBeDualBanked为1，cfi_in_bank0为1,如果这个不是分支,或者没有taken,cfi_in_bank0为0

如果地址0011 0000,cfi_idx_oh为0001 0000

这个地址mayNotBeDualBanked为0，cfi_in_bank0为0,如果cfi_idx_oh,cfi_in_bank0就为1

> 这里ignore_second_bank的意思就是第二个分支没有分支或者分支无效,
>
> 假设第二个bank有分支,我们会忽视第一个bank的分支历史,只更新第二个bank

> In the two bank case every bank ignore the history added by the previous bank

```
  def histories(bank: Int) = {
    if (nBanks == 1) {
      old_history
    } else {
      require(nBanks == 2)
      if (bank == 0) {
        old_history
      } else {
        Mux(new_saw_branch_taken                            , old_history << 1 | 1.U,
        Mux(new_saw_branch_not_taken                        , old_history << 1,
                                                              old_history))
      }
    }
  }  
def update(branches: UInt, cfi_taken: Bool, cfi_is_br: Bool, cfi_idx: UInt,
    cfi_valid: Bool, addr: UInt,
    cfi_is_call: Bool, cfi_is_ret: Bool): GlobalHistory = {
    val cfi_idx_fixed = cfi_idx(log2Ceil(fetchWidth)-1,0)
    val cfi_idx_oh = UIntToOH(cfi_idx_fixed)
    val new_history = Wire(new GlobalHistory)

    val not_taken_branches = branches & Mux(cfi_valid,
                                            MaskLower(cfi_idx_oh) & ~Mux(cfi_is_br && cfi_taken, cfi_idx_oh, 0.U(fetchWidth.W)),
                                            ~(0.U(fetchWidth.W)))

    if (nBanks == 1) {
      // In the single bank case every bank sees the history including the previous bank
      new_history := DontCare
      new_history.current_saw_branch_not_taken := false.B
      val saw_not_taken_branch = not_taken_branches =/= 0.U || current_saw_branch_not_taken
      new_history.old_history := Mux(cfi_is_br && cfi_taken && cfi_valid   , histories(0) << 1 | 1.U,
                                 Mux(saw_not_taken_branch                  , histories(0) << 1,
                                                                             histories(0)))
    } else {
      // In the two bank case every bank ignore the history added by the previous bank
      val base = histories(1)
      val cfi_in_bank_0 = cfi_valid && cfi_taken && cfi_idx_fixed < bankWidth.U
      val ignore_second_bank = cfi_in_bank_0 || mayNotBeDualBanked(addr)

      val first_bank_saw_not_taken = not_taken_branches(bankWidth-1,0) =/= 0.U || current_saw_branch_not_taken
      new_history.current_saw_branch_not_taken := false.B
      when (ignore_second_bank) {
        new_history.old_history := histories(1)
        new_history.new_saw_branch_not_taken := first_bank_saw_not_taken
        new_history.new_saw_branch_taken     := cfi_is_br && cfi_in_bank_0
      } .otherwise {
        new_history.old_history := Mux(cfi_is_br && cfi_in_bank_0                             , histories(1) << 1 | 1.U,
                                   Mux(first_bank_saw_not_taken                               , histories(1) << 1,
                                                                                                histories(1)))

        new_history.new_saw_branch_not_taken := not_taken_branches(fetchWidth-1,bankWidth) =/= 0.U
        new_history.new_saw_branch_taken     := cfi_valid && cfi_taken && cfi_is_br && !cfi_in_bank_0

      }
    }
    new_history.ras_idx := Mux(cfi_valid && cfi_is_call, WrapInc(ras_idx, nRasEntries),
                           Mux(cfi_valid && cfi_is_ret , WrapDec(ras_idx, nRasEntries), ras_idx))
    new_history
  }
```

### F2

f2阶段获得cache数据，注意f2阶段可能收到无效的cache数据，或者收到了数据但f3接收不了，这时就要重定向，然后冲刷f1阶段，f2阶段也会得到预测结果，其处理和f1阶段类似

> (s1_vpc =/= f2_predicted_target || f2_correct_f1_ghist)，f2分支预测重定向需要前面两个条件，条件1的意思就是f2阶段预测的地址和之前这条指令与目前f1的pc不一样，条件2的意思预测方向不一样

### F3

f3阶段使用了IMem Response Queue和BTB Response Queue，两个队列项数均为1，其中IMem Response Queue在F2阶段入队，在F3阶段出队，主要传递Icache响应的指令、PC、全局历史等信息；而BTB Response Queue则设置成“flow”的形式（即输入可以在同一周期内“流”过队列输出），所以它的入队出队均在F3阶段完成，主要传递分支预测器的预测信息。

这个周期也会有来自bpd的预测信息（TAGE），同样会进行重定向，该阶段有一个快速译码单元用于检查分支预测，并且这个周期会检查RVC指令并进行相应处理

#### 有效指令截断处理

也就是32位的指令分布在两个指令包

> 小插曲：f3_prev_is_half的值来自bank_prev_is_half，而bank_prev_is_half是一个var，也就是可变变量,这里他在for循环内多次被赋值，实际上就是给f3_prev_is_half提供了多个赋值条件
>
> ```
> ...   
>  bank_prev_is_half = Mux(f3_bank_mask(b),
>       (!(bank_mask(bankWidth-2) && !isRVC(bank_insts(bankWidth-2))) && !isRVC(last_inst)),
>       bank_prev_is_half)
> ...
>   when (f3.io.deq.fire) {
>     f3_prev_is_half := bank_prev_is_half
>     f3_prev_half    := bank_prev_half
>     assert(f3_bpd_resp.io.deq.bits.pc === f3_fetch_bundle.pc)
>   }
> ```
>
> 下面是一个测试用例
>
> ![1732097110110](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Legion/Desktop/arch_note/image/diplomacy&boom/1732097110110.png)
>
> ![1732097127143](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Legion/Desktop/arch_note/image/diplomacy&boom/1732097127143.png)

首先先解析bank信号，bank_data可以看到就是每个bank的data，对于largeboom就是64位的数据（其中bankwidth为4，bank为2）

```
    val bank_data  = f3_data((b+1)*bankWidth*16-1, b*bankWidth*16)
    val bank_mask  = Wire(Vec(bankWidth, Bool()))
    val bank_insts = Wire(Vec(bankWidth, UInt(32.W)))
```

bank_mask和之前提到的mask类似,揭示了一个bank每条指令是否有效,

当f3的指令有效并且没有收到重定向信号,就对bank_mask赋值

```
  for (b <- 0 until nBanks) {
.....

    for (w <- 0 until bankWidth) {
      val i = (b * bankWidth) + w
      bank_mask(w) := f3.io.deq.valid && f3_imemresp.mask(i) && valid && !redirect_found
```

bank_inst主要逻辑在内层循环内

主要有4种情况:

1. 当w=0,也就是第一条指令,注意这条指令可能是不完整的32bit指令,如果这条指令是不完整,那么就将之前存的half指令拼接到这个不完整的指令,形成32bit(bank_data(15,0), f3_prev_half),注意如果此时b>0,也即是现在是bank1,那么之前的一半指令就是(bank_data(15,0), last_inst)拼接,如果这个指令是完整的指令,就直接为bank_data(31,0),valid一定为true
2. 当w=1,bank_inst就直接为bank_data(47,16),
3. 当w=bankWidth -1,注意这里可能会发生32bit的指令不完整的情况,bank_inst为16个0和bank_data(bankWidth*16-1,(bankWidth-1)*16)拼接,
4. 其他情况,bank_data(w*16+32-1,w*16)

valid信号四种情况

w=0,恒为高

w=1,如果之前的指令为bank_prev_is_half,或者不满足括号条件(之前的指令有效但不是RVC指令),说明这个inst和之前的inst无关,valid拉高

w=bankWidth -1,这里列举所有情况:

1. 本条不是RVC,且上条也不是RVC:1.本条指令和上一条是一条指令,那么本条指令就无效,本条指令是下一个bank的前半部分指令,那么本条就为有效
2. 本条不是RVC,但上一条是RVC:恒为高
3. 本条是RVC,但上一条不是RVC,恒为高,因为上一条一定是32bit指令的后半部分,其bank_mask一定为低,!((bank_mask(w-1) &&!isRVC(bank_insts(w-1)))一定为高
4. 本条是RVC,上条也是RVC:恒为高

其他情况:只要上条指令不满足(bank_mask(w-1) &&!isRVC(bank_insts(w-1),就为高(上条指令无效,上条指令为32bit指令的后半部分或上条指令为RVC指令)

> 如下面的矩形,绿色代表4字节的指令,蓝色代表2字节的指令,四个块一个bank,其中情况1的b>0情况,第四个块就是last_inst,b=0的情况就是第一个块为4字节指令的后一半,前一半在f3_prev_half中存储,也就是之前的指令包的w=bankWidth -1,的指令

![1732108893805](image/diplomacy&boom/1732108893805.png)

```
    for (w <- 0 until bankWidth) {
...
      val brsigs = Wire(new BranchDecodeSignals)
      if (w == 0) {
        val inst0 = Cat(bank_data(15,0), f3_prev_half)
        val inst1 = bank_data(31,0)
...

        when (bank_prev_is_half) {
          bank_insts(w)                := inst0
...
          if (b > 0) {
            val inst0b     = Cat(bank_data(15,0), last_inst)
...
            when (f3_bank_mask(b-1)) {
              bank_insts(w)                := inst0b
              f3_fetch_bundle.insts(i)     := inst0b
              f3_fetch_bundle.exp_insts(i) := exp_inst0b
              brsigs                       := bpd_decoder0b.io.out
            }
          }
        } .otherwise {
          bank_insts(w)                := inst1
...
        }
        valid := true.B
      } else {
        val inst = Wire(UInt(32.W))
..
        val pc = f3_aligned_pc + (i << log2Ceil(coreInstBytes)).U
...
        bank_insts(w)                := inst
...
        if (w == 1) {
          // Need special case since 0th instruction may carry over the wrap around
          inst  := bank_data(47,16)
          valid := bank_prev_is_half || !(bank_mask(0) && !isRVC(bank_insts(0)))
        } else if (w == bankWidth - 1) {
          inst  := Cat(0.U(16.W), bank_data(bankWidth*16-1,(bankWidth-1)*16))
          valid := !((bank_mask(w-1) && !isRVC(bank_insts(w-1))) ||
            !isRVC(inst))
        } else {
          inst  := bank_data(w*16+32-1,w*16)
          valid := !(bank_mask(w-1) && !isRVC(bank_insts(w-1)))
        }
      }
   last_inst = bank_insts(bankWidth-1)(15,0)
   ...
    }
```

OK,bank信号已经解释完了,接下来进行分支指令解码

#### 分支指令预解码

ExpandRVC判断这个指令是否为RVC,如果为RVC,返回相应的扩展指令,如果不是RVC,直接返回输入的inst,inst0和1对应的是两种情况,一种是本指令包的第一条为32位指令,但有一半在上个指令包,另一种就是指令是整齐的

> 如果这个指令对应两条RVC指令呢?
>
> RVC和RVI指令如何区分的呢
>
> f3_bank_mask信号有用吗

```
        val inst0 = Cat(bank_data(15,0), f3_prev_half)
        val inst1 = bank_data(31,0)
        val exp_inst0 = ExpandRVC(inst0)
        val exp_inst1 = ExpandRVC(inst1)//inst0和1分别对应了RVI指令和未知的指令
        val pc0 = (f3_aligned_pc + (i << log2Ceil(coreInstBytes)).U - 2.U)
        val pc1 = (f3_aligned_pc + (i << log2Ceil(coreInstBytes)).U)

```

分支预解码也是分情况

1. w=0,如果遇到了不完整的指令,就采用decoder0的结果,b>0,同样要做出处理,将inst0的f3_prev_half换为last_inst(其实这里bank_prev_half也可以),之后对这个指令解码就可以,否则就使用inst1的解码结果
2. 其他情况,就直接对inst解码,注意inst的生成也会分情况(之前讲过)

```
 for (b <- 0 until nBanks) {
...
    for (w <- 0 until bankWidth) {
...
      val brsigs = Wire(new BranchDecodeSignals)
      if (w == 0) {
        val inst0 = Cat(bank_data(15,0), f3_prev_half)
        val inst1 = bank_data(31,0)
        val exp_inst0 = ExpandRVC(inst0)
        val exp_inst1 = ExpandRVC(inst1)//inst0和1分别对应了RVI指令和未知的指令
        val pc0 = (f3_aligned_pc + (i << log2Ceil(coreInstBytes)).U - 2.U)
        val pc1 = (f3_aligned_pc + (i << log2Ceil(coreInstBytes)).U)
        val bpd_decoder0 = Module(new BranchDecode)
        bpd_decoder0.io.inst := exp_inst0
        bpd_decoder0.io.pc   := pc0
        val bpd_decoder1 = Module(new BranchDecode)
        bpd_decoder1.io.inst := exp_inst1
        bpd_decoder1.io.pc   := pc1

        when (bank_prev_is_half) {
          bank_insts(w)                := inst0
...
          bpu.io.pc                    := pc0
          brsigs                       := bpd_decoder0.io.out//指令不完整.且一定为32位,选择decode0的br信号
...
          if (b > 0) {
            val inst0b     = Cat(bank_data(15,0), last_inst)
            val exp_inst0b = ExpandRVC(inst0b)
            val bpd_decoder0b = Module(new BranchDecode)
            bpd_decoder0b.io.inst := exp_inst0b
            bpd_decoder0b.io.pc   := pc0

            when (f3_bank_mask(b-1)) {
...
              brsigs                       := bpd_decoder0b.io.out
            }
          }
        } .otherwise {
...
          bpu.io.pc                    := pc1
          brsigs                       := bpd_decoder1.io.out

        }
        valid := true.B
      } else {
        val inst = Wire(UInt(32.W))
        val exp_inst = ExpandRVC(inst)
        val pc = f3_aligned_pc + (i << log2Ceil(coreInstBytes)).U
        val bpd_decoder = Module(new BranchDecode)
        bpd_decoder.io.inst := exp_inst
        bpd_decoder.io.pc   := pc
...
        bpu.io.pc                    := pc
        brsigs                       := bpd_decoder.io.out
...
      }

...
  }
```

f3阶段的目标来自多个地方(f3_targs):如果是jalr指令,那么目标地址只能为bpd预测的地址,如果是条件分支或者jal,目标地址就是解码出来的地址

如果是jal指令:需要对目标地址检测,如果目标地址预测正确,不刷新BTB表项,

如果进行重定向,那么先检测是不是ret指令.如果是,就从RAS取出数据,否则从f3_targs取数据,

如果不重定向,就对pc+bankbyte或者fetchbyte

```
      f3_targs (i) := Mux(brsigs.cfi_type === CFI_JALR,
        f3_bpd_resp.io.deq.bits.preds(i).predicted_pc.bits,
        brsigs.target)

      // Flush BTB entries for JALs if we mispredict the target
      f3_btb_mispredicts(i) := (brsigs.cfi_type === CFI_JAL && valid &&
        f3_bpd_resp.io.deq.bits.preds(i).predicted_pc.valid &&
        (f3_bpd_resp.io.deq.bits.preds(i).predicted_pc.bits =/= brsigs.target)
      )


      f3_npc_plus4_mask(i) := (if (w == 0) {
        !f3_is_rvc(i) && !bank_prev_is_half
      } else {
        !f3_is_rvc(i)
      })
...  
  val f3_predicted_target = Mux(f3_redirects.reduce(_||_),
    Mux(f3_fetch_bundle.cfi_is_ret && useBPD.B && useRAS.B,
      ras.io.read_addr,
      f3_targs(PriorityEncoder(f3_redirects))
    ),
    nextFetch(f3_fetch_bundle.pc)
  )
```

#### 总结

F3阶段算是前端的一个核心阶段,这个阶段进行分支预解码,TAGE出结果,并且对RVC指令检测,将RVC变为32位指令,之后将f3_fetch_bundle送入F4

### F4

F4阶段主要进行的工作就是重定向操作,这个在F0中已经讲解,f4阶段还会将指令写入Fetchbuffer和FTQ

f4阶段还会修复前端的BTB或RAS,首先有一个仲裁器选择重定向信息来自FTQ还是f4阶段的BTB重定向信息,(低位优先级高),如果FTQ传来RAS修复信号,就对RAS进行修复

```
  val bpd_update_arbiter = Module(new Arbiter(new BranchPredictionUpdate, 2))
  bpd_update_arbiter.io.in(0).valid := ftq.io.bpdupdate.valid
  bpd_update_arbiter.io.in(0).bits  := ftq.io.bpdupdate.bits
  assert(bpd_update_arbiter.io.in(0).ready)
  bpd_update_arbiter.io.in(1) <> f4_btb_corrections.io.deq
  bpd.io.update := bpd_update_arbiter.io.out
  bpd_update_arbiter.io.out.ready := true.B

  when (ftq.io.ras_update && enableRasTopRepair.B) {
    ras.io.write_valid := true.B
    ras.io.write_idx   := ftq.io.ras_update_idx
    ras.io.write_addr  := ftq.io.ras_update_pc
  }
```

### F5

虚拟的阶段,主要对将IFU数据送入IDU,进行重定向操作

## BOOM FTQ

获取目标队列是一个队列，用于保存从 i-cache 接收到的 PC 以及与该地址关联的分支预测信息。它保存此信息，供管道在执行其[微操作 (UOP)](https://docs.boom-core.org/en/latest/sections/terminology.html#term-micro-op-uop)时参考。一旦提交指令，ROB 就会将其从队列中移出，并在重定向/误推测期间进行更新。

### 入队

当do_enq拉高，表示入队信号拉高，进入入队逻辑，new_entry和new_ghist接受入队数据，如现阶段有分支预测失败，就将入队glist写入new_list，否则，按照之前的数据更新new_list,然后写入ghist和lhist

### 重定向

> 为什么bpd_idx要增加
>
> 为什么要用两个ghist,

如下面波形,bpd_repair就是ftq_idx对应的pc

![1732436351046](image/diplomacy&boom/1732436351046.png)

```
//下面是一次预测失败要经过的状态
//| br_info   |  b2     |reg(b2)        |             |             |  
//|   b1      |  red_val|mispred(false) |mispred(true)|mispred(false)|mispred(false)|          |____
//|           |         |repair(false)  |repair(false)|repair(true) |repair(true)  |          |    |一直运行直到修复完成
//                |                        repair_idx   repair_idx     repair_pc   |repair_idx| +__| 
//                |                        end_idx                     repair_idx  |          |
// (找到分支预测失败的ftq表项)                                          ()  
//

  when (io.redirect.valid) {
    bpd_update_mispredict := false.B
    bpd_update_repair     := false.B
  } .elsewhen (RegNext(io.brupdate.b2.mispredict)) {
    bpd_update_mispredict := true.B
    bpd_repair_idx        := RegNext(io.brupdate.b2.uop.ftq_idx)
    bpd_end_idx           := RegNext(enq_ptr)
  } .elsewhen (bpd_update_mispredict) {//
    bpd_update_mispredict := false.B
    bpd_update_repair     := true.B
    bpd_repair_idx        := WrapInc(bpd_repair_idx, num_entries)
  } .elsewhen (bpd_update_repair && RegNext(bpd_update_mispredict)) {
    bpd_repair_pc         := bpd_pc
    bpd_repair_idx        := WrapInc(bpd_repair_idx, num_entries)
  } .elsewhen (bpd_update_repair) {
    bpd_repair_idx        := WrapInc(bpd_repair_idx, num_entries)
    when (WrapInc(bpd_repair_idx, num_entries) === bpd_end_idx ||
      bpd_pc === bpd_repair_pc)  {
      bpd_update_repair := false.B
    }

  }
```

分支预测失败的状态机如上面所示,

接下来就是传入更新信息,首先将enq_ptr设置为传入的ftq_idx+1,如果这个重定向来自分支预测失败,就将更新信息写入redirect_new_entry,然后下个周期将更新信息写入prev_entry,将重定向的信息写入entry_ram;

```
  when (io.redirect.valid) {//传入更新信息
    enq_ptr    := WrapInc(io.redirect.bits, num_entries)

    when (io.brupdate.b2.mispredict) {
    val new_cfi_idx = (io.brupdate.b2.uop.pc_lob ^
      Mux(redirect_entry.start_bank === 1.U, 1.U << log2Ceil(bankBytes), 0.U))(log2Ceil(fetchWidth), 1)
.......
    }

.......

  } .elsewhen (RegNext(io.redirect.valid)) {//信息传入完成
    prev_entry := RegNext(redirect_new_entry)
    prev_ghist := bpd_ghist
    prev_pc    := bpd_pc

    ram(RegNext(io.redirect.bits)) := RegNext(redirect_new_entry)
  }
```

### 后端读pc

有两个端口,其中0端口是送入后端jmp_unit的,端口1主要是进行重定向获取pc的,主要代码如下

```
  for (i <- 0 until 2) {
    val idx = io.get_ftq_pc(i).ftq_idx
    val next_idx = WrapInc(idx, num_entries)
    val next_is_enq = (next_idx === enq_ptr) && io.enq.fire
    val next_pc = Mux(next_is_enq, io.enq.bits.pc, pcs(next_idx))
    val get_entry = ram(idx)
    val next_entry = ram(next_idx)
    io.get_ftq_pc(i).entry     := RegNext(get_entry)
    if (i == 1)
      io.get_ftq_pc(i).ghist   := ghist(1).read(idx, true.B)
    else
      io.get_ftq_pc(i).ghist   := DontCare
    io.get_ftq_pc(i).pc        := RegNext(pcs(idx))
    io.get_ftq_pc(i).next_pc   := RegNext(next_pc)
    io.get_ftq_pc(i).next_val  := RegNext(next_idx =/= enq_ptr || next_is_enq)
    io.get_ftq_pc(i).com_pc    := RegNext(pcs(Mux(io.deq.valid, io.deq.bits, deq_ptr)))
  }
```

> 这些bpd_pc和mispred以及repair到底是干什么的
>
> 一条分支指令处理的流程
>
> globalhistory的current_saw_branch_not_taken是干什么的
>
> cfi这些信号是干什么的?

## Fetch Buffer

Fetch Buffer本质上是一个FIFO，寄存器堆构成,主要是作为缓冲,其可以配置为流式fifo，Fetch Buffer每次从F4阶段输入一个Fetch Packets，根据掩码将无效指令去掉后，从Buffer的尾部进入，每次从Buffer的头部输出coreWidth（后续流水线并行执行的宽度）个指令到译码级。

### 入队出队信号

might_hit_head得出这次访问可能会满,at_head表示tail已经和head重叠了,只有前面信号都不满足,才可以写入

> 假如fb大小为8项,每次最多写入4条,最多读出四条,假设连续写入两次,这时候tail和head就重合了,表示写满了

will_hit_tail信号揭示了head是否会和tail重合,也就是指令是否还够取(每次必须corewidth条)

> 参数和上一个例子一样,假如没有读出head为01,然后tail指针为0000 1000,表示写入了三条指令,这样得出来的tail_collisions就为0000 1000,然后will_hit_tail就为高,表示内部没有四条指令(妙)

```
  def rotateLeft(in: UInt, k: Int) = {
    val n = in.getWidth
    Cat(in(n-k-1,0), in(n-1, n-k))
  }

  val might_hit_head = (1 until fetchWidth).map(k => VecInit(rotateLeft(tail, k).asBools.zipWithIndex.filter
    {case (e,i) => i % coreWidth == 0}.map {case (e,i) => e}).asUInt).map(tail => head & tail).reduce(_|_).orR
  val at_head = (VecInit(tail.asBools.zipWithIndex.filter {case (e,i) => i % coreWidth == 0}
    .map {case (e,i) => e}).asUInt & head).orR
  val do_enq = !(at_head && maybe_full || might_hit_head)

  io.enq.ready := do_enq
...  
val tail_collisions = VecInit((0 until numEntries).map(i =>
                          head(i/coreWidth) && (!maybe_full || (i % coreWidth != 0).B))).asUInt & tail
  val slot_will_hit_tail = (0 until numRows).map(i => tail_collisions((i+1)*coreWidth-1, i*coreWidth)).reduce(_|_)
  val will_hit_tail = slot_will_hit_tail.orR

  val do_deq = io.deq.ready && !will_hit_tail
```

### 转换输入

代码如下,注意当w=0,需要考虑edge_inst,

```
  for (b <- 0 until nBanks) {
    for (w <- 0 until bankWidth) {
      val i = (b * bankWidth) + w
      val pc = (bankAlign(io.enq.bits.pc) + (i << 1).U)
      in_mask(i)                := io.enq.valid && io.enq.bits.mask(i)
...

      if (w == 0) {
        when (io.enq.bits.edge_inst(b)) {
          in_uops(i).debug_pc  := bankAlign(io.enq.bits.pc) + (b * bankBytes).U - 2.U
          in_uops(i).pc_lob    := bankAlign(io.enq.bits.pc) + (b * bankBytes).U
          in_uops(i).edge_inst := true.B
        }
      }
      in_uops(i).ftq_idx        := io.enq.bits.ftq_idx
      in_uops(i).inst           := io.enq.bits.exp_insts(i)
      in_uops(i).debug_inst     := io.enq.bits.insts(i)
      in_uops(i).is_rvc         := io.enq.bits.insts(i)(1,0) =/= 3.U
      in_uops(i).taken          := io.enq.bits.cfi_idx.bits === i.U && io.enq.bits.cfi_idx.valid

      in_uops(i).xcpt_pf_if     := io.enq.bits.xcpt_pf_if
      in_uops(i).xcpt_ae_if     := io.enq.bits.xcpt_ae_if
      in_uops(i).bp_debug_if    := io.enq.bits.bp_debug_if_oh(i)
      in_uops(i).bp_xcpt_if     := io.enq.bits.bp_xcpt_if_oh(i)

      in_uops(i).debug_fsrc     := io.enq.bits.fsrc
    }
  }
```

### 生成oh写索引

向量大小为fetchwidth=8,如果输入指令是有效的,就写入inc的索引,否则写入之前的值

> tail初始值为1,之后如果inc就将最高位移入最低位,哪一位为1就说明写入哪一位

```
  val enq_idxs = Wire(Vec(fetchWidth, UInt(numEntries.W)))

  def inc(ptr: UInt) = {
    val n = ptr.getWidth
    Cat(ptr(n-2,0), ptr(n-1))
  }

  var enq_idx = tail
  for (i <- 0 until fetchWidth) {
    enq_idxs(i) := enq_idx
    enq_idx = Mux(in_mask(i), inc(enq_idx), enq_idx)
  }
```

#### 写入fb

只将有效的写入fb,也就是,如果入队信号拉高,并且输入指令有效,且找到对应的写索引,就将数据写入fb

```
  for (i <- 0 until fetchWidth) {
    for (j <- 0 until numEntries) {
      when (do_enq && in_mask(i) && enq_idxs(i)(j)) {
        ram(j) := in_uops(i)
      }
    }
  }
```

#### 出队信号

deq_vec就是把fb数据转换换为出队的,这里i/coreWidth得出的是出去的是第几行,i%coreWidth表示的是行内的哪条uops,然后使用Mux1H选出head的前corewidth条数据

```
  // Generate vec for dequeue read port.
  for (i <- 0 until numEntries) {
    deq_vec(i/coreWidth)(i%coreWidth) := ram(i)
  }

  io.deq.bits.uops zip deq_valids           map {case (d,v) => d.valid := v}
  io.deq.bits.uops zip Mux1H(head, deq_vec) map {case (d,q) => d.bits  := q}
  io.deq.valid := deq_valids.reduce(_||_)
```

#### 指针状态更新

如果入队信号来了,就修改tail指针为enq_idx,出队就inc head指针,如果clear,就重置指针

```
  when (do_enq) {
    tail := enq_idx
    when (in_mask.reduce(_||_)) {
      maybe_full := true.B
    }
  }

  when (do_deq) {
    head := inc(head)
    maybe_full := false.B
  }

  when (io.clear) {
    head := 1.U
    tail := 1.U
    maybe_full := false.B
  }
```

#### 总结

这里使用oh编码来对地址编码,然后fb还通过一些特殊的方法来判断head和tail关系,十分巧妙

## 分支预测器

composer模块将各个模块的请求和更新连接到IO,然后将各个模块的meta送出,

> 所有模块共用meta,只不过是使用的位域不同,传入的meta同理,update信息之所以reverse,是因为低位的meta对应的是靠后的components

```
  var metas = 0.U(1.W)
  var meta_sz = 0
  for (c <- components) {
    c.io.f0_valid  := io.f0_valid
    c.io.f0_pc     := io.f0_pc
    c.io.f0_mask   := io.f0_mask
    c.io.f1_ghist  := io.f1_ghist
    c.io.f1_lhist  := io.f1_lhist
    c.io.f3_fire   := io.f3_fire
    if (c.metaSz > 0) {
      metas = (metas << c.metaSz) | c.io.f3_meta(c.metaSz-1,0)
    }
    meta_sz = meta_sz + c.metaSz
  }
  require(meta_sz < bpdMaxMetaLength)
  io.f3_meta := metas


  var update_meta = io.update.bits.meta
  for (c <- components.reverse) {
    c.io.update := io.update
    c.io.update.bits.meta := update_meta
    update_meta = update_meta >> c.metaSz
  }
```

### BranchPredictor

分支预测器的选择都是在下面代码中,这里是分bank的,然后返回的为ComposedBranchPredictorBank

> 为什么分bank?
>
> respose_in是干什么的

```
  val bpdStr = new StringBuilder
  bpdStr.append(BoomCoreStringPrefix("==Branch Predictor Memory Sizes==\n"))
  val banked_predictors = (0 until nBanks) map ( b => {
    val m = Module(if (useBPD) new ComposedBranchPredictorBank else new NullBranchPredictorBank)
    for ((n, d, w) <- m.mems) {
      bpdStr.append(BoomCoreStringPrefix(f"bank$b $n: $d x $w = ${d * w / 8}"))
      total_memsize = total_memsize + d * w / 8
    }
    m
  })
  bpdStr.append(BoomCoreStringPrefix(f"Total bpd size: ${total_memsize / 1024} KB\n"))
  override def toString: String = bpdStr.toString
```

然后这个bank内主要就是分发逻辑,将更新信号分发到每个预测器,以及将预测信息送出,下面代码中getBPDComponents就是获得预测器信息,然后返回预测结果

```
  val (components, resp) = getBPDComponents(io.resp_in(0), p)
  io.resp := resp
```

最终的分支预测信息来自下面代码,这是典型的TAGE_L结构,分支预测器的主要器件都包含在内

![1732447150001](image/diplomacy&boom/1732447150001.png)

#### 预测请求传入

预测请求分bank讨论,但这里只讨论bank为2的情况,只考虑全局历史

1. 传入请求的bank为0,这时bank0预测这个vpc,bank1预测下个bank的vpc
2. 如果传入请求的bank为1,就让bank0预测下一个bank,bank预测这个bank

具体代码如下

```
....
    when (bank(io.f0_req.bits.pc) === 0.U) {
.......

      banked_predictors(0).io.f0_valid := io.f0_req.valid
      banked_predictors(0).io.f0_pc    := bankAlign(io.f0_req.bits.pc)
      banked_predictors(0).io.f0_mask  := fetchMask(io.f0_req.bits.pc)

      banked_predictors(1).io.f0_valid := io.f0_req.valid
      banked_predictors(1).io.f0_pc    := nextBank(io.f0_req.bits.pc)
      banked_predictors(1).io.f0_mask  := ~(0.U(bankWidth.W))
    } .otherwise {
....
      banked_predictors(0).io.f0_valid := io.f0_req.valid && !mayNotBeDualBanked(io.f0_req.bits.pc)
      banked_predictors(0).io.f0_pc    := nextBank(io.f0_req.bits.pc)
      banked_predictors(0).io.f0_mask  := ~(0.U(bankWidth.W))
      banked_predictors(1).io.f0_valid := io.f0_req.valid
      banked_predictors(1).io.f0_pc    := bankAlign(io.f0_req.bits.pc)
      banked_predictors(1).io.f0_mask  := fetchMask(io.f0_req.bits.pc)
    }
    when (RegNext(bank(io.f0_req.bits.pc) === 0.U)) {
      banked_predictors(0).io.f1_ghist  := RegNext(io.f0_req.bits.ghist.histories(0))
      banked_predictors(1).io.f1_ghist  := RegNext(io.f0_req.bits.ghist.histories(1))
    } .otherwise {
      banked_predictors(0).io.f1_ghist  := RegNext(io.f0_req.bits.ghist.histories(1))
      banked_predictors(1).io.f1_ghist  := RegNext(io.f0_req.bits.ghist.histories(0))
    }
```

#### 预测结果传出

首先获得bank0和bank1的有效信号b0_fire,b1_fire,然后预测器送出f3阶段的预测信号,代码如下

```
    val b0_fire = io.f3_fire && RegNext(RegNext(RegNext(banked_predictors(0).io.f0_valid)))
    val b1_fire = io.f3_fire && RegNext(RegNext(RegNext(banked_predictors(1).io.f0_valid)))
    banked_predictors(0).io.f3_fire := b0_fire
    banked_predictors(1).io.f3_fire := b1_fire

    banked_lhist_providers(0).io.f3_fire := b0_fire
    banked_lhist_providers(1).io.f3_fire := b1_fire
    // The branch prediction metadata is stored un-shuffled
    io.resp.f3.meta(0)    := banked_predictors(0).io.f3_meta
    io.resp.f3.meta(1)    := banked_predictors(1).io.f3_meta

    io.resp.f3.lhist(0)   := banked_lhist_providers(0).io.f3_lhist
    io.resp.f3.lhist(1)   := banked_lhist_providers(1).io.f3_lhist

...

    when (bank(io.resp.f3.pc) === 0.U) {
      for (i <- 0 until bankWidth) {
        io.resp.f3.preds(i)           := banked_predictors(0).io.resp.f3(i)
        io.resp.f3.preds(i+bankWidth) := banked_predictors(1).io.resp.f3(i)
      }
    } .otherwise {
      for (i <- 0 until bankWidth) {
        io.resp.f3.preds(i)           := banked_predictors(1).io.resp.f3(i)
        io.resp.f3.preds(i+bankWidth) := banked_predictors(0).io.resp.f3(i)
      }
    }

```

#### 更新逻辑

将输入的更新信息送入每个bank,这里给出仿真图辅助理解,指令包的起始地址80000004位于bank0,所以bank0的valid一定为1,但cfi_valid却为0,因为输入的cfi_idx为6,说明分支在第六条,不在这个bank,所以bank0的cfi_valid为0

![1732456469917](image/diplomacy&boom/1732456469917.png)

接下来会基于largeboom(tage_l)来解析各个器件的主要逻辑,这些模块的IO都基于BranchPredictorBank,首先就是输入的分支预测请求,然后有预测信号resp,还有就是更新信号update,这三个信号是核心信号

```
  val io = IO(new Bundle {
    val f0_valid = Input(Bool())
    val f0_pc    = Input(UInt(vaddrBitsExtended.W))
    val f0_mask  = Input(UInt(bankWidth.W))
    // Local history not available until end of f1
    val f1_ghist = Input(UInt(globalHistoryLength.W))
    val f1_lhist = Input(UInt(localHistoryLength.W))

    val resp_in = Input(Vec(nInputs, new BranchPredictionBankResponse))
    val resp = Output(new BranchPredictionBankResponse)

    // Store the meta as a UInt, use width inference to figure out the shape
    val f3_meta = Output(UInt(bpdMaxMetaLength.W))

    val f3_fire = Input(Bool())

    val update = Input(Valid(new BranchPredictionBankUpdate))
  })
```

### NLP分支预测

NLP的分支预测结构由BIM表,RAS和BTB组成,如过查询BTB是ret,说明目标来自RAS,如果条目是无条件跳转,不查询BIM,

#### UBTB

![1732457973586](image/diplomacy&boom/1732457973586.png)

每个BTB条目对应的tag都是整个fetch_packet的pc这样的预测粒度就是一整个packet,当前端或者BPD被重定向,BTB更新,如果分支没找到条目,就分配一个条目

> BTB更新的tricky:

UBTB默认参数如下

```
case class BoomFAMicroBTBParams(
  nWays: Int = 16,
  offsetSz: Int = 13
)
```

##### 预测逻辑

首先检查是否hitBTB,如果hit,就预测地址,从btb取出偏移量,得出最终地址,同时得出是br还是jal,以及是否taken,br默认不taken

```
  for (w <- 0 until bankWidth) {
    val entry_meta = meta(s1_hit_ways(w))(w)
    s1_resp(w).valid := s1_valid && s1_hits(w)
    s1_resp(w).bits  := (s1_pc.asSInt + (w << 1).S + btb(s1_hit_ways(w))(w).offset).asUInt
    s1_is_br(w)      := s1_resp(w).valid &&  entry_meta.is_br
    s1_is_jal(w)     := s1_resp(w).valid && !entry_meta.is_br
    s1_taken(w)      := !entry_meta.is_br || entry_meta.ctr(1)

    s1_meta.hits(w)     := s1_hits(w)
  }
...
  for (w <- 0 until bankWidth) {
    io.resp.f1(w).predicted_pc := s1_resp(w)
    io.resp.f1(w).is_br        := s1_is_br(w)
    io.resp.f1(w).is_jal       := s1_is_jal(w)
    io.resp.f1(w).taken        := s1_taken(w)

    io.resp.f2(w) := RegNext(io.resp.f1(w))
    io.resp.f3(w) := RegNext(io.resp.f2(w))
  }
```

如果未命中,就会采用下面的分配逻辑,

> 这个分配逻辑暂时未搞明白是什么,可能涉及到了折叠,可以看分支历史的折叠

```
  val alloc_way = {
    val r_metas = Cat(VecInit(meta.map(e => VecInit(e.map(_.tag)))).asUInt, s1_idx(tagSz-1,0))
    val l = log2Ceil(nWays)
    val nChunks = (r_metas.getWidth + l - 1) / l
    val chunks = (0 until nChunks) map { i =>
      r_metas(min((i+1)*l, r_metas.getWidth)-1, i*l)
    }
    chunks.reduce(_^_)
  }
  s1_meta.write_way := Mux(s1_hits.reduce(_||_),
    PriorityEncoder(s1_hit_ohs.map(_.asUInt).reduce(_|_)),
    alloc_way)
```

##### 更新逻辑

BTB的更新主要分为更新offset和更新标签,更新offset,只要找到需要更新的way,然后将数据,传入这个way就可以,

更新meta,主要看ctr计数器,如果一开始这一项在预测时没有命中(新分配的项),则先初始化ctr,否则即使根据was_taken更新这个ctr计数器

```
  // Write the BTB with the target
  when (s1_update.valid && s1_update.bits.cfi_taken && s1_update.bits.cfi_idx.valid && s1_update.bits.is_commit_update) {
    btb(s1_update_write_way)(s1_update_cfi_idx).offset := new_offset_value
  }

  // Write the meta
  for (w <- 0 until bankWidth) {
    when (s1_update.valid && s1_update.bits.is_commit_update &&
      (s1_update.bits.br_mask(w) ||
        (s1_update_cfi_idx === w.U && s1_update.bits.cfi_taken && s1_update.bits.cfi_idx.valid))) {
      val was_taken = (s1_update_cfi_idx === w.U && s1_update.bits.cfi_idx.valid &&
        (s1_update.bits.cfi_taken || s1_update.bits.cfi_is_jal))

      meta(s1_update_write_way)(w).is_br := s1_update.bits.br_mask(w)
      meta(s1_update_write_way)(w).tag   := s1_update_idx
      meta(s1_update_write_way)(w).ctr   := Mux(!s1_update_meta.hits(w),
        Mux(was_taken, 3.U, 0.U),
        bimWrite(meta(s1_update_write_way)(w).ctr, was_taken)
      )
    }
  }
```

#### BIM

BIM使用pc一部分索引,只在提交时更新(饱和计数器,即使少更新,只要训练到位,预测结果大差不差)

##### 方向预测逻辑

BIM的默认set为2048,并且BIMset只能为2的幂次方,该预测器在f2阶段之后可以给出结果,s2阶段的resp就是预测方向信息,如果s2阶段有效,并且这个bank读出的bim表的项第1位为1,表示taken,否则为0

> 注意,这里感觉浪费了空间,因为BIM的写入都是对每个w写入相同内容,而且读出也是相同,所以每个w读出的也是一样的

```
  val s2_req_rdata    = RegNext(data.read(s0_idx   , s0_valid))

  val s2_resp         = Wire(Vec(bankWidth, Bool()))

  for (w <- 0 until bankWidth) {

    s2_resp(w)        := s2_valid && s2_req_rdata(w)(1) && !doing_reset
    s2_meta.bims(w)   := s2_req_rdata(w)
  }
... 
 for (w <- 0 until bankWidth) {
    io.resp.f2(w).taken := s2_resp(w)
    io.resp.f3(w).taken := RegNext(io.resp.f2(w).taken)
  }
```

##### 更新逻辑

更新是在f1阶段,如果一个bank里有br指令(taken)或者jal,就说明taken,旧的BIM值是传入的重定向值,或者就是之前的bypass值

> 这里设置bypass主要就是为了减少SRAM访问次数,如果上次更新的数据idx和这次的一样,就直接把上次的值作为旧的值,否则就是之前读出的值(只有commit时才可以更新这个bypass值)

s1_update_wdata更新计数器的值,然后在提交时写入data,

> old_bim_value要得到的是正确的旧值,s1_update_meta可能是分支预测失败时传来的update值,bypass是提交的值,数据一定正确,而写入又是在提交阶段,所以old_value一定是正确的值,另一种做法就是在提交直接读出旧值,不过可能引入多余的延迟

> 为什么s1阶段更新,s2阶段给出预测结果?一方面防止同时读写,另一方面,s1阶段更新,s2阶段就可以享受到更新的结果

> 注意这里更新逻辑条件包括了jal/jalr指令,看之前的issue,说这个地方不对,但目前都没改

```
  for (w <- 0 until bankWidth) {
    s1_update_wmask(w)         := false.B
    s1_update_wdata(w)         := DontCare

    val update_pc = s1_update.bits.pc + (w << 1).U

    when (s1_update.bits.br_mask(w) ||
      (s1_update.bits.cfi_idx.valid && s1_update.bits.cfi_idx.bits === w.U)) {
      val was_taken = (
        s1_update.bits.cfi_idx.valid &&
        (s1_update.bits.cfi_idx.bits === w.U) &&
        (
          (s1_update.bits.cfi_is_br && s1_update.bits.br_mask(w) && s1_update.bits.cfi_taken) ||
          s1_update.bits.cfi_is_jal
        )
      )
      val old_bim_value = Mux(wrbypass_hit, wrbypass(wrbypass_hit_idx)(w), s1_update_meta.bims(w))

      s1_update_wmask(w)     := true.B

      s1_update_wdata(w)     := bimWrite(old_bim_value, was_taken)
    }


  }

  when (doing_reset || (s1_update.valid && s1_update.bits.is_commit_update)) {
    data.write(
      Mux(doing_reset, reset_idx, s1_update_index),
      Mux(doing_reset, VecInit(Seq.fill(bankWidth) { 2.U }), s1_update_wdata),
      Mux(doing_reset, (~(0.U(bankWidth.W))), s1_update_wmask.asUInt).asBools
    )
  }
```

#### RAS

RAS的逻辑比较简单,主要分为读逻辑和写逻辑

读RAS在f3阶段,判断指令是否为ret,写RAS在ftq传入更新RAS信息或者f3阶段的指令为call指令

```
class BoomRAS(implicit p: Parameters) extends BoomModule()(p)
{
  val io = IO(new Bundle {
    val read_idx   = Input(UInt(log2Ceil(nRasEntries).W))
    val read_addr  = Output(UInt(vaddrBitsExtended.W))

    val write_valid = Input(Bool())
    val write_idx   = Input(UInt(log2Ceil(nRasEntries).W))
    val write_addr  = Input(UInt(vaddrBitsExtended.W))
  })
  val ras = Reg(Vec(nRasEntries, UInt(vaddrBitsExtended.W)))

  io.read_addr := Mux(RegNext(io.write_valid && io.write_idx === io.read_idx),
    RegNext(io.write_addr),
    RegNext(ras(io.read_idx)))

  when (io.write_valid) {
    ras(io.write_idx) := io.write_addr
  }
}

```

### BPD

BPD仅仅对条件分支的方向进行预测,其他信息,比如那些指令是分支,目标是什么,无需在意,这些信息可以从BTB得知,所以BPD无需存储tag和分支目标地址,jal和jalr指令均由NLP预测,如果NLP预测失败,只能之后重定向

![1732524502617](image/diplomacy&boom/1732524502617.png)

BPD在f3给出结果,f4进行重定向,

BPD采用全局历史,GHR进行推测更新,每个分支都有GHR快照,同时在BPD维护提交阶段的GHR

> **请注意，在F0**阶段开始进行预测（读取全局历史记录时）和在F4阶段重定向[前端](https://docs.boom-core.org/en/latest/sections/terminology.html#term-front-end)（更新全局历史记录时）之间存在延迟。这会导致“影子”，其中在F0中开始进行预测的分支将看不到程序中一个（或两个）周期之前出现的分支（或其结果）（目前处于F1/2/3阶段）。但至关重要的是，这些“影子分支”必须反映在全局历史快照中。

> 每个[FTQ](https://docs.boom-core.org/en/latest/sections/terminology.html#term-fetch-target-queue-ftq)条目对应一个**提取**周期。对于每次预测，分支预测器都会打包稍后执行更新所需的数据。例如，分支预测器需要记住预测来自哪个 *索引，以便稍后更新该索引处的计数器。此数据存储在*[FTQ](https://docs.boom-core.org/en/latest/sections/terminology.html#term-fetch-target-queue-ftq)中。[当Fetch Packet](https://docs.boom-core.org/en/latest/sections/terminology.html#term-fetch-packet)中的最后一条指令被提交时，[FTQ条目将被释放并返回到分支预测器。使用存储在](https://docs.boom-core.org/en/latest/sections/terminology.html#term-fetch-target-queue-ftq)[FTQ](https://docs.boom-core.org/en/latest/sections/terminology.html#term-fetch-target-queue-ftq)条目中的数据，分支预测器可以对其预测状态执行任何所需的更新。

> FTQ保存着在提交期间更新分支预测器所需的分支预测器数据（无论是[正确](https://docs.boom-core.org/en/latest/sections/terminology.html#term-fetch-target-queue-ftq)预测还是错误预测）。但是，当分支预测器做出错误预测时，需要额外的状态，必须立即更新。例如，如果发生错误预测，则必须将推测更新的GHR重置为正确值，然后处理器才能再次开始提取（和预测）。[](https://docs.boom-core.org/en/latest/sections/terminology.html#term-global-history-register-ghr)

> **此状态可能非常昂贵，但一旦在执行**阶段解析了分支，就可以释放它。因此，状态与[分支重命名](https://docs.boom-core.org/en/latest/sections/terminology.html#term-branch-rename-snapshot)快照并行存储。在**解码** 和**重命名**期间，会为每个分支分配一个 **分支标记** ，并制作重命名表的快照，以便在发生错误预测时进行单周期回滚。与分支标记和**重命名映射表**快照一样， 一旦分支在 执行阶段由分支单元解析，就可以释放相应的[分支重命名快照](https://docs.boom-core.org/en/latest/sections/terminology.html#term-branch-rename-snapshot)。

##### 抽象分支类

![1732526267100](image/diplomacy&boom/1732526267100.png)

#### TAGE

TAGE的默认参数如下.可以看到BOOM例化了6个表,最大历史长度为64,并且ubit的更新周期为2048个周期,饱和计数器为3bits,user为2bit,

```
case class BoomTageParams(
  //                                           nSets, histLen, tagSz
  tableInfo: Seq[Tuple3[Int, Int, Int]] = Seq((  128,       2,     7),
                                              (  128,       4,     7),
                                              (  256,       8,     8),
                                              (  256,      16,     8),
                                              (  128,      32,     9),
                                              (  128,      64,     9)),
  uBitPeriod: Int = 2048
)

```

##### TageTable

**预测阶段**

首先计算出hash_idx,根据该idx得出ctr和user_bit以及tag,然后将读出的信息传入tage进一步处理

**写入逻辑**

写入逻辑主要写入userbit,table

**table:写入提交阶段传入的update_idx(这里的update同样有bypass)**

```
  table.write(
    Mux(doing_reset, reset_idx                                          , update_idx),
    Mux(doing_reset, VecInit(Seq.fill(bankWidth) { 0.U(tageEntrySz.W) }), VecInit(update_wdata.map(_.asUInt))),
    Mux(doing_reset, ~(0.U(bankWidth.W))                                , io.update_mask.asUInt).asBools
  )

```

user_bit分为两个段:hi和lo,主要讲hi:

写入的idx来自reset_idx,clear_idx和update_idx,user_bit需要定期清0,clear前缀的就是清零有关信号,这里就是每2048个周期就去清零高位或者低位,

> 由于是sram结构,一周期只能读1写1,所以也没啥问题,但为啥不同时清0hi和lo,猜想可能是先缓冲一下

```
  val doing_clear_u = clear_u_ctr(log2Ceil(uBitPeriod)-1,0) === 0.U
  val doing_clear_u_hi = doing_clear_u && clear_u_ctr(log2Ceil(uBitPeriod) + log2Ceil(nRows)) === 1.U
  val doing_clear_u_lo = doing_clear_u && clear_u_ctr(log2Ceil(uBitPeriod) + log2Ceil(nRows)) === 0.U
  val clear_u_idx = clear_u_ctr >> log2Ceil(uBitPeriod)
...  
hi_us.write(
    Mux(doing_reset, reset_idx, Mux(doing_clear_u_hi, clear_u_idx, update_idx)),
    Mux(doing_reset || doing_clear_u_hi, VecInit((0.U(bankWidth.W)).asBools), update_hi_wdata),
    Mux(doing_reset || doing_clear_u_hi, ~(0.U(bankWidth.W)), io.update_u_mask.asUInt).asBools
  )
```

##### TAGE主要逻辑

首先，定义所有产生tag匹配的预测表中所需历史长度最长者为provider，而其余产生tag匹配的预测表（若存在的话）被称为altpred。

1. 当provider产生的预测被证实为一个正确的预测时，首先将产生的正确预测的对应provider表项的pred计数器自增1。其次，若此时的provider与altpred的预测结果不同，则provider的userfulness计数器自增1。
2. 当provider产生的预测被证实为一个错误的预测时，首先将产生的错误预测的对应provider表项的pred预测器自减1。其次，若存在产生正确预测的altpred，则provider的usefulness计数器自减1。接下来，若该provider所源自的预测表并非所需历史长度最高的预测表，则此时执行如下的表项增添操作。首先，读取所有历史长度长于provider的预测表的usefulness计数器，若此时有某表的u计数器值为0，则在该表中分配一对应的表项。当有多个预测表（如Tj,Tk两项）的u计数器均为0，则将表项分配给Tk的几率为分配给Tj的2^(k-j)倍（这一概率分配在硬件上可以通过一个LFSR来实现）。若所有TAGE内预测表的u值均不为0，则所有预测表的u值同时减1。
3. 只有provider和altpred的预测不同时才会更新

###### 预测逻辑

tage预测逻辑分为provider,和altpred,其中provider为历史最长的tag命中对应的table,altpred则是次高历史命中对应的table,如果table没有命中,则选择默认的结果,源论文为bim表得出的结果

> 这里暂时不清楚默认预测器是什么,应该也是bim表

这里首先遍历所有历史表,如果table hit,就将选择taken结果,如果ctr ===3.U|| ctr ===4.U,认为这个provider不可信,选择altpred的结果作为预测结果,否则选择ctr(2)为预测结果

```
    var altpred = io.resp_in(0).f3(w).taken
    val final_altpred = WireInit(io.resp_in(0).f3(w).taken)
    var provided = false.B
    var provider = 0.U
    io.resp.f3(w).taken := io.resp_in(0).f3(w).taken
    //
    for (i <- 0 until tageNTables) {
      val hit = f3_resps(i)(w).valid
      val ctr = f3_resps(i)(w).bits.ctr
      when (hit) {
        io.resp.f3(w).taken := Mux(ctr === 3.U || ctr === 4.U, altpred, ctr(2))//预测可能不准
        final_altpred       := altpred
      }

      provided = provided || hit
      provider = Mux(hit, i.U, provider)
      altpred  = Mux(hit, f3_resps(i)(w).bits.ctr(2), altpred)
    }
    f3_meta.provider(w).valid := provided
    f3_meta.provider(w).bits  := provider
    f3_meta.alt_differs(w)    := final_altpred =/= io.resp.f3(w).taken//有预测未命中的项
    f3_meta.provider_u(w)     := f3_resps(provider)(w).bits.u
    f3_meta.provider_ctr(w)   := f3_resps(provider)(w).bits.ctr
```

###### 更新逻辑

更新阶段就是去更新ctr和u计数器,如果预测失败可能还会去分配新的表项

allocatable_slots就是找到未命中并且u为0的slot,如果这个多于一个,就通过LSFR大概率选择分支历史长的,这样就得到了要分配的table表项,如果是提交阶段更新,并且是条件分支指令,如果此时provider是有效的,就将信息写入对应的table,然后更新u_bit,以及ctr计数器,代码如下

```
    val allocatable_slots = (
      VecInit(f3_resps.map(r => !r(w).valid && r(w).bits.u === 0.U)).asUInt &
      ~(MaskLower(UIntToOH(provider)) & Fill(tageNTables, provided))
    )
    val alloc_lfsr = random.LFSR(tageNTables max 2)//如果u=0的个数大于1,使用LSFR选择,概率是历史长的大于历史短的

    val first_entry = PriorityEncoder(allocatable_slots)
    val masked_entry = PriorityEncoder(allocatable_slots & alloc_lfsr)
    val alloc_entry = Mux(allocatable_slots(masked_entry),
      masked_entry,
      first_entry)

    f3_meta.allocate(w).valid := allocatable_slots =/= 0.U
    f3_meta.allocate(w).bits  := alloc_entry

    val update_was_taken = (s1_update.bits.cfi_idx.valid &&
                            (s1_update.bits.cfi_idx.bits === w.U) &&
                            s1_update.bits.cfi_taken)
    when (s1_update.bits.br_mask(w) && s1_update.valid && s1_update.bits.is_commit_update) {
      when (s1_update_meta.provider(w).valid) {
        val provider = s1_update_meta.provider(w).bits

        s1_update_mask(provider)(w) := true.B
        s1_update_u_mask(provider)(w) := true.B

        val new_u = inc_u(s1_update_meta.provider_u(w),
                          s1_update_meta.alt_differs(w),
                          s1_update_mispredict_mask(w))
        s1_update_u      (provider)(w) := new_u
        s1_update_taken  (provider)(w) := update_was_taken
        s1_update_old_ctr(provider)(w) := s1_update_meta.provider_ctr(w)
        s1_update_alloc  (provider)(w) := false.B

      }
    }

```

###### 分配逻辑

分配阶段其实是在更新阶段内的,但有自己独特的操作,故列出单讲

首先分配表项是在提交阶段,发现provider预测失败,并且这个表项的表不是分支历史最长的表,进行表项分配,如果找到了可以分配的表项,就对表项分配,并且将对应的table表项u置为0,如果没有找到表项,就将符合条件的表项u置为0,但是不分配表项

> 分配还会初始化ctr,原论文中新分配的表项为弱taken(4),这里只有这次更新taken才为4,否则为3

> 这里好像boom和源论文做法不一样,原论文是将ubit递减,而不是直接置为0

主要代码如下

```
  when (s1_update.valid && s1_update.bits.is_commit_update && s1_update.bits.cfi_mispredicted && s1_update.bits.cfi_idx.valid) {
    val idx = s1_update.bits.cfi_idx.bits
    val allocate = s1_update_meta.allocate(idx)
    when (allocate.valid) {
      s1_update_mask (allocate.bits)(idx) := true.B
      s1_update_taken(allocate.bits)(idx) := s1_update.bits.cfi_taken
      s1_update_alloc(allocate.bits)(idx) := true.B

      s1_update_u_mask(allocate.bits)(idx) := true.B
      s1_update_u     (allocate.bits)(idx) := 0.U

    } .otherwise {
      val provider = s1_update_meta.provider(idx)
      val decr_mask = Mux(provider.valid, ~MaskLower(UIntToOH(provider.bits)), 0.U)

      for (i <- 0 until tageNTables) {
        when (decr_mask(i)) {
          s1_update_u_mask(i)(idx) := true.B
          s1_update_u     (i)(idx) := 0.U
        }
      }
    }

  }
```

### 总结

目前前端逻辑还没搞明白ghist内部信号到底什么含义,还有wrbypass是为了干什么

> 举个例子理解wrbypass,
>
> 假设下面指令:
>
> test:	addi a1,a1,1
>
> bne a1,a2,test
>
> 提交阶段可能是
>
> 假设bne为指令包1,addi,bne为指令包2,那么有
>
> bne		|addi,bne				|
>
> s0		|s1					|s2
>
> |指令包1写入新的bim值	|写入完成,同时也写入bypass内
>
> |					|指令包2写入新的bim值,此时旧的bim值来自bypass的,本身带的bim值太老
>
> 上面这种情况就解释了wrbypass的作用:及时的更新正确的bim值,防止出现performance bug

> ghist是推测更新,也就是在分支预测每个阶段都会更新:
>
> 在f1阶段,这主要是UBTB,如果是br指令并且taken,就更新ghist
>
> f2阶段是bim的结果,bim实际上也不需要使用ghist,f2阶段预测的按理一定是br分支,但boom加入了jal,绝对会对ghr产生影响
>
> f3阶段是tage预测阶段,这个阶段ghist才有作用,
>
> 在f2和f3对之前的分支预测目标和方向进行检查,只要一个不满足,就重定向
>
> 之后就是后端传来的重定向信号,

# 分支预测全流程

分支指令在boom中会经过预测/推测更新阶段(ifu)->检测/重定向阶段(exu)->更新阶段,boom采用的checkpoint来恢复CPU状态,每个分支都有自己的掩码,分支预测失败根据这个掩码定向冲刷指令,更新,刷新,重定向前端,

## 预测阶段

分支指令的预测阶段主要在F1,F2,F3阶段.这三个阶段会送出BPD的预测信息,并进行重定向操作,这个可以看之前IFU流水线讲解的F0阶段和F1阶段

> 目前的问题是,一个fetchpacket可能有多条分支指令,如何去正确记录分支历史,比如bne,bne 指令包.前一个不taken,后一个taken,这时候就要正确记录之前没有taken的指令历史,可能这个是按照bank更新分支历史,
>
> 分支预测是将一个指令包的指令全部送进去预测,分别得出结果

预测阶段每个周期都会有新的ghist生成,比如在f3阶段有f3_predicted_ghist,这个就是更新后的历史,注意这个存的还是旧历史,但分支的taken信息已经包含在内了,假如f3 taken,对前面重定向,f1_predicted_ghist,读出的旧历史就是f3阶段更新后的历史(他会延迟更新,等到其他的去update,才会更新旧值)

> 注意,此时存入ftq的ghist不是f3_predicted_ghist,而是f3_fetch_bundle.ghist,也就是相当于只存入的旧值,并未存入taken信息

```
  val f3_predicted_ghist = f3_fetch_bundle.ghist.update(
    f3_fetch_bundle.br_mask,
    f3_fetch_bundle.cfi_idx.valid,
    f3_fetch_bundle.br_mask(f3_fetch_bundle.cfi_idx.bits),
    f3_fetch_bundle.cfi_idx.bits,
    f3_fetch_bundle.cfi_idx.valid,
    f3_fetch_bundle.pc,
    f3_fetch_bundle.cfi_is_call,
    f3_fetch_bundle.cfi_is_ret
  )
```

## 检测阶段

(alu)这里主要对br指令进行了检测,br或者jalr,目标地址可能出错,所以会对方向检测,如果pc_sel为npc,就说明实际不taken,预测失败就说明前端预测taken,如果为PC_BRJMP就说明实际taken,就需要对预测的taken信号取反

```
 when (is_br || is_jalr) {
    if (!isJmpUnit) {
      assert (pc_sel =/= PC_JALR)
    }
    when (pc_sel === PC_PLUS4) {
      mispredict := uop.taken
    }
    when (pc_sel === PC_BRJMP) {
      mispredict := !uop.taken
    }
  }

  val brinfo = Wire(new BrResolutionInfo)
  // note: jal doesn't allocate a branch-mask, so don't clear a br-mask bit
  brinfo.valid          := is_br || is_jalr
  brinfo.mispredict     := mispredict
  brinfo.uop            := uop
  brinfo.cfi_type       := Mux(is_jalr, CFI_JALR,
                           Mux(is_br  , CFI_BR, CFI_X))
  brinfo.taken          := is_taken
  brinfo.pc_sel         := pc_sel
  brinfo.jalr_target    := DontCare
```

如果此时发生分支预测失败,就将分支预测失败路径指令全部删除,并且重定向前端,修改前端信息,重定向信息分为b1,b2,其中b1是在第一个周期br_mask,b2就是携带了重定向信息(第二个周期),

```
  val b1 = new BrUpdateMasks
  // On the second cycle we get indices to reset pointers
  val b2 = new BrResolutionInfo

```

在core.scala中,如果发现了mispredict,就要得出真正预测的目标,以及重定向信号

```
    val use_same_ghist = (brupdate.b2.cfi_type === CFI_BR &&//只有条件分支预测方向
                          !brupdate.b2.taken &&//实际不用跳转
                          bankAlign(block_pc) === bankAlign(npc))//最后一个条件意思是npc也在这个block内,如果是这样,那抹其实不需要更新ghist,
    val ftq_entry = io.ifu.get_pc(1).entry
    val cfi_idx = (brupdate.b2.uop.pc_lob ^
      Mux(ftq_entry.start_bank === 1.U, 1.U << log2Ceil(bankBytes), 0.U))(log2Ceil(fetchWidth), 1)//得到这个分支的位置
    val ftq_ghist = io.ifu.get_pc(1).ghist
    val next_ghist = ftq_ghist.update(
      ftq_entry.br_mask.asUInt,
      brupdate.b2.taken,
      brupdate.b2.cfi_type === CFI_BR,
      cfi_idx,
      true.B,
      io.ifu.get_pc(1).pc,
      ftq_entry.cfi_is_call && ftq_entry.cfi_idx.bits === cfi_idx,
      ftq_entry.cfi_is_ret  && ftq_entry.cfi_idx.bits === cfi_idx)


    io.ifu.redirect_ghist   := Mux(
      use_same_ghist,
      ftq_ghist,
      next_ghist)
    io.ifu.redirect_ghist.current_saw_branch_not_taken := use_same_ghist
```

## 重定向阶段

如果分支预测失败,进入重定向逻辑,刷新前端,此时读出ftq对应表项的内容,包括ghist

> 猜测分支预测的粒度是bank,这样use_same_ghist就可以解释清楚了,如果没有taken,并且npc和这个指令在同一个bank,则认为这个分支可以使用和ftq一样的历史,然后将current_saw_branch_not_taken置为高,之后如果update就会发现有分支未taken
>
> 这样current_saw_branch_not_taken也可以解释清楚了

对于ghist选择有ftq_ghist和next_ghist,根据use_same_ghist选择对应的分支历史

```
    val use_same_ghist = (brupdate.b2.cfi_type === CFI_BR &&//只有条件分支预测方向
                          !brupdate.b2.taken &&//实际不用跳转
                          bankAlign(block_pc) === bankAlign(npc))//最后一个条件意思是npc也在这个block内,如果是这样,那么其实不需要更新ghist,如果是以bank为粒度预测,那么这个分支相当于没有预测,所以不计入历史
...
    val ftq_ghist = io.ifu.get_pc(1).ghist
    val next_ghist = ftq_ghist.update(
      ftq_entry.br_mask.asUInt,
      brupdate.b2.taken,
      brupdate.b2.cfi_type === CFI_BR,
      cfi_idx,
      true.B,
      io.ifu.get_pc(1).pc,
      ftq_entry.cfi_is_call && ftq_entry.cfi_idx.bits === cfi_idx,
      ftq_entry.cfi_is_ret  && ftq_entry.cfi_idx.bits === cfi_idx)
    io.ifu.redirect_ghist   := Mux(
      use_same_ghist,
      ftq_ghist,
      next_ghist)
```

# BOOM Decode

首先就是IO,Decode模块的enq是传入的指令,deq是输出的指令,之后是CSR逻辑,和中断,BOOM模块主要就是复用lrocket的decodelogic模块,其他并无特色的地方

```
class DecodeUnitIo(implicit p: Parameters) extends BoomBundle
{
  val enq = new Bundle { val uop = Input(new MicroOp()) }
  val deq = new Bundle { val uop = Output(new MicroOp()) }

  // from CSRFile
  val status = Input(new freechips.rocketchip.rocket.MStatus())
  val csr_decode = Flipped(new freechips.rocketchip.rocket.CSRDecodeIO)
  val interrupt = Input(Bool())
  val interrupt_cause = Input(UInt(xLen.W))
}
```

# BOOM RENAME

boom采用的是统一的PRF结构，

![1731307352707](image/diplomacy/1731307352707.png)

RAT就是图中的map table，busytable揭示每个物理寄存器的忙碌情况，

## Busy table

busytable在唤醒阶段把寄存器设置为空闲，在rename阶段将寄存器设置为忙

首先列出输入输出信号

```
  val io = IO(new BoomBundle()(p) {
    val ren_uops = Input(Vec(plWidth, new MicroOp))
    val busy_resps = Output(Vec(plWidth, new BusyResp))
    val rebusy_reqs = Input(Vec(plWidth, Bool()))

    val wb_pdsts = Input(Vec(numWbPorts, UInt(pregSz.W)))
    val wb_valids = Input(Vec(numWbPorts, Bool()))

    val debug = new Bundle { val busytable = Output(Bits(numPregs.W)) }
  })

```

ren_uops表示查询busytable，busy_reps表示寄存器的忙碌状态，wb前缀的表示写回阶段要更新的寄存器状态，最后一个是debug信号

```
  val busy_table = RegInit(0.U(numPregs.W))
  // Unbusy written back registers.
  val busy_table_wb = busy_table & ~(io.wb_pdsts zip io.wb_valids)
    .map {case (pdst, valid) => UIntToOH(pdst) & Fill(numPregs, valid.asUInt)}.reduce(_|_)
  // Rebusy newly allocated registers.
  val busy_table_next = busy_table_wb | (io.ren_uops zip io.rebusy_reqs)
    .map {case (uop, req) => UIntToOH(uop.pdst) & Fill(numPregs, req.asUInt)}.reduce(_|_)

  busy_table := busy_table_next
```

接下来是主要模块，首先将写回的寄存器unbusy，我们看busy_table_wb，首先看io.wb_pdsts zip io.wb_valids表示将两个作为一个元组，然后使用map函数，对每个院组都进行操作，操作的内容是后面｛｝内容，这个｛首先使用模式匹配case，然后输出的值是=>后面的值，也就是把写回的寄存器变成oh编码，然后把这些元素通过reduce按位或，得到写回寄存器的oh编码，然后取非再&busytable，就相当于释放了写回的寄存器

之后的busy_table_next，就是为寄存器分配忙位

```
  // Read the busy table.
  for (i <- 0 until plWidth) {
    val prs1_was_bypassed = (0 until i).map(j =>
      io.ren_uops(i).lrs1 === io.ren_uops(j).ldst && io.rebusy_reqs(j)).foldLeft(false.B)(_||_)
    val prs2_was_bypassed = (0 until i).map(j =>
      io.ren_uops(i).lrs2 === io.ren_uops(j).ldst && io.rebusy_reqs(j)).foldLeft(false.B)(_||_)
    val prs3_was_bypassed = (0 until i).map(j =>
      io.ren_uops(i).lrs3 === io.ren_uops(j).ldst && io.rebusy_reqs(j)).foldLeft(false.B)(_||_)

    io.busy_resps(i).prs1_busy := busy_table(io.ren_uops(i).prs1) || prs1_was_bypassed && bypass.B
    io.busy_resps(i).prs2_busy := busy_table(io.ren_uops(i).prs2) || prs2_was_bypassed && bypass.B
    io.busy_resps(i).prs3_busy := busy_table(io.ren_uops(i).prs3) || prs3_was_bypassed && bypass.B
    if (!float) io.busy_resps(i).prs3_busy := false.B
  }

  io.debug.busytable := busy_table
```

然后就是读busytable，这个的意思就是先检查写入的新映射关系有没有和src1一样的，有的话就说明这个可能有依赖（也即是RAW），也就是这个寄存器在使用，之后只要busytable和prs1_was_bypassed一个成立，就说明这个寄存器在使用

## Map table

其实就是RAT，首先先把交互信号放上来，以供后续阅读

```
class MapReq(val lregSz: Int) extends Bundle
{
  val lrs1 = UInt(lregSz.W)
  val lrs2 = UInt(lregSz.W)
  val lrs3 = UInt(lregSz.W)
  val ldst = UInt(lregSz.W)
}

class MapResp(val pregSz: Int) extends Bundle
{
  val prs1 = UInt(pregSz.W)
  val prs2 = UInt(pregSz.W)
  val prs3 = UInt(pregSz.W)
  val stale_pdst = UInt(pregSz.W)
}

class RemapReq(val lregSz: Int, val pregSz: Int) extends Bundle
{
  val ldst = UInt(lregSz.W)
  val pdst = UInt(pregSz.W)
  val valid = Bool()
}
```

然后就是Maptable的IO信号了，主要就是映射请求，映射答复，重新映射，保存snapshot，恢复snapshot

```
  val io = IO(new BoomBundle()(p) {
    // Logical sources -> physical sources.
    val map_reqs    = Input(Vec(plWidth, new MapReq(lregSz)))
    val map_resps   = Output(Vec(plWidth, new MapResp(pregSz)))

    // Remapping an ldst to a newly allocated pdst?
    val remap_reqs  = Input(Vec(plWidth, new RemapReq(lregSz, pregSz)))

    // Dispatching branches: need to take snapshots of table state.
    val ren_br_tags = Input(Vec(plWidth, Valid(UInt(brTagSz.W))))

    // Signals for restoring state following misspeculation.
    val brupdate      = Input(new BrUpdateInfo)
    val rollback    = Input(Bool())
  })
```

接下来就是这个模块的主要信号，首先map_table就是这个模块的核心了，存储寄存器映射关系的，然后就是snapshot，这里为什么要remap？就是把最新的寄存器关系写进去，具体需要看重命名过程干了什么（逻辑源寄存器读RAT，目的寄存器在freelist找空闲，目的寄存器读RAT，将读出的值写入ROB，目的寄存器写入RAT，更新新的映射关系）这样其实就理解了设置这些信号的含义，remap_pdsts就是把物理寄存器号提取出来，如果一周期重命名2条，那么这个就是一个大小为2的向量，remap_ldsts_oh就是给每个逻辑寄存器编码，假设两条指令目的寄存器为1，3，那么编码后的就是（32‘b...10,32'b...1000）

```
  // The map table register array and its branch snapshots.
  val map_table = RegInit(VecInit(Seq.fill(numLregs){0.U(pregSz.W)}))
  val br_snapshots = Reg(Vec(maxBrCount, Vec(numLregs, UInt(pregSz.W))))

  // The intermediate states of the map table following modification by each pipeline slot.
  val remap_table = Wire(Vec(plWidth+1, Vec(numLregs, UInt(pregSz.W))))

  // Uops requesting changes to the map table.
  val remap_pdsts = io.remap_reqs map (_.pdst)
  val remap_ldsts_oh = io.remap_reqs map (req => UIntToOH(req.ldst) & Fill(numLregs, req.valid.asUInt))
```

然后弄明白新的每个指令新的映射关系，第一个意思就是把0号寄存器清0，如果不是0号寄存器，就设置一个remapped_row，这个的大小是plwidth的大小，这个之后的意思就是，为每个逻辑寄存器找到他的映射关系是来自RAT还是传入的映射关系,我们首先需要知道scanleft的意思，这个的工作模式如下（从左到右依次是reduce，fold，scan），这个remapped_row干的事情就是先把ldst位提取出来，这表示哪个逻辑寄存器是有更新请求，然后zip pdst形成元组，假设有如下映射ldst1->pdst2,ldst3->pdst4,这里前面是逻辑。后面是物理，假设一周期2条指令，i=1，这个zip形成的元组就是（true，2），（false，2），然后scanleft（有累积性）的初值为map_table（1）,也就是remapped_row第0个元素为来自map的值，然后这句话生成的元组就是（map，pdst2，pdst2），map为来自map-table的物理寄存器，最后把这些赋值给remaptable,然后假如i=3，remapped_row就是（map，map，pdst4），此时remap_table（1）为（0，pdst2，map，map，...）remap（2）为（0，pdst2，map，pdst4，...）所以这里可以看到remaptable的最高索引才是正确的映射关系（巧妙但晦涩难懂的操作）

![1731476588727](image/diplomacy/1731476588727.png)

```
  // Figure out the new mappings seen by each pipeline slot.
  for (i <- 0 until numLregs) {
    if (i == 0 && !float) {
      for (j <- 0 until plWidth+1) {
        remap_table(j)(i) := 0.U
      }
    } else {
      val remapped_row = (remap_ldsts_oh.map(ldst => ldst(i)) zip remap_pdsts)
        .scanLeft(map_table(i)) {case (pdst, (ldst, new_pdst)) => Mux(ldst, new_pdst, pdst)}

      for (j <- 0 until plWidth+1) {
        remap_table(j)(i) := remapped_row(j)
      }
    }
  }
```

然后更新新的映射关系，最后就是读map，注意这个处理了读出的映射关系是来自map_table还是remap请求(处理RAW)，当i=0，映射关系来自RAT，（也就是第1条指令，最旧的指令）只讲解i=1情况的prs1，foldleft和scan类似，但只输出最终结果，所以这里就是检查第一条的目的寄存器和这一条指令（也就是第二条）的源寄存器是否相等，如果相等就使用新的映射

```
  when (io.brupdate.b2.mispredict) {
    // Restore the map table to a branch snapshot.
    map_table := br_snapshots(io.brupdate.b2.uop.br_tag)
  } .otherwise {
    // Update mappings.
    map_table := remap_table(plWidth)
  }

  // Read out mappings.
  for (i <- 0 until plWidth) {
    io.map_resps(i).prs1       := (0 until i).foldLeft(map_table(io.map_reqs(i).lrs1)) ((p,k) =>
      Mux(bypass.B && io.remap_reqs(k).valid && io.remap_reqs(k).ldst === io.map_reqs(i).lrs1, io.remap_reqs(k).pdst, p))
    io.map_resps(i).prs2       := (0 until i).foldLeft(map_table(io.map_reqs(i).lrs2)) ((p,k) =>
      Mux(bypass.B && io.remap_reqs(k).valid && io.remap_reqs(k).ldst === io.map_reqs(i).lrs2, io.remap_reqs(k).pdst, p))
    io.map_resps(i).prs3       := (0 until i).foldLeft(map_table(io.map_reqs(i).lrs3)) ((p,k) =>
      Mux(bypass.B && io.remap_reqs(k).valid && io.remap_reqs(k).ldst === io.map_reqs(i).lrs3, io.remap_reqs(k).pdst, p))
    io.map_resps(i).stale_pdst := (0 until i).foldLeft(map_table(io.map_reqs(i).ldst)) ((p,k) =>
      Mux(bypass.B && io.remap_reqs(k).valid && io.remap_reqs(k).ldst === io.map_reqs(i).ldst, io.remap_reqs(k).pdst, p))

    if (!float) io.map_resps(i).prs3 := DontCare
  }
```

然后这个链接对高阶函数做了简单总结：[高级设计](https://zhuanlan.zhihu.com/p/350301092)

## Free list

先列出IO信号

```
  val io = IO(new BoomBundle()(p) {
    // Physical register requests.
    val reqs          = Input(Vec(plWidth, Bool()))
    val alloc_pregs   = Output(Vec(plWidth, Valid(UInt(pregSz.W))))

    // Pregs returned by the ROB.
    val dealloc_pregs = Input(Vec(plWidth, Valid(UInt(pregSz.W))))

    // Branch info for starting new allocation lists.
    val ren_br_tags   = Input(Vec(plWidth, Valid(UInt(brTagSz.W))))

    // Mispredict info for recovering speculatively allocated registers.
    val brupdate        = Input(new BrUpdateInfo)

    val debug = new Bundle {
      val pipeline_empty = Input(Bool())
      val freelist = Output(Bits(numPregs.W))
      val isprlist = Output(Bits(numPregs.W))
    }
  })
```

首先明白free list什么时候分配寄存器，什么时候写入用完的寄存器（分别是重命名阶段，和提交阶段），然后就明白上面信号什么意思了

```
  // The free list register array and its branch allocation lists.
  val free_list = RegInit(UInt(numPregs.W), ~(1.U(numPregs.W)))
  val br_alloc_lists = Reg(Vec(maxBrCount, UInt(numPregs.W)))

  // Select pregs from the free list.
  val sels = SelectFirstN(free_list, plWidth)
  val sel_fire  = Wire(Vec(plWidth, Bool()))

  // Allocations seen by branches in each pipeline slot.
  val allocs = io.alloc_pregs map (a => UIntToOH(a.bits))
  val alloc_masks = (allocs zip io.reqs).scanRight(0.U(n.W)) { case ((a,r),m) => m | a & Fill(n,r) }

  // Masks that modify the freelist array.
  val sel_mask = (sels zip sel_fire) map { case (s,f) => s & Fill(n,f) } reduce(_|_)
  val br_deallocs = br_alloc_lists(io.brupdate.b2.uop.br_tag) & Fill(n, io.brupdate.b2.mispredict)
  val dealloc_mask = io.dealloc_pregs.map(d => UIntToOH(d.bits)(numPregs-1,0) & Fill(n,d.valid)).reduce(_|_) | br_deallocs

  val br_slots = VecInit(io.ren_br_tags.map(tag => tag.valid)).asUInt
```

然后free_list是一个size为物理寄存器个数的寄存器，介绍sels之前先介绍PriorityEncoderOH，这个就是返回第一个为true的oh编码，然后sel是就是找到4个为true的索引，并且为oh编码，然后就是sel_mask,这个就是将sels得到的oh组合起来，dealloc_mask就是从ROB返回的物理寄存器，把他转换为onehot，（这里不管分支预测的snapshot），

```
object PriorityEncoderOH {
  private def encode(in: Seq[Bool]): UInt = {
    val outs = Seq.tabulate(in.size)(i => (BigInt(1) << i).asUInt(in.size.W))
    PriorityMux(in :+ true.B, outs :+ 0.U(in.size.W))
  }
  def apply(in: Seq[Bool]): Seq[Bool] = {
    val enc = encode(in)
    Seq.tabulate(in.size)(enc(_))
  }
  def apply(in: Bits): UInt = encode((0 until in.getWidth).map(i => in(i)))
}
```

然后freelist更新，之后就是读出分配好的寄存器,这里有个sel_fire,注意这里的逻辑有些混乱,

```
  // Update the free list.
  free_list := (free_list & ~sel_mask | dealloc_mask) & ~(1.U(numPregs.W))

  // Pipeline logic | hookup outputs.
  for (w <- 0 until plWidth) {
    val can_sel = sels(w).orR
    val r_valid = RegInit(false.B)
    val r_sel   = RegEnable(OHToUInt(sels(w)), sel_fire(w))

    r_valid := r_valid && !io.reqs(w) || can_sel
    sel_fire(w) := (!r_valid || io.reqs(w)) && can_sel

    io.alloc_pregs(w).bits  := r_sel
    io.alloc_pregs(w).valid := r_valid
  }
```

## RenameStage

直接看链接[重命名](https://zhuanlan.zhihu.com/p/399543947)

其实有个问题：maptable本身支持解决RAW，但在rename模块将bypass给关闭了，然后在rename注册了BypassAllocations检查RAW相关，

还有：

rename有两级；第一级主要进行读RAT，第二阶段写RAT，读出freelist，写busytable（链接认为第一阶段还有读freelsit，但代码内使用的却是ren2_uops，也就是第二级）

其实感觉这里是一个比较逆天的操作,只看黄色框内容,由于r_sel是一个寄存器,在en后下个周期才可以得出新的值,这里虽然en(s2送入的请求)了,但实际上下个周期才会响应这个en,这里读出的还是之前的旧数据,但注意,这个旧寄存器值同样也是空闲的,因为他是由上一条指令读的,且freelist已经标记这个寄存器被分配出去了,非常逆天的操作,使用上个指令请求,然后这条指令正好读出,然后s2阶段就可以进行RAW检查了,这个操作完全可以在s1阶段产生请求,然后s2读出数据,还有下面这行代码,这个得结合流水线看,我们重命名一部分在decode/rename,另一部分在rename/dispatch,s1阶段主要进行读物理源寄存器(RAT),s2阶段读物理目的寄存器,然后把新的映射关系写入RAT,**所以我们不仅要处理组内相关性,还要处理组间相关性**,这句就是处理组间相关性,因为假设B指令的源寄存器和A指令的目的寄存器一样(一周期rename一条,B是新指令),B指令在s1读出的物理源寄存器可能不是最新的映射关系(A指令还没写入RAT),所以需要这行

```
    r_uop := GetNewUopAndBrMask(BypassAllocations(next_uop, ren2_uops, ren2_alloc_reqs), io.brupdate)
```

![1731584801027](image/diplomacy&boom/1731584801027.png)

下面简单讲一条指令在这个模块进行了什么操作：

### 读RAT请求和写RAT

```
  for ((((ren1,ren2),com),w) <- (ren1_uops zip ren2_uops zip io.com_uops.reverse).zipWithIndex) {
    map_reqs(w).lrs1 := ren1.lrs1
    map_reqs(w).lrs2 := ren1.lrs2
    map_reqs(w).lrs3 := ren1.lrs3
    map_reqs(w).ldst := ren1.ldst

    remap_reqs(w).ldst := Mux(io.rollback, com.ldst      , ren2.ldst)
    remap_reqs(w).pdst := Mux(io.rollback, com.stale_pdst, ren2.pdst)
  }
```

注意这里map_reqs是ren1传入，也就是从decode传入的，然后写入RAT就是ren2的逻辑和物理寄存器

### 读freelist

```
  // Freelist inputs.
  freelist.io.reqs := ren2_alloc_reqs
  freelist.io.dealloc_pregs zip com_valids zip rbk_valids map
    {case ((d,c),r) => d.valid := c || r}
  freelist.io.dealloc_pregs zip io.com_uops map
    {case (d,c) => d.bits := Mux(io.rollback, c.pdst, c.stale_pdst)}
  freelist.io.ren_br_tags := ren2_br_tags
  freelist.io.brupdate := io.brupdate
  freelist.io.debug.pipeline_empty := io.debug_rob_empty

  assert (ren2_alloc_reqs zip freelist.io.alloc_pregs map {case (r,p) => !r || p.bits =/= 0.U} reduce (_&&_),
           "[rename-stage] A uop is trying to allocate the zero physical register.")

  // Freelist outputs.
  for ((uop, w) <- ren2_uops.zipWithIndex) {
    val preg = freelist.io.alloc_pregs(w).bits
    uop.pdst := Mux(uop.ldst =/= 0.U || float.B, preg, 0.U)
  }
```

可以看到我们请求的前缀为ren2

### 读busytable

```
  busytable.io.ren_uops := ren2_uops  // expects pdst to be set up.
  busytable.io.rebusy_reqs := ren2_alloc_reqs
  busytable.io.wb_valids := io.wakeups.map(_.valid)
  busytable.io.wb_pdsts := io.wakeups.map(_.bits.uop.pdst)

  assert (!(io.wakeups.map(x => x.valid && x.bits.uop.dst_rtype =/= rtype).reduce(_||_)),
   "[rename] Wakeup has wrong rtype.")

  for ((uop, w) <- ren2_uops.zipWithIndex) {
    val busy = busytable.io.busy_resps(w)

    uop.prs1_busy := uop.lrs1_rtype === rtype && busy.prs1_busy
    uop.prs2_busy := uop.lrs2_rtype === rtype && busy.prs2_busy
    uop.prs3_busy := uop.frs3_en && busy.prs3_busy

    val valid = ren2_valids(w)
    assert (!(valid && busy.prs1_busy && rtype === RT_FIX && uop.lrs1 === 0.U), "[rename] x0 is busy??")
    assert (!(valid && busy.prs2_busy && rtype === RT_FIX && uop.lrs2 === 0.U), "[rename] x0 is busy??")
  }
```

同样是在阶段2进行

### 输出结果

```
  for (w <- 0 until plWidth) {
    val can_allocate = freelist.io.alloc_pregs(w).valid

    // Push back against Decode stage if Rename1 can't proceed.
    io.ren_stalls(w) := (ren2_uops(w).dst_rtype === rtype) && !can_allocate

    val bypassed_uop = Wire(new MicroOp)
    if (w > 0) bypassed_uop := BypassAllocations(ren2_uops(w), ren2_uops.slice(0,w), ren2_alloc_reqs.slice(0,w))
    else       bypassed_uop := ren2_uops(w)

    io.ren2_uops(w) := GetNewUopAndBrMask(bypassed_uop, io.brupdate)
  }
```

注意这里检测了一个指令包内的RAW，那我们还有WAW，但其实已经解决了，maptable的scanleft会写入最新的映射关系

## 总结

这里boom用了很多花活，巧妙但晦涩难懂，也体现了chisel的强大之处，本篇解读将分支预测失败的全部略过

# BOOM Dispatch

![1731596242893](image/diplomacy&boom/1731596242893.png)

首先上IO.ren_uops由rename传来，然后后面的dis_uops表示送入每个IQ的指令，假设N 个IQ，每个IQ周期每个周期都可以接受dispawidth指令

```
  // incoming microops from rename2
  val ren_uops = Vec(coreWidth, Flipped(DecoupledIO(new MicroOp)))

  // outgoing microops to issue queues
  // N issues each accept up to dispatchWidth uops
  // dispatchWidth may vary between issue queues
  val dis_uops = MixedVec(issueParams.map(ip=>Vec(ip.dispatchWidth, DecoupledIO(new MicroOp))))
```

然后就是boom目前使用的dispatcher,首先是ren_ready,也就是指令已经被写入IQ，这时把他拉高，注意这里所有指令只能去一个IQ，所以有一个reduce，检查所有指令是否都送入这个IQ了，然后就是把ren_uops请求分发到对应IQ，对于Boom，有三个IQ，FP，MEM和ALU，其中IQ和MEM为一个issue unit，每周期轮换，这个有的问题就是如果一周期指令既有MEM，又有INT，会导致某些指令无法全部发出

```
class BasicDispatcher(implicit p: Parameters) extends Dispatcher
{
  issueParams.map(ip=>require(ip.dispatchWidth == coreWidth))

  val ren_readys = io.dis_uops.map(d=>VecInit(d.map(_.ready)).asUInt).reduce(_&_)

  for (w <- 0 until coreWidth) {
    io.ren_uops(w).ready := ren_readys(w)
  }

  for {i <- 0 until issueParams.size
       w <- 0 until coreWidth} {
    val issueParam = issueParams(i)
    val dis        = io.dis_uops(i)

    dis(w).valid := io.ren_uops(w).valid && ((io.ren_uops(w).bits.iq_type & issueParam.iqType.U) =/= 0.U)
    dis(w).bits  := io.ren_uops(w).bits
  }
}
```

接下来为Boom没使用的模块，这个模块是每周期尽可能送入发射队列，也就是没有只能发射到一个IQ的限制，只有在IQ满了才会stall，

这个模块的ren_ready就很清晰，意思和上面的一样，然后循环体内就是主要逻辑,ren大小和ren_ops大小一样(corewidth),然后uses_iq就是指出指令要送去哪个IQ,之后就是为ren_valid赋值,假如这次循环是检测INT的,对于lw,add,sub就是(false,true,true),之后有一个Boom自己的api,Compactor,意思是找出前k个有效的输出,然后将输出链接到dis,最后得出这个IQ是否空闲,如果use_iq为false,就说明空闲,

```
/**
 *  Tries to dispatch as many uops as it can to issue queues,
 *  which may accept fewer than coreWidth per cycle.
 *  When dispatchWidth == coreWidth, its behavior differs
 *  from the BasicDispatcher in that it will only stall dispatch when
 *  an issue queue required by a uop is full.
 */
class CompactingDispatcher(implicit p: Parameters) extends Dispatcher
{
  issueParams.map(ip => require(ip.dispatchWidth >= ip.issueWidth))

  val ren_readys = Wire(Vec(issueParams.size, Vec(coreWidth, Bool())))

  for (((ip, dis), rdy) <- issueParams zip io.dis_uops zip ren_readys) {
    val ren = Wire(Vec(coreWidth, Decoupled(new MicroOp)))
    ren <> io.ren_uops

    val uses_iq = ren map (u => (u.bits.iq_type & ip.iqType.U).orR)

    // Only request an issue slot if the uop needs to enter that queue.
    (ren zip io.ren_uops zip uses_iq) foreach {case ((u,v),q) =>
      u.valid := v.valid && q}

    val compactor = Module(new Compactor(coreWidth, ip.dispatchWidth, new MicroOp))
    compactor.io.in  <> ren
    dis <> compactor.io.out

    // The queue is considered ready if the uop doesn't use it.
    rdy := ren zip uses_iq map {case (u,q) => u.ready || !q}
  }

  (ren_readys.reduce((r,i) =>
      VecInit(r zip i map {case (r,i) =>
        r && i})) zip io.ren_uops) foreach {case (r,u) =>
          u.ready := r}
}

```

接下来介绍**Compactor**,作用就是在n个valid选出k个,首先gen为数据的类型,首先IO为n入k出,如果n=k,就直接把输出连到输入,否则就要去选出前k个,sels得出的是选择哪一个的OH编码,假如in_valid为(0,1,1)

n=3,k=2,sels就为(0010,0100),in_readys的意思就是可以传入数据了,也就是这批指令已经分配完IQ了,这个模块的找前几个有效的数据设置也很巧妙,

# BOOM ROB

* [ ] 由于ROB有很多信号目前是从执行级传来，导致解析可能有误，之后会更新解析

![1731673694620](image/diplomacy&boom/1731673694620.png)

首先先理清,ROB在Dispatch写入指令信息,在提交阶段读出信息,提交总是最旧的指令,这里ROB是W个存储体(W=dispatch长度),每次写入ROB就是一个W宽度的指令信息,ROB仅存储一个指令包的首地址,bank(0)(指令包地址连续),但遇到分支指令就得产生气泡,重新开一行,不然无法读到正确的PC,**运行图就是下图,注意0x0008有问题,跳转地址为0x0028**

![1731675223257](image/diplomacy&boom/1731675223257.png)

## ROB状态机

ROB状态机有四个状态，这种情况是不含CRAT，也就是checkpoint，然后还有含有CRAT，这时候就会少一个s_rollback

![1731739470401](image/diplomacy&boom/1731739470401.png)

**is_unique** 信号是定义在 MicroOp 中的一个成员，表示只允许该指令一条指令存在于流水线中，流水线要对 is_unique 的指令做出的响应包括：

* 等待 STQ (Store Queue) 中的指令全部提交
* 清空该指令之后的取到的指令
* ROB 标记为 unready，等待清空

RISCV 指令集中 is_unique 有效的指令主要包括：

* CSR(Control and Status Register) 指令
* 原子指令
* 内存屏障指令
* 休眠指令
* 机器模式特权指令

下面是状态机代码

```
  // ROB FSM
  if (!enableCommitMapTable) {
    switch (rob_state) {
      is (s_reset) {
        rob_state := s_normal
      }
      is (s_normal) {
        // Delay rollback 2 cycles so branch mispredictions can drain
        when (RegNext(RegNext(exception_thrown))) {
          rob_state := s_rollback
        } .otherwise {
          for (w <- 0 until coreWidth) {
            when (io.enq_valids(w) && io.enq_uops(w).is_unique) {
              rob_state := s_wait_till_empty
            }
          }
        }
      }
      is (s_rollback) {
        when (empty) {
          rob_state := s_normal
        }
      }
      is (s_wait_till_empty) {
        when (RegNext(exception_thrown)) {
          rob_state := s_rollback
        } .elsewhen (empty) {
          rob_state := s_normal
        }
      }
    }
  } else {
    switch (rob_state) {
      is (s_reset) {
        rob_state := s_normal
      }
      is (s_normal) {
        when (exception_thrown) {
          ; //rob_state := s_rollback
        } .otherwise {
          for (w <- 0 until coreWidth) {
            when (io.enq_valids(w) && io.enq_uops(w).is_unique) {
              rob_state := s_wait_till_empty
            }
          }
        }
      }
      is (s_rollback) {
        when (rob_tail_idx  === rob_head_idx) {
          rob_state := s_normal
        }
      }
      is (s_wait_till_empty) {
        when (exception_thrown) {
          ; //rob_state := s_rollback
        } .elsewhen (rob_tail === rob_head) {
          rob_state := s_normal
        }
      }
    }
  }
```

## ROB输入

输入就是在dispatch入队

BOOM 为每个bank中的所有指令定义了若干变量记录重命名缓存的状态信息，主要包括：

* rob_val         ：当前 bank 中每行指令的有效信号，初始化为0
* rob_bsy        ：当前 bank 中每行指令的 busy 信号，busy=1时表示指令还在流水线中，当入队的指令不是fence或者fence.i都为busy，fence是保证内存顺序，不执行任何操作，故不busy
* rob_unsafe   ：当前 bank 中每行指令的 unsafe 信号，指令 safe 表示一定可以被提交
* rob_uop       ：当前 bank 中的每行指令

其中unsafe有四种情况：

* 使用LD队列
* 使用ST队列，并且不是fence指令
* 是分支或者jalr

```
def unsafe           = uses_ldq || (uses_stq && !is_fence) || is_br || is_jalr
```

当输入的指令有效时，就把相关信息写入ROB的tail位置

```
    when (io.enq_valids(w)) {
      rob_val(rob_tail)       := true.B
      rob_bsy(rob_tail)       := !(io.enq_uops(w).is_fence ||
                                   io.enq_uops(w).is_fencei)
      rob_unsafe(rob_tail)    := io.enq_uops(w).unsafe
      rob_uop(rob_tail)       := io.enq_uops(w)
      rob_exception(rob_tail) := io.enq_uops(w).exception
      rob_predicated(rob_tail)   := false.B
      rob_fflags(rob_tail)    := 0.U

      assert (rob_val(rob_tail) === false.B, "[rob] overwriting a valid entry.")
      assert ((io.enq_uops(w).rob_idx >> log2Ceil(coreWidth)) === rob_tail)
    } .elsewhen (io.enq_valids.reduce(_|_) && !rob_val(rob_tail)) {
      rob_uop(rob_tail).debug_inst := BUBBLE // just for debug purposes
    }

```

## 写回级操作

这个就是响应写回级操作，当写回有效，并且匹配到相关的bank，将busy和unsafe置为低，然后rob的pred设置为写回的pred，

```
    for (i <- 0 until numWakeupPorts) {
      val wb_resp = io.wb_resps(i)
      val wb_uop = wb_resp.bits.uop
      val row_idx = GetRowIdx(wb_uop.rob_idx)
      when (wb_resp.valid && MatchBank(GetBankIdx(wb_uop.rob_idx))) {
        rob_bsy(row_idx)      := false.B
        rob_unsafe(row_idx)   := false.B
        rob_predicated(row_idx)  := wb_resp.bits.predicated
      }
    }
```

## 响应LSU输入

> 注意：这里引用的[ROB](https://www.zhihu.com/search?type=content&q=boom%20ROB)，目前我还不太清楚LSU操作，故先引用，之后可能会加入自己理解

* lsu_clr_bsy       ：当要 LSU 模块正确接受了要保存的数据时，清除 store 命令的 busy 状态，同时将指令标记为 safe。clr_bsy 信号的值与存储目标地址是否有效、TLB是否命中、是否处于错误的分支预测下、该指令在存储队列中的状态等因素有关。
* lsu_clr_unsafe   ：推测 load 命令除了 Memory Ordering Failure 之外不会出现其他异常时，将 load 指令标记为 safe。lsu_clr_unsafe 信号要等广播异常之后才能输出，采用 RegNext 类型寄存器来延迟一个时钟周期。
* lxcpt                  ：来自LSU的异常，包括异常的指令、异常是否有效、异常原因等信息。异常的指令在 rob_exception 中对应的值将置为1。

```
    for (clr_rob_idx <- io.lsu_clr_bsy) {
      when (clr_rob_idx.valid && MatchBank(GetBankIdx(clr_rob_idx.bits))) {
        val cidx = GetRowIdx(clr_rob_idx.bits)
        rob_bsy(cidx)    := false.B
        rob_unsafe(cidx) := false.B
        assert (rob_val(cidx) === true.B, "[rob] store writing back to invalid entry.")
        assert (rob_bsy(cidx) === true.B, "[rob] store writing back to a not-busy entry.")
      }
    }
    for (clr <- io.lsu_clr_unsafe) {
      when (clr.valid && MatchBank(GetBankIdx(clr.bits))) {
        val cidx = GetRowIdx(clr.bits)
        rob_unsafe(cidx) := false.B
      }
    }
    when (io.lxcpt.valid && MatchBank(GetBankIdx(io.lxcpt.bits.uop.rob_idx))) {
      rob_exception(GetRowIdx(io.lxcpt.bits.uop.rob_idx)) := true.B
      when (io.lxcpt.bits.cause =/= MINI_EXCEPTION_MEM_ORDERING) {
        // In the case of a mem-ordering failure, the failing load will have been marked safe already.
        assert(rob_unsafe(GetRowIdx(io.lxcpt.bits.uop.rob_idx)),
          "An instruction marked as safe is causing an exception")
      }
    }
    can_throw_exception(w) := rob_val(rob_head) && rob_exception(rob_head)
```

**store** 命令特殊之处在于不需要写回 (Write Back) 寄存器，因此 LSU 模块将 store 指令从存储队列提交后，store 命令就可以从流水线中退休，即 io.lsu_clr_bsy 信号将 store 指令置为 safe 时同时置为 unbusy。

**MINI_EXCEPTION_MEM_ORDERING** 是指发生存储-加载顺序异常(Memory Ordering Failure)。当 store 指令与其后的 load 指令有共同的目标地址时，类似 RAW 冲突，若 load 指令在 store 之前发射(Issue)，load 命令将从内存中读取错误的值。处理器在提交 store 指令时需要检查是否发生了 Memory Ordering Failure，如果有，则需要刷新流水线、修改重命名映射表等。Memory Ordering Failure 是处理器乱序执行带来的问题,是处理器设计的缺陷，不属于 RISCV 规定的异常，采用 MINI_EXCEPTION_MEM_ORSERING 来弥补。

```scala
can_throw_exception(w) := rob_val(rob_head) && rob_exception(rob_head)
```

当位于 ROB头(head) 的指令有效且异常时，才允许抛出异常。

## 响应提交

在 ROB 头的指令有效且已不在流水线中且未收到来自 CSR 的暂停信号（例如wfi指令）时有效，表示此时在 ROB 头的指令可以提交。

```
can_commit(w) := rob_val(rob_head) && !(rob_bsy(rob_head)) && !io.csr_stall
```

提交和抛出异常只能在提交阶段

**will_commit** 这一段代码的主要作用是为 head 指针指向的 ROB 行中的每一个 bank 生成 will_commit 信号，will_commit 信号指示下一时钟周期指令是否提交。will_commit 信号有效的条件是：

* 该 bank 中的指令可以提交
* 该 bank 中的指令不会抛出异常
* ROB 的提交没有被封锁

**block_commit**  block_commit=1 时，ROB 既不能提交指令，也不能抛出异常。对于每个bank，都有一个自己的 block_commit 信号，只要一个 bank 被封锁提交，其后的所有 bank 都将被封锁提交。block_commit 信号保证 ROB 只能顺序提交。若 ROB 处于 s_rollback 或 s_reset 状态，或在前两个时钟周期内抛出异常时，block_commit将被初始化为1,即该行所有指令的提交都被封锁。

 **will_throw_exception** ： 表示下一时钟周期将要抛出异常，该信号初始化为0，使信号有效的条件包括：

* 当前bank可以抛出异常
* 没有封锁提交
* 上一个bank没有要提交的指令

```
  var block_commit = (rob_state =/= s_normal) && (rob_state =/= s_wait_till_empty) || RegNext(exception_thrown) || RegNext(RegNext(exception_thrown))
  var will_throw_exception = false.B
  var block_xcpt   = false.B

  for (w <- 0 until coreWidth) {
    will_throw_exception = (can_throw_exception(w) && !block_commit && !block_xcpt) || will_throw_exception

    will_commit(w)       := can_commit(w) && !can_throw_exception(w) && !block_commit
    block_commit         = (rob_head_vals(w) &&
                           (!can_commit(w) || can_throw_exception(w))) || block_commit
    block_xcpt           = will_commit(w)
  }
```

## 异常跟踪逻辑

ROB接受的异常信息来自两个方面：

* 前端发生的异常，输入端口为 io.enq_valid 和 io.enq_uops.exception
* LSU发生的异常，输入端口为 io.lxcpt

只存储最旧的异常，因为本来异常就冲刷流水线，之后的异常无意义，首先将dispatch异常原因写入enq_xcpts,

然后就是r_xcpt_uop的更新逻辑,

如果发生回滚，或者冲刷流水线，或者异常被抛出了，不更新，

如果是lsu的异常，首先将uop更新为lsu的uop，然后检查这个是否是最旧的异常（IsOlder）或者是否有效，如果是最旧的异常，或者r_xcpt_val无效，就进入更新逻辑更新next_xcpt_uop（其实就是next_xcpt_uop），

如果是dispatch的，且是最旧的指令，更新信息

如果这个异常位于分支预测失败路径，直接把r_xcpt_val无效

```
  val next_xcpt_uop = Wire(new MicroOp())
  next_xcpt_uop := r_xcpt_uop
  val enq_xcpts = Wire(Vec(coreWidth, Bool()))
  for (i <- 0 until coreWidth) {
    enq_xcpts(i) := io.enq_valids(i) && io.enq_uops(i).exception
  }

  when (!(io.flush.valid || exception_thrown) && rob_state =/= s_rollback) {
    when (io.lxcpt.valid) {
      val new_xcpt_uop = io.lxcpt.bits.uop

      when (!r_xcpt_val || IsOlder(new_xcpt_uop.rob_idx, r_xcpt_uop.rob_idx, rob_head_idx)) {
        r_xcpt_val              := true.B
        next_xcpt_uop           := new_xcpt_uop
        next_xcpt_uop.exc_cause := io.lxcpt.bits.cause
        r_xcpt_badvaddr         := io.lxcpt.bits.badvaddr
      }
    } .elsewhen (!r_xcpt_val && enq_xcpts.reduce(_|_)) {
      val idx = enq_xcpts.indexWhere{i: Bool => i}

      // if no exception yet, dispatch exception wins
      r_xcpt_val      := true.B
      next_xcpt_uop   := io.enq_uops(idx)
      r_xcpt_badvaddr := AlignPCToBoundary(io.xcpt_fetch_pc, icBlockBytes) | io.enq_uops(idx).pc_lob

    }
  }

  r_xcpt_uop         := next_xcpt_uop
  r_xcpt_uop.br_mask := GetNewBrMask(io.brupdate, next_xcpt_uop)
  when (io.flush.valid || IsKilledByBranch(io.brupdate, next_xcpt_uop)) {
    r_xcpt_val := false.B
  }
```

## 分支预测失败

主要是消除mask一样的分支，否则就更新这个指令的br_mask，

```
    // -----------------------------------------------
    // Kill speculated entries on branch mispredict
    for (i <- 0 until numRobRows) {
      val br_mask = rob_uop(i).br_mask

      //kill instruction if mispredict & br mask match
      when (IsKilledByBranch(io.brupdate, br_mask))
      {
        rob_val(i) := false.B
        rob_uop(i.U).debug_inst := BUBBLE
      } .elsewhen (rob_val(i)) {
        // clear speculation bit even on correct speculation
        rob_uop(i).br_mask := GetNewBrMask(io.brupdate, br_mask)
      }
    }
```

## ROB Head Logic

当一个bank的所有指令都可以提交，才可以改变head指针状态，finished_committing_row只有当commit指令有效，并且将在下个周期提交，并且head有效

* [ ] 弄明白r_partial_row是什么意思

这时就会自增ROB的head指针，否则将rob_head_lsb指向第一个为1的bank

```
  val rob_deq = WireInit(false.B)
  val r_partial_row = RegInit(false.B)

  when (io.enq_valids.reduce(_|_)) {
    r_partial_row := io.enq_partial_stall
  }

  val finished_committing_row =
    (io.commit.valids.asUInt =/= 0.U) &&
    ((will_commit.asUInt ^ rob_head_vals.asUInt) === 0.U) &&
    !(r_partial_row && rob_head === rob_tail && !maybe_full)

  when (finished_committing_row) {
    rob_head     := WrapInc(rob_head, numRobRows)
    rob_head_lsb := 0.U
    rob_deq      := true.B
  } .otherwise {
    rob_head_lsb := OHToUInt(PriorityEncoderOH(rob_head_vals.asUInt))
  }
```

## ROB Tail Logic

tail主要有以下优先级：

1. 当处于回滚状态，并且还没操作完或者ROB满了，此时自减tail，设置deq为true，
2. 当处于回滚，但tail等于head并且没有满，lsb设置为head的lsb

* [X] lsb意思就是bank的偏移

3. 当分支预测失败，自增
4. 当dispatch，自增，然后指向第0个bank
5. 当指令未派遣完，将LSB设置为最后一个有效指令的下一个bank

```
  val rob_enq = WireInit(false.B)

  when (rob_state === s_rollback && (rob_tail =/= rob_head || maybe_full)) {
    // Rollback a row
    rob_tail     := WrapDec(rob_tail, numRobRows)
    rob_tail_lsb := (coreWidth-1).U
    rob_deq := true.B
  } .elsewhen (rob_state === s_rollback && (rob_tail === rob_head) && !maybe_full) {
    // Rollback an entry
    rob_tail_lsb := rob_head_lsb
  } .elsewhen (io.brupdate.b2.mispredict) {
    rob_tail     := WrapInc(GetRowIdx(io.brupdate.b2.uop.rob_idx), numRobRows)
    rob_tail_lsb := 0.U
  } .elsewhen (io.enq_valids.asUInt =/= 0.U && !io.enq_partial_stall) {
    rob_tail     := WrapInc(rob_tail, numRobRows)
    rob_tail_lsb := 0.U
    rob_enq      := true.B
  } .elsewhen (io.enq_valids.asUInt =/= 0.U && io.enq_partial_stall) {
    rob_tail_lsb := PriorityEncoder(~MaskLower(io.enq_valids.asUInt))
  }
```

## ROB PNR逻辑

1. [ ] TODO

## ROB输出逻辑

提交是否有效，以及回滚是否有效

```
    io.commit.valids(w) := will_commit(w)
    io.commit.arch_valids(w) := will_commit(w) && !rob_predicated(com_idx)
    io.commit.uops(w)   := rob_uop(com_idx)
    io.commit.debug_insts(w) := rob_debug_inst_rdata(w)
...
    io.commit.rbk_valids(w) := rbk_row && rob_val(com_idx) && !(enableCommitMapTable.B)
    io.commit.rollback := (rob_state === s_rollback)
```

送往前端的flush信号

```
   // delay a cycle for critical path considerations
  io.flush.valid          := flush_val
  io.flush.bits.ftq_idx   := flush_uop.ftq_idx
  io.flush.bits.pc_lob    := flush_uop.pc_lob
  io.flush.bits.edge_inst := flush_uop.edge_inst
  io.flush.bits.is_rvc    := flush_uop.is_rvc
  io.flush.bits.flush_typ := FlushTypes.getType(flush_val,
                                                exception_thrown && !is_mini_exception,
                                                flush_commit && flush_uop.uopc === uopERET,
                                                refetch_inst)
```

输出异常信息，提交异常

```
  // Note: exception must be in the commit bundle.
  // Note: exception must be the first valid instruction in the commit bundle.
  exception_thrown := will_throw_exception
  val is_mini_exception = io.com_xcpt.bits.cause === MINI_EXCEPTION_MEM_ORDERING
  io.com_xcpt.valid := exception_thrown && !is_mini_exception
  io.com_xcpt.bits.cause := r_xcpt_uop.exc_cause
...
  io.com_xcpt.bits.badvaddr := Sext(r_xcpt_badvaddr, xLen)
...
  io.com_xcpt.bits.ftq_idx   := com_xcpt_uop.ftq_idx
  io.com_xcpt.bits.edge_inst := com_xcpt_uop.edge_inst
  io.com_xcpt.bits.is_rvc    := com_xcpt_uop.is_rvc
  io.com_xcpt.bits.pc_lob    := com_xcpt_uop.pc_lob
```

送往ren2/dispatch的信号

```
  io.empty        := empty
  io.ready        := (rob_state === s_normal) && !full && !r_xcpt_val
```

# BOOM V3 ISSUE 模块解析

## issue slot

![1731223168047](image/diplomacy/1731223168047.png)

首先明确：这个slot需要能写入东西，能读出东西，控制信号可以改变（唤醒）

写入就是dispatch模块写入，读出就是准备好了可以发射了

然後列出状态机:

```
trait IssueUnitConstants
{
  // invalid  : slot holds no valid uop.
  // s_valid_1: slot holds a valid uop.
  // s_valid_2: slot holds a store-like uop that may be broken into two micro-ops.
  val s_invalid :: s_valid_1 :: s_valid_2 :: Nil = Enum(3)
}
```

可以看到有三个状态

```
  val io = IO(new IssueSlotIO(numWakeupPorts))

  // slot invalid?
  // slot is valid, holding 1 uop
  // slot is valid, holds 2 uops (like a store)
  def is_invalid = state === s_invalid
  def is_valid = state =/= s_invalid

  val next_state      = Wire(UInt()) // the next state of this slot (which might then get moved to a new slot)
  val next_uopc       = Wire(UInt()) // the next uopc of this slot (which might then get moved to a new slot)
  val next_lrs1_rtype = Wire(UInt()) // the next reg type of this slot (which might then get moved to a new slot)
  val next_lrs2_rtype = Wire(UInt()) // the next reg type of this slot (which might then get moved to a new slot)

  val state = RegInit(s_invalid)
  val p1    = RegInit(false.B)
  val p2    = RegInit(false.B)
  val p3    = RegInit(false.B)
  val ppred = RegInit(false.B)

  // Poison if woken up by speculative load.
  // Poison lasts 1 cycle (as ldMiss will come on the next cycle).
  // SO if poisoned is true, set it to false!
  val p1_poisoned = RegInit(false.B)
  val p2_poisoned = RegInit(false.B)
  p1_poisoned := false.B
  p2_poisoned := false.B
  val next_p1_poisoned = Mux(io.in_uop.valid, io.in_uop.bits.iw_p1_poisoned, p1_poisoned)
  val next_p2_poisoned = Mux(io.in_uop.valid, io.in_uop.bits.iw_p2_poisoned, p2_poisoned)

  val slot_uop = RegInit(NullMicroOp)
  val next_uop = Mux(io.in_uop.valid, io.in_uop.bits, slot_uop)
```

接下来为主要信号，next_state這個slot的下一個狀態,之后这些next前缀的都是这个意思,他们是去构造压缩式队列使用的,然后state是这个slot的状态,p1,p2,p3表示操作数是否准备好了,ppred涉及到load的推测唤醒,但目前他们文档说不支持,下面的p1_poisoned表示推测唤醒失败,需要将这个p1给置为false,next_p1_poisoned是指输入的bit的p1是否被poisoned,slot_uop保存这个slot内容,然后next_uop,仍然用于压缩队列

```
  //-----------------------------------------------------------------------------
  // next slot state computation
  // compute the next state for THIS entry slot (in a collasping queue, the
  // current uop may get moved elsewhere, and a new uop can enter

  when (io.kill) {
    state := s_invalid
  } .elsewhen (io.in_uop.valid) {
    state := io.in_uop.bits.iw_state
  } .elsewhen (io.clear) {
    state := s_invalid
  } .otherwise {
    state := next_state
  }

```

然后就是下一个slot状态计算,kill表示冲刷流水线,clear表示slot被移到其他的地方了,如果输入的uop.valid有效,就把state置为输入uop的state,否则就为next_state

```
  //-----------------------------------------------------------------------------
  // "update" state
  // compute the next state for the micro-op in this slot. This micro-op may
  // be moved elsewhere, so the "next_state" travels with it.

  // defaults
  next_state := state
  next_uopc := slot_uop.uopc
  next_lrs1_rtype := slot_uop.lrs1_rtype
  next_lrs2_rtype := slot_uop.lrs2_rtype

  when (io.kill) {
    next_state := s_invalid
  } .elsewhen ((io.grant && (state === s_valid_1)) ||
    (io.grant && (state === s_valid_2) && p1 && p2 && ppred)) {
    // try to issue this uop.
    when (!(io.ldspec_miss && (p1_poisoned || p2_poisoned))) {
      next_state := s_invalid
    }
  } .elsewhen (io.grant && (state === s_valid_2)) {
    when (!(io.ldspec_miss && (p1_poisoned || p2_poisoned))) {
      next_state := s_valid_1
      when (p1) {
        slot_uop.uopc := uopSTD
        next_uopc := uopSTD
        slot_uop.lrs1_rtype := RT_X
        next_lrs1_rtype := RT_X
      } .otherwise {
        slot_uop.lrs2_rtype := RT_X
        next_lrs2_rtype := RT_X
      }
    }
  }

  when (io.in_uop.valid) {
    slot_uop := io.in_uop.bits
    assert (is_invalid || io.clear || io.kill, "trying to overwrite a valid issue slot.")
  }
```

当冲刷流水线,就把next_state设置为无效,当grant为高,可以并且状态为v1(s_valid_1),或者是v2,且操作数准备好了,就说明可以发射了,如果没有遇到load推测唤醒失败,就把next_state设置为s_invalid,假如state为v2并且grant,如果没发生load推测唤醒失败,就把next_state设置为v1,然后看准备好的是数据还是地址,分别被uopc赋值为相应类型,如果in_uop.valid,就把slot更新为io.in_uop.bits

```
  // Wakeup Compare Logic

  // these signals are the "next_p*" for the current slot's micro-op.
  // they are important for shifting the current slot_uop up to an other entry.
  val next_p1 = WireInit(p1)
  val next_p2 = WireInit(p2)
  val next_p3 = WireInit(p3)
  val next_ppred = WireInit(ppred)

  when (io.in_uop.valid) {
    p1 := !(io.in_uop.bits.prs1_busy)
    p2 := !(io.in_uop.bits.prs2_busy)
    p3 := !(io.in_uop.bits.prs3_busy)
    ppred := !(io.in_uop.bits.ppred_busy)
  }

  when (io.ldspec_miss && next_p1_poisoned) {
    assert(next_uop.prs1 =/= 0.U, "Poison bit can't be set for prs1=x0!")
    p1 := false.B
  }
  when (io.ldspec_miss && next_p2_poisoned) {
    assert(next_uop.prs2 =/= 0.U, "Poison bit can't be set for prs2=x0!")
    p2 := false.B
  }

  for (i <- 0 until numWakeupPorts) {
    when (io.wakeup_ports(i).valid &&
         (io.wakeup_ports(i).bits.pdst === next_uop.prs1)) {
      p1 := true.B
    }
    when (io.wakeup_ports(i).valid &&
         (io.wakeup_ports(i).bits.pdst === next_uop.prs2)) {
      p2 := true.B
    }
    when (io.wakeup_ports(i).valid &&
         (io.wakeup_ports(i).bits.pdst === next_uop.prs3)) {
      p3 := true.B
    }
  }
  when (io.pred_wakeup_port.valid && io.pred_wakeup_port.bits === next_uop.ppred) {
    ppred := true.B
  }

  for (w <- 0 until memWidth) {
    assert (!(io.spec_ld_wakeup(w).valid && io.spec_ld_wakeup(w).bits === 0.U),
      "Loads to x0 should never speculatively wakeup other instructions")
  }

  // TODO disable if FP IQ.
  for (w <- 0 until memWidth) {
    when (io.spec_ld_wakeup(w).valid &&
      io.spec_ld_wakeup(w).bits === next_uop.prs1 &&
      next_uop.lrs1_rtype === RT_FIX) {
      p1 := true.B
      p1_poisoned := true.B
      assert (!next_p1_poisoned)
    }
    when (io.spec_ld_wakeup(w).valid &&
      io.spec_ld_wakeup(w).bits === next_uop.prs2 &&
      next_uop.lrs2_rtype === RT_FIX) {
      p2 := true.B
      p2_poisoned := true.B
      assert (!next_p2_poisoned)
    }
  }
```

接下来是唤醒逻辑,首先定义了四个next前缀的信号,这些信号用于压缩队列,然后就是如果输入有效数据,检查输入的rs1,rs2,rs3是否busy,也就是是否被写入prf(在Busytable没表项),如果推测唤醒失败,就把p1置为false,其他同理,然后检查每个wakeupport,如果有port有效,并且pdst等于slot的src,就把该寄存器ready,然后是推测唤醒逻辑:

TODO

```
  // Request Logic
  io.request := is_valid && p1 && p2 && p3 && ppred && !io.kill
  val high_priority = slot_uop.is_br || slot_uop.is_jal || slot_uop.is_jalr
  io.request_hp := io.request && high_priority

  when (state === s_valid_1) {
    io.request := p1 && p2 && p3 && ppred && !io.kill
  } .elsewhen (state === s_valid_2) {
    io.request := (p1 || p2) && ppred && !io.kill
  } .otherwise {
    io.request := false.B
  }

```

接下来为req逻辑,只要p1,p2,p3准备好就可以req了,由于大部分指令为两个src,所以p3一般为默认值,也就是true,最后就是一些连线逻辑

## Issue Unit

```
/**
 * Abstract top level issue unit
 *
 * @param numIssueSlots depth of issue queue
 * @param issueWidth amoutn of operations that can be issued at once
 * @param numWakeupPorts number of wakeup ports for issue unit
 * @param iqType type of issue queue (mem, int, fp)
 */
abstract class IssueUnit(
  val numIssueSlots: Int,
  val issueWidth: Int,
  val numWakeupPorts: Int,
  val iqType: BigInt,
  val dispatchWidth: Int)
  (implicit p: Parameters)
  extends BoomModule
  with IssueUnitConstants
{
  val io = IO(new IssueUnitIO(issueWidth, numWakeupPorts, dispatchWidth))

  //-------------------------------------------------------------
  // Set up the dispatch uops
  // special case "storing" 2 uops within one issue slot.

  val dis_uops = Array.fill(dispatchWidth) {Wire(new MicroOp())}
  for (w <- 0 until dispatchWidth) {
    dis_uops(w) := io.dis_uops(w).bits
    dis_uops(w).iw_p1_poisoned := false.B
    dis_uops(w).iw_p2_poisoned := false.B
    dis_uops(w).iw_state := s_valid_1

    if (iqType == IQT_MEM.litValue || iqType == IQT_INT.litValue) {
      // For StoreAddrGen for Int, or AMOAddrGen, we go to addr gen state
      when ((io.dis_uops(w).bits.uopc === uopSTA && io.dis_uops(w).bits.lrs2_rtype === RT_FIX) ||
             io.dis_uops(w).bits.uopc === uopAMO_AG) {
        dis_uops(w).iw_state := s_valid_2
        // For store addr gen for FP, rs2 is the FP register, and we don't wait for that here
      } .elsewhen (io.dis_uops(w).bits.uopc === uopSTA && io.dis_uops(w).bits.lrs2_rtype =/= RT_FIX) {
        dis_uops(w).lrs2_rtype := RT_X
        dis_uops(w).prs2_busy  := false.B
      }
      dis_uops(w).prs3_busy := false.B
    } else if (iqType == IQT_FP.litValue) {
      // FP "StoreAddrGen" is really storeDataGen, and rs1 is the integer address register
      when (io.dis_uops(w).bits.uopc === uopSTA) {
        dis_uops(w).lrs1_rtype := RT_X
        dis_uops(w).prs1_busy  := false.B
      }
    }

    if (iqType != IQT_INT.litValue) {
      assert(!(io.dis_uops(w).bits.ppred_busy && io.dis_uops(w).valid))
      dis_uops(w).ppred_busy := false.B
    }
  }

  
```

我们这个抽象类,主要参数有issue queue大小,一次可以发射多少,唤醒port,issue的类型(mem,int,fp),然后创建了一个dis_uops,将来自dispatch的信号传入,然后将dip_uops初始化为dispatch数据,状态设置为v1(代表一般指令,),然后根据iq类型来分别进一步初始化,对于int类型的之后将prs3置为空闲,而mem不仅置为空闲,还检查是STA对state初始化为v2

```
  //-------------------------------------------------------------
  // Issue Table

  val slots = for (i <- 0 until numIssueSlots) yield { val slot = Module(new IssueSlot(numWakeupPorts)); slot }
  val issue_slots = VecInit(slots.map(_.io))

  for (i <- 0 until numIssueSlots) {
    issue_slots(i).wakeup_ports     := io.wakeup_ports
    issue_slots(i).pred_wakeup_port := io.pred_wakeup_port
    issue_slots(i).spec_ld_wakeup   := io.spec_ld_wakeup
    issue_slots(i).ldspec_miss      := io.ld_miss
    issue_slots(i).brupdate         := io.brupdate
    issue_slots(i).kill             := io.flush_pipeline
  }

  io.event_empty := !(issue_slots.map(s => s.valid).reduce(_|_))

  val count = PopCount(slots.map(_.io.valid))
  dontTouch(count)
```

接下来就是创建slot,连线,

## IssueUnitStatic

然后讲解非压缩队列

```
  val entry_wen_oh = VecInit(Seq.fill(numIssueSlots){ Wire(Bits(dispatchWidth.W)) })
  for (i <- 0 until numIssueSlots) {
    issue_slots(i).in_uop.valid := entry_wen_oh(i).orR
    issue_slots(i).in_uop.bits  := Mux1H(entry_wen_oh(i), dis_uops)
    issue_slots(i).clear        := false.B
  }
```

首先是表项写使能,这个entry_wen_oh会在后面赋值,这个是dispatch传来的,然后将数据传入issue slot,这里使用one hot 编码,这个会在之后讲解,将clear设置为false

```
  //-------------------------------------------------------------
  // Dispatch/Entry Logic
  // find a slot to enter a new dispatched instruction

  val entry_wen_oh_array = Array.fill(numIssueSlots,dispatchWidth){false.B}
  var allocated = VecInit(Seq.fill(dispatchWidth){false.B}) // did an instruction find an issue width?

  for (i <- 0 until numIssueSlots) {
    var next_allocated = Wire(Vec(dispatchWidth, Bool()))
    var can_allocate = !(issue_slots(i).valid)

    for (w <- 0 until dispatchWidth) {
      entry_wen_oh_array(i)(w) = can_allocate && !(allocated(w))

      next_allocated(w) := can_allocate | allocated(w)
      can_allocate = can_allocate && allocated(w)
    }

    allocated = next_allocated
  }

```

这是分发逻辑,首先创建一个entry_wen_oh_array,记录每个slot是否有dispatch的指令,然后allocated表示这个指令已经被分配了,然后进入两重循环,最底层循环就是看看这个slot是否空闲,如果空闲就将使能信号写入进去,然后把这个表项锁住,也就是将can_allocate置低,举例:

假设dispatch为4位使用一个四位变量allocate=(0,0,0,0)表示指令都没分发出去,假设指令0,找到了一个空slot,我们就可以把这个空槽占据了,然后next_allocate=(1,0,0,0)然后can_allocate由于allocated为false,所以置低,最后第一次循环完,next_allocate为(1,0,0,0),can_allocate=false,这个slot接受不到其他的指令了,已经被指令0占据了,内层循环完毕,把next_allocate赋值给allocate

```
  // if we can find an issue slot, do we actually need it?
  // also, translate from Scala data structures to Chisel Vecs
  for (i <- 0 until numIssueSlots) {
    val temp_uop_val = Wire(Vec(dispatchWidth, Bool()))

    for (w <- 0 until dispatchWidth) {
      // TODO add ctrl bit for "allocates iss_slot"
      temp_uop_val(w) := io.dis_uops(w).valid &&
                         !dis_uops(w).exception &&
                         !dis_uops(w).is_fence &&
                         !dis_uops(w).is_fencei &&
                         entry_wen_oh_array(i)(w)
    }
    entry_wen_oh(i) := temp_uop_val.asUInt
  }

  for (w <- 0 until dispatchWidth) {
    io.dis_uops(w).ready := allocated(w)
  }

```

这段代码将上面得出的wen信号进一步处理,然后将wen赋值给一开始的entry_wen_oh,这样最上面的代码就可以找到哪个slot这次会被写入了,并且这个也得出了是那一条指令占据了哪个slot,假设有4个slot,dis大小也是4,最后这个entry_wen_oh可能是(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1),也就是得到了每条指令要写入哪个slot的信息,完成分配的信号就是allocate对应位为1,

```
  for (w <- 0 until issueWidth) {
    io.iss_valids(w) := false.B
    io.iss_uops(w)   := NullMicroOp
    // unsure if this is overkill
    io.iss_uops(w).prs1 := 0.U
    io.iss_uops(w).prs2 := 0.U
    io.iss_uops(w).prs3 := 0.U
    io.iss_uops(w).lrs1_rtype := RT_X
    io.iss_uops(w).lrs2_rtype := RT_X
  }
```

接下来为仲裁逻辑,首先对issue信号初始化

```
// TODO can we use flatten to get an array of bools on issue_slot(*).request?
  val lo_request_not_satisfied = Array.fill(numIssueSlots){Bool()}
  val hi_request_not_satisfied = Array.fill(numIssueSlots){Bool()}

  for (i <- 0 until numIssueSlots) {
    lo_request_not_satisfied(i) = issue_slots(i).request
    hi_request_not_satisfied(i) = issue_slots(i).request_hp
    issue_slots(i).grant := false.B // default
  }

  for (w <- 0 until issueWidth) {
    var port_issued = false.B

    // first look for high priority requests
    for (i <- 0 until numIssueSlots) {
      val can_allocate = (issue_slots(i).uop.fu_code & io.fu_types(w)) =/= 0.U

      when (hi_request_not_satisfied(i) && can_allocate && !port_issued) {
        issue_slots(i).grant := true.B
        io.iss_valids(w)     := true.B
        io.iss_uops(w)       := issue_slots(i).uop
      }

      val port_already_in_use     = port_issued
      port_issued                 = (hi_request_not_satisfied(i) && can_allocate) | port_issued
      // deassert lo_request if hi_request is 1.
      lo_request_not_satisfied(i) = (lo_request_not_satisfied(i) && !hi_request_not_satisfied(i))
      // if request is 0, stay 0. only stay 1 if request is true and can't allocate
      hi_request_not_satisfied(i) = (hi_request_not_satisfied(i) && (!can_allocate || port_already_in_use))
    }

    // now look for low priority requests
    for (i <- 0 until numIssueSlots) {
      val can_allocate = (issue_slots(i).uop.fu_code & io.fu_types(w)) =/= 0.U

      when (lo_request_not_satisfied(i) && can_allocate && !port_issued) {
        issue_slots(i).grant := true.B
        io.iss_valids(w)     := true.B
        io.iss_uops(w)       := issue_slots(i).uop
      }

      val port_already_in_use     = port_issued
      port_issued                 = (lo_request_not_satisfied(i) && can_allocate) | port_issued
      // if request is 0, stay 0. only stay 1 if request is true and can't allocate or port already in use
      lo_request_not_satisfied(i) = (lo_request_not_satisfied(i) && (!can_allocate || port_already_in_use))
    }
  }
```

首先把低级req和高级req从issue slot读出来,将grant置为低(初始化),然后进入仲裁逻辑,首先检查高优先级的req,首先有一个can_allocate信号,也就是匹配FU,如果匹配到FU,并且有高优先级请求,并且port_issue没有置为高,就发出grant信号,表示可以发射了,将slot的uop读出来,然后将这个port_issued置为高,接下来重新赋值低位请求,必须没有高位请求,低位请求才生效,如果有高级请求,但FU没匹配成功或者这个FU在用,就一直置为高位请求,接下来就是低级请求,其和高级请求的思路类似

## IssueUnitCollapsing

```
  //-------------------------------------------------------------
  // Figure out how much to shift entries by

  val maxShift = dispatchWidth
  val vacants = issue_slots.map(s => !(s.valid)) ++ io.dis_uops.map(_.valid).map(!_.asBool)
  val shamts_oh = Array.fill(numIssueSlots+dispatchWidth) {Wire(UInt(width=maxShift.W))}
  // track how many to shift up this entry by by counting previous vacant spots
  def SaturatingCounterOH(count_oh:UInt, inc: Bool, max: Int): UInt = {
     val next = Wire(UInt(width=max.W))
     next := count_oh
     when (count_oh === 0.U && inc) {
       next := 1.U
     } .elsewhen (!count_oh(max-1) && inc) {
       next := (count_oh << 1.U)
     }
     next
  }
  shamts_oh(0) := 0.U
  for (i <- 1 until numIssueSlots + dispatchWidth) {
    shamts_oh(i) := SaturatingCounterOH(shamts_oh(i-1), vacants(i-1), maxShift)
  }
```

首先定义最大位移的数字maxshift,然后vacants就是把issue slot和要写入的看看是不是有效的,之后讲解SaturatingCounterOH方法,这个方法定义了每个位置要位移多少,首先最底部的绝对不用位移,之后的位置位移取决于下面的是否是空的,如果是空的,就在下面的一个位置位移的基础上左移一位(one hot编码),如果不是one hot,只要在下面位置位移的基础+1即可,然后我们经过这个循环就得到了每一项要位移的数(one hot),

> 不太明白这个maxshift为什么要以dispatchwidth为最大值,不该为issuewidth吗

```
  //-------------------------------------------------------------

  // which entries' uops will still be next cycle? (not being issued and vacated)
  val will_be_valid = (0 until numIssueSlots).map(i => issue_slots(i).will_be_valid) ++
                      (0 until dispatchWidth).map(i => io.dis_uops(i).valid &&
                                                        !dis_uops(i).exception &&
                                                        !dis_uops(i).is_fence &&
                                                        !dis_uops(i).is_fencei)

  val uops = issue_slots.map(s=>s.out_uop) ++ dis_uops.map(s=>s)
  for (i <- 0 until numIssueSlots) {
    issue_slots(i).in_uop.valid := false.B
    issue_slots(i).in_uop.bits  := uops(i+1)
    for (j <- 1 to maxShift by 1) {
      when (shamts_oh(i+j) === (1 << (j-1)).U) {
        issue_slots(i).in_uop.valid := will_be_valid(i+j)
        issue_slots(i).in_uop.bits  := uops(i+j)
      }
    }
    issue_slots(i).clear        := shamts_oh(i) =/= 0.U
  }

```

这几段代码主要讲的就是issue和dispatch的表项是否在下个周期还有效,也就是他是否发射出去了或者被清除了,然后循环内主要就是对slot移位,就是设置一个小循环,这个小循环检测是哪个移位进来的,

举例:

假设我们有四个slot,然后slot(0)是空的,其他都有数据,那么shamt(0)=0,shamt(1)=01,shamt(2)=01,shamt(3)=01,所以我们移位后就是3->2,2->1,1->0,假设i=0,小循环第一次进入when,此时j=1,这就完成了1->0的操作,由于slot(1)不是空的,所以这个循环只会进入一次when,最后出小循环将slot(0)的clear根据shamt(0)置为false

> 最后一步的clear对移位后有数据的没什莫影响,因为in_valid优先级大于clear,但对高位置的slot有影响,比如这里就是对3有影响(假设没有指令dispatch进来)

```
  //-------------------------------------------------------------
  // Dispatch/Entry Logic
  // did we find a spot to slide the new dispatched uops into?

  val will_be_available = (0 until numIssueSlots).map(i =>
                            (!issue_slots(i).will_be_valid || issue_slots(i).clear) && !(issue_slots(i).in_uop.valid))
  val num_available = PopCount(will_be_available)
  for (w <- 0 until dispatchWidth) {
    io.dis_uops(w).ready := RegNext(num_available > w.U)
  }
```

这段代码就是检测dispatch的指令是否写进来,will_be_available检查空的slot并且之后还被移入数据,然后num_available得到空slot的数目,如果num_available大于dispatchwidth,就说明分发好了,这里也就是空的slot大于分发的数目,注意,这里不保证每个都写进去,

```

  //-------------------------------------------------------------
  // Issue Select Logic

  // set default
  for (w <- 0 until issueWidth) {
    io.iss_valids(w) := false.B
    io.iss_uops(w)   := NullMicroOp
    // unsure if this is overkill
    io.iss_uops(w).prs1 := 0.U
    io.iss_uops(w).prs2 := 0.U
    io.iss_uops(w).prs3 := 0.U
    io.iss_uops(w).lrs1_rtype := RT_X
    io.iss_uops(w).lrs2_rtype := RT_X
  }

  val requests = issue_slots.map(s => s.request)
  val port_issued = Array.fill(issueWidth){Bool()}
  for (w <- 0 until issueWidth) {
    port_issued(w) = false.B
  }

  for (i <- 0 until numIssueSlots) {
    issue_slots(i).grant := false.B
    var uop_issued = false.B

    for (w <- 0 until issueWidth) {
      val can_allocate = (issue_slots(i).uop.fu_code & io.fu_types(w)) =/= 0.U

      when (requests(i) && !uop_issued && can_allocate && !port_issued(w)) {
        issue_slots(i).grant := true.B
        io.iss_valids(w) := true.B
        io.iss_uops(w) := issue_slots(i).uop
      }
      val was_port_issued_yet = port_issued(w)
      port_issued(w) = (requests(i) && !uop_issued && can_allocate) | port_issued(w)
      uop_issued = (requests(i) && can_allocate && !was_port_issued_yet) | uop_issued
    }
  }
```

最后是仲裁逻辑,首先将issue信息初始化,然后找slot的req,之后去寻找可以issue的项,这里和非压缩类似,

## 总结

无论是压缩还是非压缩,issue都使用相同的slot,而且仲裁逻辑都是一样的,也就是从低slot扫描到高slot,直到凑齐发射指令

# Boom regfile

## Regfile模块

### 读逻辑

首先检查是否有bypass数据,如果有的话,就选择读出的数据是bypass数据,注意这里选择bypass数据时是选择最新写入这个寄存器的值,,也就是采用Mux1H,得到bypass数据,注意这里提交都是等一个ROB行算完才可以提交并bypass,如果无bypass数据,就直接读出regfile的数

> 这里的bypass是指W->R bypass

```
  if (bypassableArray.reduce(_||_)) {
    val bypassable_wports = ArrayBuffer[Valid[RegisterFileWritePort]]()
    io.write_ports zip bypassableArray map { case (wport, b) => if (b) { bypassable_wports += wport} }

    for (i <- 0 until numReadPorts) {
      val bypass_ens = bypassable_wports.map(x => x.valid &&
        x.bits.addr === read_addrs(i))
      //使用Mux1H得出最新的指令的bypass的结果
      val bypass_data = Mux1H(VecInit(bypass_ens.toSeq), VecInit(bypassable_wports.map(_.bits.data).toSeq))

      io.read_ports(i).data := Mux(bypass_ens.reduce(_|_), bypass_data, read_data(i))
    }
  } else {
    for (i <- 0 until numReadPorts) {
      io.read_ports(i).data := read_data(i)
    }
  }
```

### 写逻辑

代码如下.

```
  for (wport <- io.write_ports) {
    when (wport.valid) {
      regfile(wport.bits.addr) := wport.bits.data
    }
  }
```

## RegisterRead模块

### 读端口逻辑

首先读出issue模块送入的rs的addr，将其送入rf模块，然后根据addr读出相应数据，主要这里读寄存器在issue，读出寄存器在RF阶段，然后exe_reg_uops是送往exe阶段的uops，这里的idx的意思就是充分利用每个端口，端口不与指令绑定，比如我有两条指令，一个需要2个读，一个需要1个写，所以我的读idx在循环内为（0，2）

* [ ] 暂时不知道为什么延迟的原因

```
  var idx = 0 // index into flattened read_ports array
  for (w <- 0 until issueWidth) {
    val numReadPorts = numReadPortsArray(w)

    // NOTE:
    // rrdLatency==1, we need to send read address at end of ISS stage,
    //    in order to get read data back at end of RRD stage.

    val rs1_addr = io.iss_uops(w).prs1
    val rs2_addr = io.iss_uops(w).prs2
    val rs3_addr = io.iss_uops(w).prs3
    val pred_addr = io.iss_uops(w).ppred

    if (numReadPorts > 0) io.rf_read_ports(idx+0).addr := rs1_addr
    if (numReadPorts > 1) io.rf_read_ports(idx+1).addr := rs2_addr
    if (numReadPorts > 2) io.rf_read_ports(idx+2).addr := rs3_addr

    if (enableSFBOpt) io.prf_read_ports(w).addr := pred_addr

    if (numReadPorts > 0) rrd_rs1_data(w) := Mux(RegNext(rs1_addr === 0.U), 0.U, io.rf_read_ports(idx+0).data)
    if (numReadPorts > 1) rrd_rs2_data(w) := Mux(RegNext(rs2_addr === 0.U), 0.U, io.rf_read_ports(idx+1).data)
    if (numReadPorts > 2) rrd_rs3_data(w) := Mux(RegNext(rs3_addr === 0.U), 0.U, io.rf_read_ports(idx+2).data)

    if (enableSFBOpt) rrd_pred_data(w) := Mux(RegNext(io.iss_uops(w).is_sfb_shadow), io.prf_read_ports(w).data, false.B)

    val rrd_kill = io.kill || IsKilledByBranch(io.brupdate, rrd_uops(w))

    exe_reg_valids(w) := Mux(rrd_kill, false.B, rrd_valids(w))
    // TODO use only the valids signal, don't require us to set nullUop
    exe_reg_uops(w)   := Mux(rrd_kill, NullMicroOp, rrd_uops(w))

    exe_reg_uops(w).br_mask := GetNewBrMask(io.brupdate, rrd_uops(w))

    idx += numReadPorts
  }
```

### BYPASS逻辑

bypass不bypass寄存器rs3（FU），也就是只bypass INT，其中rs1_cases，rs2_cases得出了mux控制信号和data，然后MUXcase的意思就是默认为rrd_rs1_data，如果之后的条件满足，就选择之后的值

```
  for (w <- 0 until issueWidth) {
    val numReadPorts = numReadPortsArray(w)
    var rs1_cases = Array((false.B, 0.U(registerWidth.W)))
    var rs2_cases = Array((false.B, 0.U(registerWidth.W)))
    var pred_cases = Array((false.B, 0.U(1.W)))

    val prs1       = rrd_uops(w).prs1
    val lrs1_rtype = rrd_uops(w).lrs1_rtype
    val prs2       = rrd_uops(w).prs2
    val lrs2_rtype = rrd_uops(w).lrs2_rtype
    val ppred      = rrd_uops(w).ppred

    for (b <- 0 until numTotalBypassPorts)
    {
      val bypass = io.bypass(b)
      // can't use "io.bypass.valid(b) since it would create a combinational loop on branch kills"
      rs1_cases ++= Array((bypass.valid && (prs1 === bypass.bits.uop.pdst) && bypass.bits.uop.rf_wen
        && bypass.bits.uop.dst_rtype === RT_FIX && lrs1_rtype === RT_FIX && (prs1 =/= 0.U), bypass.bits.data))
      rs2_cases ++= Array((bypass.valid && (prs2 === bypass.bits.uop.pdst) && bypass.bits.uop.rf_wen
        && bypass.bits.uop.dst_rtype === RT_FIX && lrs2_rtype === RT_FIX && (prs2 =/= 0.U), bypass.bits.data))
    }

    for (b <- 0 until numTotalPredBypassPorts)
    {
      val bypass = io.pred_bypass(b)
      pred_cases ++= Array((bypass.valid && (ppred === bypass.bits.uop.pdst) && bypass.bits.uop.is_sfb_br, bypass.bits.data))
    }

    if (numReadPorts > 0) bypassed_rs1_data(w)  := MuxCase(rrd_rs1_data(w), rs1_cases)
    if (numReadPorts > 1) bypassed_rs2_data(w)  := MuxCase(rrd_rs2_data(w), rs2_cases)
    if (enableSFBOpt)     bypassed_pred_data(w) := MuxCase(rrd_pred_data(w), pred_cases)
  }
```

### 送往执行阶段信号

代码如下，主要送了valid，数据和uops，注意这里是有pipe reg的

```
  // set outputs to execute pipelines
  for (w <- 0 until issueWidth) {
    val numReadPorts = numReadPortsArray(w)

    io.exe_reqs(w).valid    := exe_reg_valids(w)
    io.exe_reqs(w).bits.uop := exe_reg_uops(w)
    if (numReadPorts > 0) io.exe_reqs(w).bits.rs1_data := exe_reg_rs1_data(w)
    if (numReadPorts > 1) io.exe_reqs(w).bits.rs2_data := exe_reg_rs2_data(w)
    if (numReadPorts > 2) io.exe_reqs(w).bits.rs3_data := exe_reg_rs3_data(w)
    if (enableSFBOpt)     io.exe_reqs(w).bits.pred_data := exe_reg_pred_data(w)
  }
```

### 总结

regfile和regfile_read均有bypass逻辑，但前者只bypassW->R ,后者bypass所有有效的FU的数据（不包括FPU）

# BOOM EXU

![1731814497133](image/diplomacy&boom/1731814497133.png)

BOOM是非数据捕捉模式，可以看到alu模块插入了寄存器，这里是为了和mul与FPU匹配，简化写入端口调度

## 执行单元

![1731843369344](image/diplomacy&boom/1731843369344.png)

这个例子是一个INT ALU，和乘法器，issue每个issue端口，只与一个FU对话，执行单元就是一个抽象单元，其包装的rocketchip的功能单元

### PipelinedFunctionalUnit模块

这是流水线功能单元的抽象类,主要补充下面ALU的模块

#### Response 信号

这里分了两种情况

1. pipestage>0:这时候,输出有效信号就是r_valid的最高索引,r_valid每个周期都检测是否有kill信号,以及分支预测失败,

```
    io.resp.valid    := r_valids(numStages-1) && !IsKilledByBranch(io.brupdate, r_uops(numStages-1))
    io.resp.bits.predicated := false.B
    io.resp.bits.uop := r_uops(numStages-1)
    io.resp.bits.uop.br_mask := GetNewBrMask(io.brupdate, r_uops(numStages-1))
```

2. pipestage==0,这时候,输出有效信号直接是输入的有效信号并且不能在失败路径上,

```
    io.resp.valid    := io.req.valid && !IsKilledByBranch(io.brupdate, io.req.bits.uop)
    io.resp.bits.predicated := false.B
    io.resp.bits.uop := io.req.bits.uop
    io.resp.bits.uop.br_mask := GetNewBrMask(io.brupdate, io.req.bits.uop)
```

#### bypass 信号

只有stage>0才有bypass,如果earliestBypassStage为0(表示第一个周期就可以bypass),那么第一个bypass的uops就是输入的uops,之后的的bypass_uops就是相对应的r_uops,

> 注:这里bypass为i,但r_uops为i-1,主要就是r_uops为流水线寄存器,在下一个周期才可以获得数据

* [ ] 暂时不知道第一句是干什莫的,似乎在earliestBypassStage不为0才有用,但目前都是为0的情况

```
      if (numBypassStages > 0) {
        io.bypass(i-1).bits.uop := r_uops(i-1)
      }
...
    if (numBypassStages > 0 && earliestBypassStage == 0) {
      io.bypass(0).bits.uop := io.req.bits.uop

      for (i <- 1 until numBypassStages) {
        io.bypass(i).bits.uop := r_uops(i-1)
      }
    }
```

### ALU模块

alu逻辑包含BR分支计算，以及正常指令计算

#### 数据选择

* op1的数据来源有两个地方，PC以及读出的rs1
* op2的数据来源有四个来源，IMM，IMM_C（仅限于CSR指令），RS2,NEXT（也即是下一个pc的位移，2or4）

#### 分支处理

* BR_N：也就是PC+4
* BR_NE:不相等
* BR_EQ：相等
* 。。。
* BR_J:JUMP（jal）
* BR_JR:JUMP REG（jalr）
* PC_PLUS4：pc+4
* PC_BRJMP：BR 目标地址
* PC_BRJMP：jalr目标地址

这里是检查送入的指令的类型是什么分支类型，根据控制信号该选什么样的target

is_taken的意思是这个分支是否跳转，假如输入有效，没有在错误路径，是分支指令并且PC不为pc+4，就进行跳转

```
  val pc_sel = MuxLookup(uop.ctrl.br_type, PC_PLUS4,
                 Seq(   BR_N   -> PC_PLUS4,
                        BR_NE  -> Mux(!br_eq,  PC_BRJMP, PC_PLUS4),
                        BR_EQ  -> Mux( br_eq,  PC_BRJMP, PC_PLUS4),
                        BR_GE  -> Mux(!br_lt,  PC_BRJMP, PC_PLUS4),
                        BR_GEU -> Mux(!br_ltu, PC_BRJMP, PC_PLUS4),
                        BR_LT  -> Mux( br_lt,  PC_BRJMP, PC_PLUS4),
                        BR_LTU -> Mux( br_ltu, PC_BRJMP, PC_PLUS4),
                        BR_J   -> PC_BRJMP,
                        BR_JR  -> PC_JALR
                        ))
  val is_taken = io.req.valid &&
                   !killed &&
                   (uop.is_br || uop.is_jalr || uop.is_jal) &&
                   (pc_sel =/= PC_PLUS4)
```

#### 分支地址计算

主要就是計算jalr的target,然后得出cfi_idx,访问前端FTQ,获得pc,next_val意思是下一条指令是否有效

jalr指令的误预测逻辑:

* 下一条指令无效
* 下一条指令有效但pc不是实际计算的pc
* 没有被预测跳转,(在cfi找不到或者找到了但是无效预测)

br指令的分支预测目标地址为target,供重定向使用

```
    brinfo.jalr_target := jalr_target
    val cfi_idx = ((uop.pc_lob ^ Mux(io.get_ftq_pc.entry.start_bank === 1.U, 1.U << log2Ceil(bankBytes), 0.U)))(log2Ceil(fetchWidth),1)

    when (pc_sel === PC_JALR) {
      mispredict := !io.get_ftq_pc.next_val ||
                    (io.get_ftq_pc.next_pc =/= jalr_target) ||
                    !io.get_ftq_pc.entry.cfi_idx.valid ||
                    (io.get_ftq_pc.entry.cfi_idx.bits =/= cfi_idx)
    }
brinfo.target_offset := target_offset
```

#### 分支预测失败检测

首先，jal不参与检查，因为jal是必然跳转，且地址固定，jalr和br进行地址检测

如果pc_sel为PC_PLUS4，说明实际为不跳转，如果之前为taken，就说明地址预测失败

如果pc_sel为PC_BRJMP,说明实际跳转，如果之前预测taken，则地址预测成功

```
  when (is_br || is_jalr) {
    if (!isJmpUnit) {
      assert (pc_sel =/= PC_JALR)
    }
    when (pc_sel === PC_PLUS4) {
      mispredict := uop.taken
    }
    when (pc_sel === PC_BRJMP) {
      mispredict := !uop.taken
    }
  }
```

#### Response逻辑

ALU out有以下来源:

* 如果是is_sfb_shadow,并且pred_data,如果是ldst_rs1需要rs1,则把rs1当作结果,否則就是rs2(这个和BOOM的SFB有关)
* 如果为MOV指令,就选择rs2为输出,否则就是选择alu计算的结果

然后就是流水线逻辑,在s1将数据送入流水线,时候根据numstage选择流水级,最后输出的数据就是r_data的最高索引

```
  r_val (0) := io.req.valid
  r_data(0) := Mux(io.req.bits.uop.is_sfb_br, pc_sel === PC_BRJMP, alu_out)
  r_pred(0) := io.req.bits.uop.is_sfb_shadow && io.req.bits.pred_data
  for (i <- 1 until numStages) {
    r_val(i)  := r_val(i-1)
    r_data(i) := r_data(i-1)
    r_pred(i) := r_pred(i-1)
  }
  io.resp.bits.data := r_data(numStages-1)
  io.resp.bits.predicated := r_pred(numStages-1)
```

#### Bypass逻辑

将各阶段的输出进行bypass,注意这里是有延迟一个周期的,也就是计算出来下个周期再bypass,

```
  io.bypass(0).valid := io.req.valid
  io.bypass(0).bits.data := Mux(io.req.bits.uop.is_sfb_br, pc_sel === PC_BRJMP, alu_out)
  for (i <- 1 until numStages) {
    io.bypass(i).valid := r_val(i-1)
    io.bypass(i).bits.data := r_data(i-1)
  }
```

### 其他模块

* MemAddrCalcUnit:完成地址计算以及store_data接受,同时进行misalign检查
* DIV模块,是unpipe的,调用rocket模块
* MUL模块,调用了rocket的模块

## ALUExeUnit

这个模块是各种单独FU的集合,目前允许,ALU和MUL和DIV在一块,但MEM只能单独一个ALUExeUnit,

ALU Unit:这个模块包含BRU,他接受输入信号,然后只有ALU支持bypass

### 输出逻辑

输出信号主要有有效信号,数据,以及uops等,根据数据有效信号来得出数据

```
    io.iresp.valid     := iresp_fu_units.map(_.io.resp.valid).reduce(_|_)
    io.iresp.bits.uop  := PriorityMux(iresp_fu_units.map(f =>
      (f.io.resp.valid, f.io.resp.bits.uop)).toSeq)
    io.iresp.bits.data := PriorityMux(iresp_fu_units.map(f =>
      (f.io.resp.valid, f.io.resp.bits.data)).toSeq)
    io.iresp.bits.predicated := PriorityMux(iresp_fu_units.map(f =>
      (f.io.resp.valid, f.io.resp.bits.predicated)).toSeq)
```

## ExecutionUnits

就是简单的连线模块

> 为什么执行完直接写入寄存器，不会改变ARCH state吗?虽然写入寄存器，但仍然属于推测状态，这时，如果之前指令发生异常情况，这个指令的计算结果无效，从发生异常的指令重新执行，假如：r0之前的映射关系为（r0：p21），由于这里是统一PRF，只有改变了映射关系（提交阶段），状态才算改变，也就是虽然向p30写入数据了，但r0的映射关系目前还是p21，只有正确提交，r0的映射关系才会变为p30，如果下面指令之前有分支预测失败，假设我要是读取寄存器r0，那么还是p21的值，也就是最近正确写入的值

```
add r0，r1，r2
add r0，r3，r0
重命名后
add p30，p11,p12
add p31,p13,p30
```

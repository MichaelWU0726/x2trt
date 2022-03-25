# 可视化TensorRT的engine

众所周知，TensorRT会对模型做很多的优化，比如前后层融合（CONV+BN+RELU）、比如水平层融合、又比如去掉concat直接操作等等，通过TensorRT优化后的模型，基本已经“面目全非”了，TensorRT支持很多层的融合，你的模型扔给TensorRT再出来，会发现很多层都被**合体**了。当然这样做的目的是为了优化访存，减少数据在每层之间传输的消耗。不过，这样做并不都没毛病，有时候会有奇奇怪怪的[BUG](https://mp.weixin.qq.com/s?__biz=Mzg3ODU2MzY5MA==&mid=2247487388&idx=1&sn=915ef017a4de77dd545f788bfcd4d86d&chksm=cf109799f8671e8ff68dd5ac34929982d99a9d4a75638ac23ba77741f2f2d0d42c019de9b36e&token=2127318739&lang=zh_CN&scene=21#wechat_redirect)，我们需要注意。

被合体之后的模型，我们一般无法通过`Netron`来读取查看，毕竟TensorRT是闭源的，其生成的engine结构之复杂，只靠猜是不行的。不过TensorRT知道其闭源的缺点，为我们引入了log接口，如果我们想看到融合后的模型长什么样，只要在`build engine`开启verbose模式即可，但生成的文本内容只是一层层罗列没有规律，不画图根本看不出来什么。

参考[jerryzh168开源的Facebook内部查看engine的工具](https://github.com/pytorch/pytorch/pull/66431/files)，使用[pydot](https://pypi.org/project/pydot)和[graphviz](https://pypi.org/project/graphviz)来画网络结构图

# 参考

[终于把TensorRT的engine模型结构图画出来了](https://mp.weixin.qq.com/s?__biz=Mzg3ODU2MzY5MA==&mid=2247488759&idx=1&sn=254c8c288bf3b87b80c47593d6e3b740&chksm=cf108cf2f86705e44616970ac9f4d2644c12f8063492906d9f0827913c6b1e71581f139f0322&scene=0&subscene=92&sessionid=1648191452&clicktime=1648191473&enterid=1648191473&ascene=7&devicetype=android-29&version=2800133d&nettype=3gnet&abtest_cookie=AAACAA%3D%3D&lang=zh_CN&exportkey=ASqF2zPykIQsQ6ujfgBH%2FOU%3D&pass_ticket=d2IH4I%2B3URuXaZlZJ8KMC3EyfiV2tra5nFyv1ArVcIAZFwAi132X5el2zL6sKl8V&wx_header=3)


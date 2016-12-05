---
layout: post
comments: true
categories: DeepLearning
---
## Autoencoders
Autoencoder也是一种神经网络，它的目的是将input“复制”到output，从内部结构来看，它有一个隐含层h用来表示input的压缩编码。autoencoder由两个部分组成：一个编码函数 $$h=f(x)$$ 以及一个用来解码的解码函数 $$r=g(h)$$. 如果autoencoder能够完全的解码，即对所有的x，都有 $$g(f(x))=x$$, 这样autoencoder是没有什么特殊的作用的。相反，我们并不需要完全的解码能力，大多数情况下，只允许他们完成近似的拷贝。 因为这个模型需要考虑哪些input是需要拷贝，只从input中拷贝有用的数据。
现在的autoencoder的设计灵感是从以前的编码和解码（将$$p_encoder(h|x)$$映射到p_decoder(x|h)）的过程产生的。

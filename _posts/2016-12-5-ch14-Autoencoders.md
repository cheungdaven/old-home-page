---
layout: post
comments: true
categories: DeepLearning
---
## Autoencoders
<p>Autoencoder也是一种神经网络，它的目的是将input“复制”到output，从内部结构来看，它有一个隐含层\(h\)用来表示input的压缩编码。autoencoder由两个部分组成：一个编码函数 \(h=f(x)\) 以及一个用来解码的解码函数 \(r=g(h)\). 如果autoencoder能够完全的解码，即对所有的\(x\)，都有 \(g(f(x))=x\), 这样autoencoder是没有什么特殊的作用的。相反，我们并不需要完全的解码能力，大多数情况下，只允许他们完成近似的拷贝。 因为这个模型需要考虑哪些input是需要拷贝，只从input中拷贝有用的数据。
<p>现在的autoencoder的设计灵感是从以前的编码和解码（将\(p_{encoder}(h|x)\)映射到\(p_{decoder}(x|h)\)）的过程产生的。
<p>Autoencoder的概念在1987年就已经出现在了神经网络研究当中，传统的autoencoder主要用来降维和特征学习，近年来，autoencoder和隐含变量模型的理论联系将autoencoder带入了生产模型研究的前沿，在接下来的第20章中，autoencoder可以被看成feedforward网络的一种特例，甚至用相同的方法进行训练（即minibatch梯度下降，gradients使用back-propagation计算）。但和普通的feedforward网络不同，autoencoder可以使用recirculation（比较原始输入的激活函数和输出的激活函数）来进行训练，recirculation相对back-propagation来说，更加的合理，但是很是被使用在机器学习领域。
## Undercomplete Autoencoders
<p>将输入复制到输出，听起来似乎什么用的都没有，但是通常情况下我们并不关注输入。相反，我们关注的是隐含层\(h\)，我们希望隐含层能够包含很多有用的属性。
<p>如果\(h\)曾的维数小于\(x\)，autoencoder就可以用来进行特征提取，我们将这种情况称为undercomplete。在进行undercomplet表征学习时迫使autoencoder去捕捉训练数据中的最精华的信息。
<p>简单地，学习过程可以表示为最小化如下的损失函数：
\(L(x,g(f(x)))\)
 
  

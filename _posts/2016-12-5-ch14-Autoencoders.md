---
layout: post
comments: true
categories: DeepLearning
---

* content
{:toc}

## Autoencoders 

<p> Autoencoder也是一种神经网络，它的目的是将input“复制”到output，从内部结构来看，它有一个隐含层\(h\)用来表示input的压缩编码。autoencoder由两个部分组成：一个编码函数 \(h=f(x)\) 以及一个用来解码的解码函数 \(r=g(h)\). 如果autoencoder能够完全的解码，即对所有的\(x\)，都有 \(g(f(x))=x\), 这样autoencoder是没有什么特殊的作用的。相反，我们并不需要完全的解码能力，大多数情况下，只允许他们完成近似的拷贝。 因为这个模型需要考虑哪些input是需要拷贝，只从input中拷贝有用的数据。    </p>
<p>现在的autoencoder的设计灵感是从以前的编码和解码（将\(p_{encoder}(h|x)\)映射到\\(p_{decoder}(x|h)\)）的过程产生的。     </p>
<p>Autoencoder的概念在1987年就已经出现在了神经网络研究当中，传统的autoencoder主要用来降维和特征学习，近年来，autoencoder和隐含变量模型的理论联系将autoencoder带入了生产模型研究的前沿，在接下来的第20章中，autoencoder可以被看成feedforward网络的一种特例，甚至用相同的方法进行训练（即minibatch梯度下降，gradients使用back-propagation计算）。但和普通的feedforward网络不同，autoencoder可以使用recirculation（比较原始输入的激活函数和输出的激活函数）来进行训练，recirculation相对back-propagation来说，更加的合理，但是很是被使用在机器学习领域。  </p>

### Undercomplete Autoencoders
<p>将输入复制到输出，听起来似乎什么用的都没有，但是通常情况下我们并不关注输入。相反，我们关注的是隐含层\(h\)，我们希望隐含层能够包含很多有用的属性。 </p>
<p>如果\(h\)曾的维数小于\(x\)，autoencoder就可以用来进行特征提取，我们将这种情况称为undercomplete。在进行undercomplete表征学习时迫使autoencoder去捕捉训练数据中的最精华的信息。  </p>
<p>简单地，学习过程可以表示为最小化如下的损失函数：  </p>
$$L(x,g(f(x)))$$  
<p>其中L为损失函数，用来衡量\(x\)和\(g(f(x))\)的差异，类似于平方差。  </p>
<p>当decoder是线性函数的时候，L就是平方差,autoencoder和PCA一样得跨越相同的子空间。当\(f\)和\(g\)都是非线性函数的时候，Autoencoder有比PCA更加强大的非线性生成能力。不幸的是，如果encoderh和decoder太过强大，以至于都不需要autoencoder进行特征提取就能恢复输入数据，这样的autoencoder就是失败的了。例如，编码成是一维的，但是编码函数可以将\(x^{(i)}\)表示为\(i\)。</p>

### Regularized Autoencoders
<p>如果编码层和输入层的维度一样或者大于输入层的维度，这个就成了overcomplete。这种情况下，即使是线性的编码函数和解码函数都可以完成从输入复制到输出的过程，并且没办法学习到任何有用的信息。</p>
<p>理想情况下，任何架构的autoencoder都可以成功的训练，前提是按照数据分布的复杂程度选择合适的编码层维度以及编码和解码函数。Rgularized Autoencoder就提供了这样的能力，它不通过选择小维度的编码层和使用浅层的编码和解码过程来实现这样的功能，而是使用一个损失函数式的模型具备除了从输入拷贝数据到输出的能力之外的一些属性。这些属性包括，特征的稀疏性，小的导数，以及对噪音和数据丢的的的鲁棒性。一个Regularized的autoencoder可以是非线性的或者是overcomplete的，但却能从训练数据中学习到有用的信息。  </p>
<p>除了这里讨论的regularized autoencoder之外，任何的含有隐含变量和推理过程的生成模型都能够被看作为一种autoencoder。例如variational autoencoder以及generative stochastic networks，这些模型因为本身就是为了最大化训练数序的概率分布，所以他能够很好的学习数据的特征。  </p>

#### Sparse Autoencoders
<p>稀疏autoencoder在训练的过程当中在\(h\)层中含有一个稀疏惩罚函数\(\Omega(h)\)，于是误差函数的形式如下  </p>
$$L(x,g(f(x)))+\Omega(h)$$    
<p>其\(g(h)\)是解码输出函数，\(h=f(x)\)为编码输出。  </p>
<p>Sparse Autoencoder的主要目的是为了分类而进行特征学习。它必须能够相应数据集的统计特征，这种方法进行训练可以在复制过程中将学习特征作为一个副产品。</p>
<p>我们可以将\(\Omega(h)\)作为一个简单的regularizer项，和其他的regularizer（如weight decay）不同的是，对于这个regularizer并没有相应的贝叶斯解释。如在5.6.1章中描述的那样，带有的weight decay和其他的正则惩罚项的训练可以看成为对参数的先验分布的正则惩罚的MAP。例如，最大化\(p(\theta|x)\),就等同于最大化\(\log p(x|\theta)+\log p(\theta)\)，其中\(\log p(\theta)\)就是参数的先验分布。而这里的惩罚函数不仅依赖输入数据，并且也不是任何形式的先验部分的定义。但是我们仍然可以将其看成是对某个函数的偏好的隐含表达方法。 </p>  
  
  ![1](https://1drv.ms/i/s!AjPKhcGRzFUoh25gGuCraC6uzjPh)

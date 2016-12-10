---
layout: post
comments: true
categories: DeepLearning
---
## DNNClassifier
<p>从前面的例子可以看出，用Tensorflow在实现深度学习的时候，我们需要实现每一层的细节，对于一些通用的功能来说，很多代码其实是可以重用的。谷歌当然知道这个问题，于是tensorflow有一个更高层次的API---tf.contrib.learn，里面有很多通用的接口。本文介绍的就是其中的DNNClassifier，这次我们使用的是Iris数据集，这个数据集也是一个非常经典的数据集，下一节会介绍这个数据集。</p>
<p>实现步骤:</p>
1. 将Iris导入tensorflow dataset
2. 构建一个DNNClassifier
3. 配置和训练model
4. 测试分类准确率
5. 用新的样本进行测试

### Iris数据集
<p>Iris数据集是关于三种花的，分别问Iris versicolor, Iris Virginica, Iris Setosa, 下面有三种花的图片：</p>
* Setosa
![Setosa](https://www.tensorflow.org/versions/r0.12/images/iris_three_species.jpg)
* versicolor
![versicolor](https://www.tensorflow.org/versions/r0.12/images/iris_three_species.jpg)
* virginica
![virginica](https://www.tensorflow.org/versions/r0.12/images/iris_three_species.jpg)
<p>从图片也看出来，三种花特征还是很明显的，实际的数据就是通过4个特征来做分类的。数据结果如下，最后一下代表品种。</p>

### Reference
1. https://en.wikipedia.org/wiki/Iris_flower_data_set    
2. 

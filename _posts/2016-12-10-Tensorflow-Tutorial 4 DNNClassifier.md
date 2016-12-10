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
* Setosa 左
* versicolor 中
* virginica 右
![Setosa](https://www.tensorflow.org/versions/r0.12/images/iris_three_species.jpg)
<p>从图片也看出来，三种花特征还是很明显的，实际的数据就是通过4个特征来做分类的。数据结果如下，最后一下代表品种。</p>
<table>
  <tr>
    <th>Sepal Length</th>
    <th>Sepal Width</th>
    <th>Petal Length</th>
    <th>Petal Width</th>
    <th>Species</th>
  </tr>
  <tr>
    <td>5.1</td>
    <td>3.5</td>
    <td>1.4</td>
    <td>0.2</td>
    <td>0</td>
  </tr>
  <tr>
    <td>7.0</td>
    <td>3.2</td>
    <td>4.7</td>
    <td>1.4</td>
    <td>1</td>
  </tr>
</table>

<p>这里程序使用的数据集有总共150条，其中120条做样本，后面30条做测试。数据下载地址：</p>
* [iris_training.csv](http://download.tensorflow.org/data/iris_training.csv)
* [iris_test.csv](http://download.tensorflow.org/data/iris_test.csv)

### 代码实现
{% highlight python %}

import tensorflow as tf
import numpy as np

IRIS_TRAINING = "dataset/iris/iris_training.csv"
IRIS_TEST ="dataset/iris/iris_test.csv"

# 我的数据有表头，因此需要用要用with_header
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename = IRIS_TRAINING,
    target_dtype = np.int,
    features_dtype = np.float32
)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)


feature_columns = [tf.contrib.layers.real_valued_column("",dimension=4)]

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10,20,10],
                                            n_classes=3,
                                            model_dir="/logs")

classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000)

accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)['accuracy']
print("Accuracy: {0:f}".format(accuracy_score))


new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = list(classifier.predict(new_samples, as_iterable = True))
print('Predictions:{}'.format(str(y)))

{% endhighlight %}
### Reference
1. https://en.wikipedia.org/wiki/Iris_flower_data_set    


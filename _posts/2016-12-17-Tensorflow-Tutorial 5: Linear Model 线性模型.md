---
layout: post
comments: true
categories: DeepLearning
---

* content
{:toc}


## 什么是线性模型
<p>相信大多数人，刚开始接触机器学习的时候，就会接触到线性模型。来看一个简单的例子：通过人的年龄、受教育年数、工作年限等信息，可以预测出一个人的基本收入水平，预测方法就是对前面的限定特征赋予不同的权值，最后计算出工资；此外，线性模型也可以用于分类，例如逻辑回归就是一种典型的线性分类器。</p>
<p>相对于其他的复杂模型来说，线性模型主要有以下几个优点：</p>  
* 训练速度快
* 大量特征集的时候工作的很好
* 方便解释和debug，参数调节比较方便
## tf.learn关于线性模型的一些API   
* FeatureColumn
* sparse_column 用于解决类别特征的稀疏问题，对于类别型的特征，一般使用的One hot方法，会导致矩阵稀疏的问题。
{% highlight python %}
eye_color = tf.contrib.layers.sparse_column_with_keys(
  column_name="eye_color", keys=["blue", "brown", "green"])
  education = tf.contrib.layers.sparse_column_with_hash_bucket(\
    "education", hash_bucket_size=1000)#不知道所有的可能值的时候用这个接口
{% endhighlight %}
* Feature Crosses 可以用来合并不同的特征
{% highlight python %}
sport = tf.contrib.layers.sparse_column_with_hash_bucket(\
    "sport", hash_bucket_size=1000)
city = tf.contrib.layers.sparse_column_with_hash_bucket(\
    "city", hash_bucket_size=1000)
sport_x_city = tf.contrib.layers.crossed_column(
    [sport, city], hash_bucket_size=int(1e4))
{% endhighlight %}
* Continuous columns 用于连续的变量特征
> age = tf.contrib.layers.real_valued_column("age")
* Bucketization 将连续的变量变成类别标签
> age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

## tf.contrib.learn.LinearClassifier和LinearRegressor 
<p>这两个一个用于分类，一个用于回归，使用步骤如下</p>
* 创建对象实例，在构造函数中传入featureColumns
* 用fit训练模型
* 用evaluate评估
<p>下面是一段示例代码：</p>
{% highlight python %}
e = tf.contrib.learn.LinearClassifier(feature_columns=[
  native_country, education, occupation, workclass, marital_status,
  race, age_buckets, education_x_occupation, age_buckets_x_race_x_occupation],
  model_dir=YOUR_MODEL_DIRECTORY)
e.fit(input_fn=input_fn_train, steps=200)
# Evaluate for one step (one pass through the test data).
results = e.evaluate(input_fn=input_fn_test, steps=1)

# Print the stats for the evaluation.
for key in sorted(results):
    print "%s: %s" % (key, results[key])
{% endhighlight %}

## Wide and deep learning
<p>最近刚看了这篇论文，打算专门写一章来详细讲解，这个训练模型的出现是为了结合memorization和generalization。下面推荐几篇文章:</p>
[research blog](https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html)
[Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
<p>模型结构如下：</p>
![wide and deep](https://1.bp.blogspot.com/-Dw1mB9am1l8/V3MgtOzp3uI/AAAAAAAABGs/mP-3nZQCjWwdk6qCa5WraSpK8A7rSPj3ACLcB/s1600/image04.png)

## 数据描述
<p>下面我们用具体的示例来演示如何使用线性模型：通过统计数据，从一个人的年龄、性别、教育背景、职业来判断这个人的年收入是否超过50000元，如果超过就为1，否则输出0.下面是我从官网截取的数据描述[数据源](https://archive.ics.uci.edu/ml/datasets/Census+Income)：</p>
* Listing of attributes: >50K, <=50K. 
* age: continuous. 
* workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
* fnlwgt: continuous. 
* education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
* education-num: continuous. 
* marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
* occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, * * Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
* relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
* race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. 
* sex: Female, Male. 
* capital-gain: continuous. 
* capital-loss: continuous. 
* hours-per-week: continuous. 
* native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.


## reference
1. https://archive.ics.uci.edu/ml/datasets/Census+Income
2. https://www.tensorflow.org/tutorials/wide/


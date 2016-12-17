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
* Feature Crosses 可以用来合并不同的特征
* Continuous columns 用于连续的变量特征
* Bucketization 将连续的变量变成类别标签
## tf.contrib.learn.LinearClassifier和LinearRegressor 
<p>下面我们用具体的示例来演示如何使用线性模型：通过统计数据，从一个人的年龄、性别、教育背景、职业来判断这个人的年收入是否超过50000元，如果超过就为1，否则输出0.下面是我从官网截取的数据描述：</p>
> Attribute Information: [数据源](https://archive.ics.uci.edu/ml/datasets/Census+Income)
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


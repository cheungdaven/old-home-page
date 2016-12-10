---
layout: post
comments: true
categories: DeepLearning
---

## Tensorlfow教程3 MNIST feed-forward neuron network


### 网络结构
### TensorBoard
<p>本来这篇文章的很重要的一部分就是介绍tensorboard，但是由于目前tensorflow的一些问题，我的电脑暂时不能使用，所以我目前也没有看到真实的效果
，下面是官网的解释：</p>
> Currently, TensorBoard external dependencies (JS scripts, css, images etc.) are not part of PIP package built by CMake. As a result, when you navigate to TensorBoard on Windows, it shows a blank screen. Added CMake scripts to download TensorBoard dependencies and make them part of PIP package.
[issue5844](https://github.com/tensorflow/tensorflow/pull/5844)
<p>tensorboard的目的是让神经网络的结果可视化，但前提是要自己利用API(summary)去采集log,按照规范才能出来结果。</p>

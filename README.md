# 使用DQN（Deep Q-Network）网络学习如何玩转Flappy Bird

<img src="./images/flappy_bird_demp.gif" width="250">

7分钟版本 : [DQN for flappy bird](https://www.youtube.com/watch?v=THhUXIhjkCM)

## 梗概
这个项目依据的是“用深度强化学习玩转Atari"[2]中的Deep Q-Network算法描述，并且显示出这个学习算法可以被Flappy Bird深远归纳。

## 运行环境:
* Python 2.7 or 3
* TensorFlow 0.7
* pygame
* OpenCV-Python

## 如何运行?
```
git clone https://github.com/yenchenlin1994/DeepLearningFlappyBird.git
cd DeepLearningFlappyBird
python deep_q_network.py
```

## 什么是 Deep Q-Network?
它是一个卷积神经网络，是被各种各样的Q-learning所训练的，这种Q-learning的输入是未经加工的像素并且输出是一个用来评估未来收益的估价函数。

对于那些对深度强化学习感兴趣的人，我强烈推荐各位去阅读下面这个链接内容：
[Demystifying Deep Reinforcement Learning](http://www.nervanasys.com/demystifying-deep-reinforcement-learning/)

## Deep Q-Network 算法结构

The pseudo-code for the Deep Q Learning algorithm, as given in [1], can be found below:

```
Initialize replay memory D to size N
Initialize action-value function Q with random weights
for episode = 1, M do
    Initialize state s_1
    for t = 1, T do
        With probability ϵ select random action a_t
        otherwise select a_t=max_a  Q(s_t,a; θ_i)
        Execute action a_t in emulator and observe r_t and s_(t+1)
        Store transition (s_t,a_t,r_t,s_(t+1)) in D
        Sample a minibatch of transitions (s_j,a_j,r_j,s_(j+1)) from D
        Set y_j:=
            r_j for terminal s_(j+1)
            r_j+γ*max_(a^' )  Q(s_(j+1),a'; θ_i) for non-terminal s_(j+1)
        Perform a gradient step on (y_j-Q(s_j,a_j; θ_i))^2 with respect to θ
    end for
end for
```

## 实验

#### 环境
因为Deep Q-network是被从游戏屏幕中每一帧观测到的未经加工的像素所训练，[3]发现如果去除游戏中的背景可以使它收敛的更快。这一过程如下图演示
<img src="./images/preprocess.png" width="450">

#### 网络结构
根据[1]，我首先将游戏的画面依据如下步骤进行了预处理：

1. 将图片转化为灰度模式
2. 将图片重新规范为80*80的尺寸
3. 堆积最后4幅画面去生成一个80*80*4大小的阵列作为网络的输入

这个网络结构如下图所示。第一层卷积用一个8x8x4x32的卷积核以尺寸为4的步长处理输入图片，卷积输出紧接着进入一个2x2的最大值池化层进行池化。第二层卷积使用的是一个4x4x32x64的卷积核，步长为2，然后同样用了最大值池化层进行池化。第三层卷积用的是3x3x64x64卷积核，步长为1，我们接着再一次使用了最大值池化层。最后的隐藏层包含了256个全连接神经元，激活函数为ReLU。

<img src="./images/network.png">

最后的输出层有着和游戏中能展示出来的各种动作相同维度，也就是第0个索引总是对应着什么都不做。输出层的值代表着在各种有效动作下输入状态Q函数。在每一个步中，这个网络表现出的动作对应着贪心策略下的最大Q值。

#### 训练
首先，我随机初始化了所有权值矩阵，用的是一个正态分布，标准差为0.01。然后设置了重复记忆最大实验次数为50000.

我开始依据随机选择的统一动作训练前10000步，但是并不更新网络中的权值。这使得这个系统在训练之前根植于重复记忆。

注意并不像[1]，[1]中初始化ϵ = 1，在接下来的3000000画面中，我线性地将ϵ从0.1退火到0.0001。我这么做的原因是代理器可以每隔0.03s(FPS=30)选择一次游戏中的动作，过高的ϵ会使得小鸟拍打的次数过多，因此很容易使得小鸟趋于屏幕的顶端并且最终傻了吧唧地撞到管子。这种情况会使得Q函数收敛的相对较慢，因为它只有在ϵ值较低的时候，才能识别其他情况。
然而，在其他游戏中，ϵ有时被初始化为1甚至更合理。
During training time, at each time step, the network samples minibatches of size 32 from the replay memory to train on, and performs a gradient step on the loss function described above using the Adam optimization algorithm with a learning rate of 0.000001. After annealing finishes, the network continues to train indefinitely, with ϵ fixed at 0.001.
在训练的每一步中，此网络的样本最小捆绑尺寸为32，从重复记忆中训练，并且是使用的Adam优化器对损失函数进行的梯度下降，学习率为0.000001。之后每一次训练结束，此网络都会继续无限的训练，这时ϵ固定在0.001。
## FAQ

#### 未找到关卡
Change [first line of `saved_networks/checkpoint`](https://github.com/yenchenlin1994/DeepLearningFlappyBird/blob/master/saved_networks/checkpoint#L1) to 

`model_checkpoint_path: "saved_networks/bird-dqn-2920000"`

#### 如何改良?
1. 评论[这个链接](https://github.com/yenchenlin1994/DeepLearningFlappyBird/blob/master/deep_q_network.py#L108-L112)

2. 优化 `deep_q_network.py`如下的参数:
```python
OBSERVE = 10000
EXPLORE = 3000000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1
```

## 文献依赖

[1] Mnih Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. **Human-level Control through Deep Reinforcement Learning**. Nature, 529-33, 2015.

[2] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. **Playing Atari with Deep Reinforcement Learning**. NIPS, Deep Learning workshop

[3] Kevin Chen. **Deep Reinforcement Learning for Flappy Bird** [Report](http://cs229.stanford.edu/proj2015/362_report.pdf) | [Youtube result](https://youtu.be/9WKBzTUsPKc)

## 免责声明
此项目很大程度上是基于如下依赖：

1. [sourabhv/FlapPyBird] (https://github.com/sourabhv/FlapPyBird)
2. [asrivat1/DeepLearningVideoGames](https://github.com/asrivat1/DeepLearningVideoGames)


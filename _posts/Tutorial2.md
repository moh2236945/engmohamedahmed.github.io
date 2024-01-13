## Tutorial 2 : Backpropagation

**Contents**

1- Basic Math for Backpropagation

2- Backpropagation in Convolution Network

3.1- Backpropagation Through the Fully Connected (Dense) layer.

3.1.1.Derivative of loss with respect to weights

3.1.2.Derivative of loss with respect to inputs

3.1.3Derivative of loss with respect to bias

3.1.4Updating weights & bias using Gradient Descent

3.2-Backpropagation Through the Convolution Layer

3.2.1.Derivative of Loss with respect to inputs in the Convolution Layer.

3.2.2.Derivative of Loss with respect to bias

3.2.3.Updating Kernels and Bias using Gradient Descent

4-Why use the backpropagation algorithm?

5-advantages of the backpropagation algorithm:

6-Types of backpropagation

7-Drawbacks of the backpropagation algorithm

7.1-vanishing gradient  

7.2-Eploding

8-Weight Initialization

8.1-Initializing weights to zero

8.2-Initializing weights randomly

8.3-Initializing weights using Heuristic

8.3.1. He-et-al Initialization.

8.3.2. Xavier initialization

9-Alternatives to traditional backpropagation

9.1 Local learning

9.1.1 HSIC Bottleneck 

9.1.2 Greedy InfoMax

9.1.3 Target Propagation 

9.2 Others

9.2.1-ReduNet  

9.2.2.Neural Tangent Kernel

9.2.3. Evolutionary Strategies

9.2.4. Feedback Alignment

9.2.5. Direct Feedback Alignment

---
**Basic Math for Backpropagation**

**Understanding Chain Rule in Backpropagation:**

Consider this equation

f(x,y,z) = (x + y)z

To make it simpler, split it into two equations.

f(x,y,z)=(x+y)z  so:

q=x+y	,	f=q*z

**Backpropagation in CNN**

the backward pass or backpropagation is doing the reverse of the forward pass where we find the gradient of each of the neuron with respect to the loss function to determine 

how much the output contribute to the overall loss. 

**In Forward Pass**, we start from the Convolution Layer until the Fully Connected Layer, 

**In backpropagation**, we start from the Fully Connected Layer all the way up to the Convolution Layer.

So we will begin from Dense layer(fully connected layer)convolution layer

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/back1.jpg)

**3.1-Backpropagation Through the Fully Connected (Dense) layer.**

**Steps**

first find a Loss Function which used to evaluate the output produced from the output layer of the Dense network. 

The choice of the Loss function depends on which task you want the network to do. (Explain later)

we will use Binary Cross Entropy loss 

Binary cross-entropy loss  finds the dissimilarity between the predicted probability and the true binary label. 

**Derivative of Binary Cross Entropy**

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/crossderv.jpg)

**3.1.1.Derivative of loss with respect to weights**

Next thing we need to do is to find how much the changes in weights affect the loss function ∂L/(∂w_ij ) which helps to find

the gradient of loss with respect to weights.

For deriving how the weights affect the overall loss function in the output layer, we  need to take into account two things, 

how much the output affects the loss 

how much the weights affect the output. This is the chain rule,

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/crossderv2.jpg)

**3.1.2.Derivative of loss with respect to inputs**

Next, let's find the gradient of the loss with respect to inputs, ie, how much the change in inputs affects the loss, note that the inputs to one corresponding layer are 

the output coming from the previous layer, and that's the whole thing about backpropagation, to pass the gradients of inputs backward. This illustration can be seen in the 

image shown. Using the chain rule, we can calculate this.

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/crossderv3.jpg)

The change in loss with respect to input is influenced by the weights and the change in loss with respect to the outputs. Here is how we can represent it,

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/crossderv4.jpg)

**3.1.3Derivative of loss with respect to bias**

Finally, we need to find the gradient of loss with respect to bias, ∂L/(∂b_j ) Using the chain rule we can find that

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/crossderv5.jpg)

This represents how much the loss changes with respect to the outputs, and how much the output changes with respect to the bias. Using the same idea for input gradient 

calculation, we can find the value of (∂y ̂_i)/(∂b_j )

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/crossderv6.jpg)

**3.1.4Updating weights & bias using Gradient Descent**

Ok, the calculations are done and we find the gradients, the next and most important step in backpropagation is to update the weights and bias using the gradients of 

weights and bias. Here is how the weights and bias can be updated,

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/crossderv7.jpg)

So we have completed the calculation of finding the gradients and updating the weights of the Fully Connected Layer of the CNN. 

**Note that**

for each layer, including the output layer, the same set of equations is used to compute the gradients and updated the parameters. This involves applying the chain rule and 

differentiating the loss function with respect to the layer's parameters, such as weights and biases, as well as with respect to the layer's inputs.

**3.2-Backpropagation Through the Convolution Layer**

the backward derivations for the Dense Layer and Convolutional Layer share certain similarities. 

The only difference is that we apply Convolution rather than dot product operation. 

we need to find the gradients with respect to weights, inputs, and bias. However, there are a few key differences to note.

Firstly, due to the 2D nature of the convolutional layer, we require three indices to track the dimensions. These indices include the indices for the height and width of 

the kernel, input image, and output, and then the index used to track the position of the kernel in the input image.

Secondly, instead of performing a dot product as in the Dense layer, the convolutional layer applies a convolution operation. This operation involves sliding a kernel or 

filter over the input matrix and computing element-wise multiplications and summations.

we have considered a single convolution. However, in a typical CNN, multiple convolutions are applied, each with its own set of weights and output feature map. In such 

cases, additional indices would be necessary to keep track of these convolutions. 

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/crossderv8.gif)

the gradients of the kernel are produced by convolving the output feature map and the input matrix.

the equation for finding how much the loss changes with respect to the weights using the chain rule,

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/crossderv9.jpg)

**3.2.1.Derivative of Loss with respect to inputs in the Convolution Layer.**

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/dervloss.gif)

The change in loss with respect to inputs can be given by,

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/dervloss1.jpg)

The change in the loss w.r.t inputs is given by the double summation of the change in loss w.r.t output produced after cross-correlation and the 180-degree rotated kernel. 

**3.2.2.Derivative of Loss with respect to bias**

Finally, let's see how to calculate the derivative of loss w.r.t bias. it depends on the change in loss with respect to the change in outputs

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/dervloss2.jpg)

That's it, we have derived all the necessary things including the gradients of Kernels, inputs, and bias, now let's update the Kernels and bias using gradient descent.

**3.2.3.Updating Kernels and Bias using Gradient Descent**

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/dervloss3.jpg)

**Why use the backpropagation algorithm?**

the network is trained using 2 passes: forward and backward. 

At the end of the forward pass, the network error is calculated, and should be as small as possible.

If the current error is high, the network didn’t learn properly from the data. It means that the current set of weights isn’t accurate enough to reduce the network error 

and make accurate predictions. so, we should update network weights to reduce the network error. 

The backpropagation algorithm is one of the algorithms responsible for updating network weights with the objective of reducing the network error. It’s quite important.

---
**5.-advantages of the backpropagation algorithm:**

1-Imemory-efficient in calculating the derivatives, as it uses less memory compared to other optimization algorithms, like the genetic algorithm. This is a very important feature, especially with large networks.

2-The backpropagation algorithm is fast, especially for small and medium-sized networks. As more layers and neurons are added, it starts to get slower as more derivatives 

are calculated. 

This algorithm is generic enough to work with different network architectures, like convolutional neural networks, generative adversarial networks, fully-connected 

networks, and more.

3-There are no parameters to tune the backpropagation algorithm, so there’s less overhead. The only parameters in the process are related to the gradient descent algorithm, 

like learning rate.

There are many variants of Gradient Descent will explain later in optimization tutorial

6-Types of backpropagation

There are 2 main types of the backpropagation algorithm:

1-Traditional backpropagation is used for static problems with a fixed input and a fixed output all the time, like predicting the class of an image. In this case, the input 

image and the output class never change. 

2-Backpropagation through time (BPTT) targets non-static problems that change over time. It’s applied in time-series models, like recurrent neural networks (RNN).

**7-Drawbacks of the backpropagation algorithm**

**vanishing and exploding gradients**

The backpropagation algorithm considers all neurons in the network equally and calculates their derivatives for each backward pass. Even when dropout layers are used, the 

derivatives of the dropped neurons are calculated, and then dropped.

When training a deep neural network with gradient based learning and backpropagation, we find the partial derivatives by traversing the network from the the final layer 

(y_hat) to the initial layer. Using the chain rule, layers that are deeper into the network go through continuous matrix multiplications in order to compute their 

derivatives.

**vanishing gradient**  if the derivatives are small then the gradient will decrease exponentially as we propagate through the model until it eventually vanishes. For 

example, Activation Functions such as the sigmoid function have a very prominent difference between the variance of their inputs and outputs. They shrink and transform a 

large input space into a smaller output space, which lies between [0,1].

**Exploding** is the opposite of Vanishing and is when the gradient continues to get larger which causes a large weight update and results in the Gradient Descent to 

diverge.

Exploding gradients occur due to the weights in the Neural Network, not the activation function.

The gradient linked to each weight in the Neural Network is equal to a product of numbers. If this contains a product of values that is greater than one, there is a 

possibility that the gradients become too large.

The weights in the lower layers of the Neural Network are more likely to be affected by Exploding Gradient as their associated gradients are products of more values. This 

leads to the gradients of the lower layers being more unstable, causing the algorithm to diverge.

----

**8-Weight Initialization**

Weight Initialization is the process of setting the weights of a Neural Network to small random values that help define the starting point for the optimization of the model.

**Weight Initialization Techniques:**

**1-Initializing weights to zero**

If we initialize all our weights to zero, our Neural Network will act as a linear model because all the layers are learning the same thing.

Therefore, the important thing to note with initializing your weights for Neural Networks is to not initialize all the weight to zero.

**2-Initializing weights randomly**

Using random initialization defeats the problem caused by initializing weights to zero, as it prevents the neurons from learning the exact same features of their inputs. 

Our aim is for each neuron to learn the different functionalities of its input.

However, using this technique can also lead to vanishing or exploding gradients, due to incorrect activation functions not being used. It currently works effectively with 

the RELU activation function.

**3-Initializing weights using Heuristic**

This is considered the best technique to initialize weights for Neural Networks.

Heuristics serve as good starting points for weight initialization as they reduce the chances of vanishing or exploding gradients from occurring. This is due to the fact 

that the weights are neither too bigger than 1, nor less than 1. They also help in the avoidance of slow convergence.

**The most common heuristics used are:**

**3.1. He-et-al Initialization.**

When using the RELU activation function, this heuristic is used by multiplying the randomly generated values of W by:

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/he1.jpg)

**3.2. Xavier initialization**

When using the Tanh activation function, this heuristic is used by multiplying the randomly generated values of W by:

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/xavier1.jpg)

**9-Alternatives to traditional backpropagation**

Backpropagation (BP) was originally published in 1986, but its large-scale popularity is inseparable from the booming development of data, computing power, and algorithms 

in the past 10 years. The training of currently popular neural network models is based on BP, and mainstream algorithm frameworks also provide convenient automatic 

differentiation (AutoGrad) mechanisms to simplify gradient calculations. However, as the scale of the model increases exponentially, the cost of a complete end-to-end BP 

(E2EBP) is getting higher and higher, and can even reach tens of millions of dollars. Scientists have been working hard to find improvements and alternatives to 

backpropagation.

BP is just an algorithm that uses dynamic programming to reduce the complexity of gradient calculations when gradient descent is applied to neural networks . Therefore, as 

long as it is a gradient-based optimization method, it is unlikely to completely bypass BP. Even Local learning requires Do local BP.

**Improvements and alternatives to end-to-end backpropagation**

**1 Local learning**

Since the global end-to-end process is too long, is there a way to independently optimize each module/layer in the network (even more fine-grained ops)? If so, can modules 

be trained in parallel?

This is the idea of an algorithm such as Local Learning: by constructing an independent local loss for each module, and then calculating the local gradient based on this 

loss, to approximate the gradient of the complete E2EBP . From the perspective of computational graphs, the lower the dependence between the local losses of different 

modules, the higher the parallelism of training. At the same time, reasonable modularization and loss design can effectively avoid the problem of gradient disappearance or 

explosion.

**1.1 HSIC Bottleneck**

-Module loss design

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/hsic.jpg)

This method utilizes **HSIC (Hilbert-Schmidt independence criterion)** HSIC bottleneck is an alternative to conventional backpropagation, to measure the correlation between the 

input and the hidden layer, thereby making the representation of the hidden layer become more independent. Specifically, this method first extracts features from the input 

data and converts them into hidden layer representations through an encoder . These hidden layer representations are then remapped into the original space through a decoder . 
Finally, HSIC is used to maximize the independence between the hidden layer representation and the input and minimize the correlation between the hidden layer 

representation and the output. This method allows the network to learn more interpretable features and avoids some problems caused by the backpropagation algorithm .

**This method has several distinct advantages. **

1-Instead of solving problems by using the chain rule as traditional backpropagation does, HSIC solves problems layer-by-layer, eliminating problematic vanishing and 

exploding gradient issues found in backpropagation. 

2-facilitates parallel processing for training layers, and as a result requires significantly fewer operations. Finally, the proposed method removes backward sweeps to 

eliminate the requirement for symmetric feedback.

**Researchers presented two approaches**

1-first approach is a standard feedforward network (above left), generating one-hot results that can be directly permuted to perform classification

2-second approach is the σ-combined network (above right), in which researchers simply append a single layer as an aggregator to assemble all the hidden representations so 

that each is trained with a specific σ, with the need to provide all information at different scales σ to the post training.

comparing the ResNet post and ResNet backpropagation methods, the HSIC bottleneck provides a significant boost in performance, which opens the possibility of learning 

classification tasks at near-competitive accuracy but without the limitations of backpropagation.
























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













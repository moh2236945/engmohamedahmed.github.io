## Turorial 1

**Part1: Convolution**

1-Convolution

2-Padding

2-1-Type of Padding

2-1-1-Vaild padding

2-1-2-Same Padding

2-1-3-Full padding

2-2-Advantage of padding

2-3-Realtionship between padding and translation invariance

3-Strise

4-Advantage of Convolution in image processing

5-Convolution Vs cross-Correlation

6-Various method of calculation convolution

6-1-Sliding window

6-2-im2col

6-3-FFT

6-4-Winograd

7-Fully connected Layer

7.1-Batch Normalization For Fully connected Layer

7.2 Different between convolution layer and convolution layer

7.3 Disadvantages o FC

8-Notes

9-Glossary

10-Pytorch Code

11-convolution operation in numpy

----
**convolution**: the process of sliding filter on image (dot product(matrix multiplication)).

a type of matrix operation, consisting of a kernel, a small matrix of weights, that slides over input data performing element-wise multiplication with the part of the input it is on, then summing the results into an output.

**filter**: a grid of a discrete number(Kernels together) The filter has the same depth as input 

**convolution layer**: extract features from image and preserve the relationship between pixels

**feature maps** the output of convolutional layer after the convolution operation has occurred on an image 

**depth**: number of filter used(output channel)

**stride**: the size of convolution layer output .large size of stride meaning no over lap 

**receptive filed:** the part of image the visible to the filter 

**type of features:** Key point  features(specific location in image )	edges

**features detector:** take image and output locations(pixel coordinate)

**features descriptor** take an image and output feature vector

**A kernel** is a matrix of weights which are multiplied with the input to extract features

**A filter** a concatenation of multiple kernels, each kernel assigned to a particular channel of the input. Filters are always one dimension more than the  kernels.

( A filter: A collection of kernels)

**kernel size** is the dimensions of the kernel (H,W) 

**number of filters** refers to the number of output channels created after the kernel is convolved over the input image.

--------------
**1-Convolution**

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/conv1.gif)

---
This kernel “slides” over the 2D input data, performing an elementwise multiplication with the part of the input it is 

currently on, and then summing up the results into a single output pixel.

The kernel repeats this process for every location it slides over, converting a 2D matrix of features into yet another 2D 

matrix of features

Each kernel of the filter "slides" over its respective input channel, producing individual processed versions. 

Some kernels may have greater weight than others to emphasize certain input channels.

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/conv2.gif)

Each core of the filter produces its own processed version which is then summed so that the entire filter produces an overall output channel.

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/conv3.gif)

Finally, there is a term: "bias". Each output filter has a bias term, and the bias is added to the output channel to produce the final output channel.

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/conv4.gif)

Bias in a neural network is required to shift the activation function across the plane either towards the left or the right. 

**Difference between weight and bias in a convolutional neural network (CNN)**

**Weights** refer to the parameters that are learned by our CNN model during training. These weights tell us how much importance each feature has when making a prediction from  data. A higher weight means higher importance, while lower weights mean lesser importance for a specific feature. 

In other words, weights represent the strength of connection between different neurons of the model and helps capture relationships between data points as they pass through 

layers of filters and nodes within the network.

**Bias** 

works alongside weights to make predictions more accurate or add additional information to them beyond what just using weights alone can do. Unlike Weights, Bias 

values remain constant throughout training since they do not change with respect to any given input vector or example image you may input into your model’s system when 

running it through its training algorithm simulations. It essentially introduces an offset or “bias” value which helps shift a model's predicted outcome either up or down 

depending on whether this bias value is positive or negative respectively - aiding an overall better prediction accuracy from your trained models output given its provided 

inputs at test time.
-----
**2-Padding**

refers to edge filling of the input during the convolution operation to control the size and shape of the output feature map. Common padding methods include:

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/padd1.gif)

**2-1.1 Valid padding**

Without padding, the output size is:  (W - F + 1) x (H - F + 1)

Where W is the input width, H is the input height, and F is the convolution kernel size.

**2-1.2. Same padding**

Also known as Zero padding, it will evenly fill the edges of the input image with zeros so that the size of the output feature map is the same as the input.

The output size is: W x H

**2-1.3. Full padding**

Also called **Full convolution**, F-1 circles 0 are filled in the entire input image boundary.

The output size is: (W + F - 1) x (H + F - 1)

**2-2 The main functions of PADDING:**

- Control the size and shape of the output feature map
  
- Maintain image edge information to avoid loss
  
- Increase network depth while keeping the receptive field unchanged
  
Care needs to be taken to avoid excessive padding leading to overfitting. 

Usually Same padding is used the most, which can maintain the output size without being too complicated.

**2-3 what is the relationship between padding and translation invariance?**

Without padding, the output of convolution does not seem to reflect the position very well:

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/padand1.png)

If padding is added, the output of the convolution will reflect strong position information:

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/padand2.png)

the addition of padding brings about translational variability , making the output and input spatially corresponding. 

So the choice of padding is important and some poor choices can lead to poor model performance.
----
**3-Striding**

The idea of Stride is to change the movement step size of the convolution kernel to skip some pixels. Stride is 1, which means that the convolution kernel slides through 

every pixel with a distance of 1, which is the most basic single-step sliding, as the standard convolution mode. Stride is 2, which means that the moving step size of the 

convolution kernel is 2, skipping adjacent pixels, and the image is reduced to 1/2 of the original size. Stride is 3, which means that the moving step size of the 

convolution kernel is 3, skipping 2 adjacent pixels, and reducing the image to 1/3 of the original size.

In Neural Network (All_CNN) the author using Conv layer with stride=2 instead of using Max_pool 

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/strid.gif)

**Each filter** in a convolution layer produces one output channel

Each of the per-channel processed versions are then summed together to form one channel. 

The kernels of a filter each produce one version of each channel, and the filter as a whole produces one overall output channel.

Finally, then there’s the bias term. The way the bias term works here is that each output filter has one bias term. The bias gets added to the output channel so far to 

produce the final output channel.

**Summary**

Each filter processes the input with its own, different set of kernels and a scalar bias with the process described above, producing a single output channel. They are then 

concatenated together to produce the overall output, with the number of output channels being the number of filters. A nonlinearity is then usually applied before passing 

this as input to another convolution layer, which then repeats this process.

-----
**4-Advantage of convolution in image recognition**

1- weight sharing - reducing the number of effective parameters 

2-image translation (allowing for the same feature to be detected in different parts of the input space).

3- Convolutions are not densely connected; not all input nodes affect all output nodes. This gives convolutional layers more flexibility in learning.

4- CNNs are particularly well-suited for tasks that involve processing and analyzing structured data with spatial properties, such as images, because they can effectively 

capture and leverage spatial information.

**(Spatial information in (CNN)** refers to the information related to the arrangement, location, and spatial relationships between features or elements in an input image 

or data grid. )

5- CNNs capture **spatial hierarchies**, meaning they can recognize low-level features (e.g., edges and corners) and combine them to recognize higher-level features (e.g., 

shapes and objects) in a hierarchical manner.

---------

**5-Convolution v.s. Cross-correlation**

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/crossvscorr.gif)

**Cross-Correlation:**

**Correlation** is the process of moving a filter mask (kernel)over the image and computing the sum of products at each location. Correlation is the function of displacement of the filter. In other words, the first value of the correlation corresponds to zero displacement of the filter, the second value corresponds to one unit of 

displacement, and so on.

**Mathematical Formula :**

The mathematical formula for the cross-correlation operation in 1-D on an Image I using a Filter F is given by Figure 3. It would be convenient to suppose that F has an odd 

number of elements, so we can suppose that as it shifts, its centre is right on top of an element of Image I. So we say that F has 2N+1 elements, and these are indexed from 

-N to N, so that the centre element of F is F(0).

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/cross1.png)

Similarly, we can extend the notion to 2-D which is represented in Figure 4. The basic idea is the same, except the image and the filter are now 2D. We can suppose that our 

filter has an odd number of elements, so it is represented by a (2N+1)x(2N+1) matrix.

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/cross2.png)

The Correlation operation in 2D is very straight-forward. We just take a filter of a given size and place it over a local region in the image having the same size as the 

filter. We continue this operation shifting the same filter through the entire image.

**Convolution**

The convolution operation is very similar to cross-correlation operation but has a slight difference. In Convolution operation, kernel is first flipped by an angle of 180 

degrees and is then applied to the image. The fundamental property of convolution is that convolving a kernel with a discrete unit impulse yields a copy of the kernel at 

the location of the impulse.

We saw in the cross-correlation section that a correlation operation yields a copy of the impulse but rotated by an angle of 180 degrees. Therefore, if we pre-rotate the 

filter and perform the same sliding sum of products operation, we should be able to obtain the desired result.

**Mathematical Formula:**

The convolution operation applied on an Image I using a kernel F is given by the formula in 1-D. Convolution is just like correlation, except we flip over the filter before 

correlating.

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/cross3.png)

-----

**6-Various methods of calculating convolution**

The first thing to be clear is that the convolution mentioned here refers to the default convolution in ConvNet, not the convolution in the mathematical sense. In fact, the 

convolution in ConvNet is related to the cross correlation in mathematics.

There are many ways to calculate convolution, and the common ones are as follows:

**6.1.Sliding window:** This method is the most intuitive and simplest method. However, this method is not easy to achieve large-scale acceleration. Therefore, this method 

is usually not used (but it is not absolutely not used. Under some specific conditions, this method is the most efficient.)

**6.2.im2col:** At present, almost all mainstream computing frameworks including Caffe, MXNet, etc. have implemented this method. This method converts the entire 

convolution process into a GEMM process, and GEMM is extremely optimized in various BLAS libraries. Generally speaking, the speed is faster.

**6.3.FFT:** Fourier transform and fast Fourier transform are calculation methods often used in classic image processing. However, they are usually not used in ConvNet, 

mainly because the convolution templates in ConvNet are usually relatively small, for example 3 × 3Wait, in this case, the time overhead of FFT is greater.

**6.4.Winograd:** Winograd is a method that has been in existence for a long time and has been recently rediscovered. In most scenarios, the Winograd method shows and has 

greater advantages. Currently, this method is used to calculate convolution in cudnn.

-----

**7-Fully Connected Layer(Dense Layer)**

1-Fully connected layers connect every neuron in one layer to every neuron in another layer. It is in principle the same as the traditional multi-layer perceptron neural network.

2-FC layers are typically found towards the end of a neural network architecture and are responsible for producing final output predictions.

3-FC layers often come after the convolutional and pooling layers. They are used to flatten the 2D spatial structure of the data into a 1D vector and process this data for 

tasks like classification.

4-The number of neurons in the final FC layer usually matches the number of output classes in a classification problem. For instance, for a 10-class digit classification 

problem, there would be 10 neurons in the final FC layer, each outputting a score for one of the classes.

For a Fully Connected layer, batch normalization would be applied to the output of the FC layer and before the activation function.

![](https://github.com/moh2236945/engmohamedahmed.github.io/blob/main/_posts/fcl1.png)

**Batch Normalization for Fully Connected Layer (Steps)**

1.	During each training iteration, the algorithm calculates the mean and variance of the activations for each layer over the current mini-batch.

2.	It then normalizes the activations by subtracting the mean and dividing by the standard deviation.

3.	After normalization, the algorithm scales and shifts the result by learning two new parameters (often denoted as gamma and beta). These parameters ensure that the normalization doesn’t limit the layer’s ability to represent more complex patterns (since sometimes the network might want the activations to have a certain mean and variance).
  
4.	At the time of inference (i.e., when the model is being used for prediction rather than being trained), the batch mean and variance are replaced with the entire dataset’s moving average to ensure a consistent prediction.

**Different between Convolution Layer and Fully Connected Layer**

**A fully connected layer** offers learns features from all the combinations of the features of the previous layer, where a convolutional layer relies on local spatial 

coherence with a small receptive field.

**Disadvantages of FC**

**fully connected networks** make no assumptions about the input they tend to perform less and aren’t good for feature extraction. 

have a higher number of weights to train that results in high training time on the other hand CNNs are trained to identify and extract the best features from the images for 

the problem at hand with relatively fewer parameters to train.

**Fully connected layers** are incredibly computationally expensive. That’s why we use them only to combine the upper layer features

The **strength** of convolutional layers over fully connected layers is precisely that they represent a narrower range of features than fully-connected layers. A neuron ina 

**fully connected layer** is connected to every neuron in the preceding layer, and so can change if any of the neurons from the preceding layer changes. 

A neuron in a convolutional layer, however, is only connected to "nearby" neurons from the preceding layer within the width of the convolutional kernel. As a result, the 

neurons from a convolutional layer can represent a narrower range of features in the sense that the activation of any one neuron is insensitive to the activations of most 

of the neurons from the previous layer.

**Notes**

**A convolutional layer** assumes a structure in the data, i. e. that the input data is a matrix, with height, width and eventually channels.

**Stacking smaller convolutional layers** is lighter, than having bigger ones. It also tends to improve the result, 

adding too much padding to increase the dimensionality would result in greater difficulty in learning

Three main types of layers are used to build CNN architecture: Convolutional Layer, Pooling Layer, and Fully-Connected Layer.

**Convolutional layers** and **a fully connected layer** better at detecting spatial features 

this means feature a convolutional layer can learn, a fully connected layer could learn 

**Spatial features** refer to the characteristics of physical space or location. These features can include things like distance, direction, shape, size, and relationships 

between different geographic elements.

**Most of the features in an image are usually local**. Therefore, it makes sense to take few local pixels at once and apply convolutions.

**Most of the features** may be found in more than one place in an image. This means that it makes sense to use a single kernel all over the image, hoping to extract that 

feature in different parts of the image.

**standard convolution** performs the channel wise and spatial-wise computation in one step

**Glossary**

**Invariance** means that you can recognize an object as an object, even when its appearance varies in some way. This is generally a good thing, because it preserves the 

object's identity, category, (etc) across changes in the specifics of the visual input, like relative positions of the viewer/camera and the object.

**difference between translation invariance and translation equivariance.**

**Translation invariance** means that the system produces exactly the same response, regardless of how its input is shifted. For example, a face-detector might report "FACE 

**FOUND" for all three images in the top row.**

Various Invariances:

- Size Invariance
  
- Translation Invariance
  
- Perspective (Rotation/Viewpoint) Invarience
  
- Lighting Invarience
  
**Translation equivariance**: means that the system works the same in different locations, but its response changes as the target location changes. For example, the 

instance splitting task needs to be translated and varied, and if the target is translated, the output instance mask should also change accordingly.

**A local feature** is an image pattern which differs from its immediate neighborhood

**How a convolutional neural network able to learn invariant features?**

the pooling operation is the main reason for the translation invariant property in CNNs. 

**invariance** is due to the convolution filters (not specifically the pooling) while the 

fully-connected layers at the end are “position-dependent”,

**The pooling operation** reduces the height and width of these volumes, while the increasing number of filters in each layer increases the volume depth.

**In conclusion**, what makes a CNN invariant to object translation is the presence of convolution filters. Additionally, 

**Why do Convolutional Neural Networks have Translation Invariance**

**convolution** is defined as a feature detector at different locations, which means that no matter where the target appears in the image, it will detect the same features 

and output the same response. For example, if a face is moved to the lower left corner of the image, the convolution kernel will not detect its features until it is moved 

to the lower left corner.








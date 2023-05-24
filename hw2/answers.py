r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**


1.A. Since the jacobain tensor represents the derivatives Y w.r.t X, and the size of Y is (64, 512) and the size of X is (64, 1024), it should take into account each sample in the batch and each output feature w.r.t each sample in the batch and each output feature, So - (64, 512, 64, 1024).

1.B. It is sparse since the samples are unrelated to each other, each derivative of output feature of one sample w.r.t input feature of another sample we be zero, since they are not related. So most of the valuess in the tensor will be zero.

1.C. No, we don't need to materialize the above Jacobian in order to calculate the downstream gratdient w.r.t. to the input, since get compute it by: $\delta\mat{X}$ = $\pderiv{L}{\mat{X}}$ = $\pderiv{L}{\mat{Y}}\cdot W^{T} $


2.A. Since the jacobain tensor represents the derivatives Y w.r.t W, and the size of Y is (64, 512) and the size of W is (512, 1024), so like before, we will get the shape of (64, 512, 512, 1024).

2.B.  It isn't sparse since the the output is related to the weights, so the it's no likely that the elements of the tensor is zeros.

2.C. No (like 1.C), we don't need to materialize the above Jacobian in order to calculate the downstream gratdient w.r.t. to the input, since get compute it by: $\delta\mat{W}$ = $\pderiv{L}{\mat{W}}$ = $\cdot X^{T}\pderiv{L}{\mat{Y}} $


"""

part1_q2 = r"""
**Your answer:**

No, back-propagation isn’t required in order to train neural networks with decent-based optimization, since we can allways calculate the entire derivative without using the chain rule (Although it is an awful idea).
"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0.1, 0.01, 0.001

    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0.001,
        0.02,
        0.02,
        0.0002,
        0.001,
    )

    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0.0001,
        0.0008,
    )
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

1. According to the graphs, we can see that without the dropout, we get more stable accurcy in the train epochs,
where with the dropout we have more spikes between the epochs.
Also, without the dropout were getting much higher test acc, and it gets lower as we increasing hte dropout,
where in test acc its opposite (not including the 0.8 dropout).
The results match with our expectations, since the point of the dropout is the increase the generalisation i.e. 
increasing the test acc on behalf of the train acc and that exectly what we see between the dropout=0 and dropout=0.4.

2. About the high dropout=0.8, we "ignoring" 80% of the neurons at the time, so it was expected that the result will be poor for
both the training and the testing, while, as we mention  before, for the lower dropout=0.4, were getting better test results than
no dropout thanks to the generalistion given by the dropout.


"""

part2_q2 = r"""
**Your answer:**

It is possible for the test loss to increase for a few epochs while the test accuracy also increases since the loss is calc using
the CE loss function, when the acc is calc by the correct_labeled/all_labeled, so for example, we can add 1 sample to the correct
labeled on an epoch, but the actual values of the CE loss also increase by updating the Xi values while still gettig the same labels.

"""

part2_q3 = r"""
**Your answer:**

1. The gradient descent is a family of methods/algorithms, used for finding the minima of a function in an iterative way, by taking
steps in the direction of the -gradient untill convergence the the minima.
The Back-propagation is an algorithm used in order to find the gradient of a function, using the chain-rule (mainly in use in SGD).

2. The difference between SGD to GD, is of course, that the SGD is a stochastic method, in a way that it does't nessecerly taking a
step toward the actual gradient of the function, but instead using only a subset of the samples (batch) and compute the gradient using
only them which is an aproximatoin to the gradient.

3. The SGD used more often in the practice of deep learning than the GD because deep learning models using a lot of samples, features,
and parameters which makes the training procces very time and compute consuming.By using SGD, we aren't calc the gradients using all of
samples becuase we using only batches of them at the time, which make the trainig faster.

4.A. His approach will produce a gradient equivalent to GD, since we can Accumulate the loss for each batch to get:
Let k be the number of the batches, each of size r:
$$ Actual-GD-Loss = \sum_{i=1}^{n} Loss(sample_i) = \sum_{i=1}^{k} \sum_{j=1}^{r} Loss(sample_i*(n/k) + j) =  \sum_{i=1}^{k} Loss(Batch_i) = Approach-GD-Loss $$

4.B. One explanation could be that the Torch Library stores other data in the memory, like grads (for the back-prp), and that will left
less space for the batches.

"""

part2_q4 = r"""
**Your answer:**
TODO:
1.A.
Instead of storing each step (function) grad, we will store the accumalted grad, since the last storing, so the complexity cound be even O(1).

1.B.
We can use the memory we used for the forward 
2.

3.

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 4  # number of layers (not including output)
    hidden_dims = 10  # number of output dimensions for each hidden layer
    activation = "relu"  # activation function to apply after each hidden layer
    out_activation = "relu"  # activation function to apply at the output layer
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = torch.nn.CrossEntropyLoss()  # One of the torch.nn losses
    lr, weight_decay, momentum = 0.02, 0.001, 0.5  # Arguments for SGD optimizer

    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


1. The Optimization error is not high since we got relativly low training error across the experiment.
2. The Generaliztion error is not high since along the experiments with higher w and h, we got higher test accuracy.
3. Although we got some approximation error (as we can see in the decision plots), it's still not high as the decistion boundry is almost optimal.


"""

part3_q2 = r"""
**Your answer:**

Since we train with mostly samples from deg=10 and noise=0.2 and less samples from deg=50 and noise-0.25, we will have more uncertainty in the validation, as the samples there can have more noise.
This will lead to higher FNR because such a data generating process will lead to decision
boundry that fits the training set, but as we can see from plotting the points of the training set,
a good decision boundry for them will result higher FNR of the validation set.

"""

part3_q3 = r"""
**Your answer:**

We will not chose the ROC as we chose above, since it tries to equalise the FPR and FNR, 
as we want to same money (FPR) and still save lifes (FNR).

1. Since the symptoms will eventually appear, and the patient is not in life-risking situation, we will want to optimaize the money saving, i.e. lower FPR.
2. In this case, we want to focus on saving the patients life, even if it will cost us money, so we would like to have lower FNR.

"""


part3_q4 = r"""
**Your answer:**
1. Explain the decision boundaries and model performance you obtained for the columns (fixed `depth`, `width` varies).
2. Explain the decision boundaries and model performance you obtained for the rows (fixed `width`, `depth` varies).
3. Compare and explain the results for the following pair of configurations, which have the same number of total parameters:
    - `depth=1, width=32` and `depth=4, width=8`
3. Explain the effect of threshold selection on the validation set: did it improve the results on the test set? why?

1. When the depth of the network is fixed and the width is increasing, the decision boundaries tend to get more complex in order to fit the data better, which result in better performance. This is because that with higher neuron count in each layer, the network can learn more complex connections from our data.

2. Now with fixed number of neurons in the layers of the network but the number of layer increases, we also perform more non linear activition layers to our data, which results lower 

3. (A) depth=1, width=32 VS (B) depth=4, width=8

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = torch.nn.CrossEntropyLoss()  # One of the torch.nn losses
    lr, weight_decay, momentum = 0.05, 0.001, 0.1  # Arguments for SGD optimizer

    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**

1. Number of parameters:
For the ResidualBlock we have $256*3*3*256 = 589,824$ as the first layer + $256*3*3*256$ = 589,824 as the second layer, thus total of: 1,179,648 parameters.
For the ResidualBottleneckBlock we have $ 64*1*1*256 = 16,384$ as the first layer + $64*3*3*64$ = 36,864 as the second layer + $256*1*1*64 = 16,384$ as the third layer, thus total of: 69,632 parameters. \n
We can see that the ResidualBottleneckBlock has much less parameters than the ResidualBlock.

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
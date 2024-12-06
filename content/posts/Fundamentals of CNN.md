---
title: Fundamentals of CNN
draft: true
tags:
---
 Discrete convolution is a linear transformation that preserves the notion of ordering, it is sparse (only few input units contribute to given output unit) and reuses parameters (same weights applied to multiple locations in the input) $\rightarrow$ A kernel slides across the input feature map. At each location, product between each element of the kernel and input element it overlaps is computed and the results are summed up to obtain the output in the current location. If there are multiple feature maps (i.e. R.G.B.), the kernel will have to be 3-dimensional / each one of the feature maps will be  convolved with a distinct kernel - and the resulting feature maps will be summed up elementwise to produce the output feature map. 

### Pooling
This operation reduces the size of feature maps by using some function to summarize subregions, such as taking the average of the maximum value. Pooling works by sliding a window across the input and feeding the content of the window to a pooling function.

### Output Feature Map Size
Given the following symbols:
$$
\begin{gathered} 
i = input \space size\\
k = kernel \space size\\
s = stride \space size\\
p = padding \space size\\\\
\text{For any i and k, and for s = 1 and p = 0:}\\
o = (i - k) + 1 \\\\
\text{For any i and k, and for s = 1:}\\
o = (i - k) + 2p + 1 \\\\
\text{For any i and for k odd (k = 2n+1, }n \in \mathbb{N}) \text{, s = 1 and p = [k/2] = n:} \\
o = i + 2[k/2] - (k - 1)\\
= i + 2n - 2n \\
= i \\\\
\text{For any i and k, and for p = k - 1 and s = 1:} \\
o = i + 2(k - 1) - (k - 1) \\
= i + (k - 1) \\\\
\text{For any i, k and s, and for p = 0:} \\
o = \begin{bmatrix} \frac{i-k}{s} \end{bmatrix} + 1 \\\\
\text{For any i, k, p and s:} \\
o = \begin{bmatrix} \frac{i + 2p - k}{s} \end{bmatrix} + 1
\end{gathered}
$$
For pooling arithmetic, since pooling does not involve zero padding, the following relationships holds for any type of pooling:
$$ 
\begin{gathered}
\text{For any i, k and s:} \\
o = \begin{bmatrix} \frac{i - k}{s} \end{bmatrix} + 1
\end{gathered}
$$

### Transposed Convolution
The need for transposed convolution arises from the desire to use a transformation going in the opposite direction of a normal convolution. This may be useful as a decoding layer of a convolutional autoencoder or to project feature maps to a higher-dimensional space.
Also called fractionally strided convolutions or deconvolutions, transposed convolution work by swapping the forward and backward passes of a convolution. 
$$  
\begin{gathered}
\text{Convolution described by s = 1, p = 0, and k has an associated transposed} \\ 
\text{convolution described by k' = k, s' = s and p' = k - 1 and its output size is:} \\
o' = i' + (k - 1) \\
\end{gathered}
$$

### Batch Normalization
This can be applied to individual or all layers, but initially, the inputs are normalized by getting a noisy estimate of the mean and standard deviation using the current minibatch. Specifically, the input element is subtracted by the estimate mean and then divided by the standard deviation, this is followed by a scale coefficient and offset that helps recover some of the lost degrees of freedom. 
$$
\textrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}. 
$$
$$ 
\hat{\boldsymbol{\mu}}_\mathcal{B} = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x}
\textrm{ and }
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.
$$
Batch normalization runs differently in training and inference modes, in training the normalization is done using minibatch statistics, during inference, the whole dataset is used to find the mean and standard variation for normalization.

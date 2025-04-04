# USD MSAAI-590 Capstone Project
The purpose of this project is to demonstrate the capability and features of the [TurbaNet](https://github.com/EthanSchmitt7/TurbaNet) library developed for my capstone project in pursuit of the MSAAI program at USD.

## TurbaNet
TurbaNet is a neural network training library designed to optimize the simultaneous training of multiple small-scale models. Unlike traditional deep learning frameworks like PyTorch, which typically optimize for large, single-model training, TurbaNet leverages parallel computation to significantly improve efficiency when training multiple networks at once.

This approach is particularly beneficial in ensemble learning, where multiple models are trained on different subsets of data to improve generalization and robustness. By training all models concurrently, TurbaNet reduces the overhead of sequential training, leading to substantial speedups while maintaining model diversity. This is especially useful in scenarios where individual models are too small to fully utilize modern GPUs, making batch-style parallelism more efficient than standard single-network training.

The diagram helps to illustrates of how small network sizes can be inefficient on modern GPU architectures. Large models take advantage of the mass compute capabilities but small networks are not able to leverage it nearly as effectively. JAX and TurbaNet can solve this by vectorizing the computations from multiple models. The first diagram demonstrates a single small network which does not require enough of the bandwidth available and so it loses out on that performance. Next is what is would look like if one naively attempted to use multiple models to utilize the available bandwidth. The networks end up being run in serial and no performance gain is realized. Finally, with JAX and TurbaNet, multiple networks can be evaluated concurrently and are able to use the full width of the GPUs compute capabilities.

![TurbaNet Architecture](images/turbanet_architecture.png)

Furthermore, TurbaNet excels in cases where multiple independent models need to learn from different data distributions, such as in federated learning, bootstrapped ensembles, or multi-agent reinforcement learning. By structuring training in a way that maximizes hardware utilization and minimizes redundant computations, TurbaNet ensures faster convergence without sacrificing the advantages of ensemble diversity.

## JAX GPU Limitations
The [JAX](https://github.com/jax-ml/jax) library which TurbaNet is built on, does not support GPUs in any environment and certain configurations will enable GPU support. For reference, see [this page](https://docs.jax.dev/en/latest/installation.html) of their documentation to see if your setup will support GPU computations. As shown in the results below, TurbaNet performs best when taking advantage of GPU computing, CPUs are not able to take advantage of vectorization as well.

# Parametric Analysis
To first demonstrate the TurbaNet library was performing as expected, a parametric study was conducted which aimed to investigate the effect on runtime as the size of the swarm increased. The expectation is that PyTorch would scale linearly with the number of networks with a similar scaling to doubling the swarm size would double the total training time. With TurbaNet the expectation is that it should hardly scale with the swarm size until the GPU's memory limit is exhausted at which point we should see it begin to scale as well.

## GPU[^1] Swarm Sweep Results
Presented below are the results of a sweep of swarm sizes from a single network up to 128 networks being trained in parallel. The networks are trained on a two spiral classification sample problem where the input is a pair of floats and the output is another pair of floats. The training was done with the same number of epochs (1024) and batch size (128). For these results in particular, no loss or accuracy curves are shown because the point is to investigate the training time performance of the models. A performance comparison is performed in later sections on more common machine learning metrics.

### Small Network
The first chart shows the analysis done on a network with 16 hidden nodes and only a single hidden layer. The library was built for small networks like this, so we can see the performance is exceptional. TurbaNet training time does not appear to scale at all as the number of networks increases whereas PyTorch clearly shows the expected trend of doubling networks, doubles the training time.
![Small Network GPU Sweep](images/gpu_sweep_small.png)

### Large Network
The next chart below is the same experiment performed for a larger network, 256 hidden nodes and 2 hidden layers, which takes considerable amounts more memory and compute. While PyTorch retains a highly similar trend to the previous experiment, TurbaNet exhibits some more interesting behavior. We can see that around a swarm size of around 60 the GPU appears to get saturated and the runtime begins to increase. It seems that even after this point, TurbaNet scales much more slowly than PyTorch (~50x slower).

![Large Network GPU Sweep](images/gpu_sweep_large.png)

## CPU[^2] Swarm Sweep Results
The same sweep was performed but forcing the calculation onto the CPU. The networks have the same parameters as described in the GPU section and the other training parameters were also identical. Here we can see that the vectorization does not yield nearly the same benefit as expected since CPUs are not designed with the same philosophy as GPUs which are meant to calculate massive amounts of data as fast as possible. Single Instruction-Multiple Data (SIMD) instructions allows CPUs to execute some vectorized operations but this is at an exponentially smaller scale to GPUs.[^3]

### Small Network
For the small network test it can see the TurbaNet execution is able to retain some vectorization up to around 40 networks at which point it begins to show signs of the linear scaling return. Again, it is worth pointing out that while it does indeed begin to scale, it scales slower than PyTorch (~17x slower).
![Small Network CPU Sweep](images/cpu_sweep_small.png)

### Large Network
For large network sizes, the performance begins to degrade rapidly and is outclassed by PyTorch which makes sense because PyTorch was built with considerations for large networks. Here, TurbaNet shows a linear increase in time as the swarm size grows from the very start and in fact shows a steeper increase than PyTorch which is the opposite of what was seen in previous experiments. Clearly, the library is coming up against the bounds of the hardware and is not able to effectively parallelize and is actually losing performance by trying to vectorize the calculation across the swarm.
![Large Network CPU Sweep](images/cpu_sweep_large.png)

# MNIST Ensemble  Model
The experiments on the timing are useful but it is also desirable to see actual performance of the library in real world situations. A classic problem to test algorithmic performance is the MNIST digit classification dataset. Both libraries use the same model architecture, 2 hidden layers with 64 and 32 nodes respectively and ReLu activations between them. A PyTorch model is trained for a baseline of performance and compared to a swarm of TurbaNet models. Training a swarm of TurbaNet models on the same data would result in incredibly similar models that would provide no real benefit over a single model, so instead the swarm of trained models can be used to create an ensemble model which averages the prediction from each of the constituents. Each of the models can be trained on a portion of the dataset and while it may be a weaker predictor on its own, combined they can be stronger.

To compare performance, below are the confusion matrices of the PyTorch model, the average confusion matrix of all the TurbaNet models in the swarm, and finally the confusion matrix of the ensemble model.

![Confusion Matrices](images/confusion_matrices.png)

Below are differences taken between the confusion matrices to more clearly show the areas where the ensemble performs over the others. The left hand matrix is comparing the ensemble model to the average of the models. Ideally, the values on the diagonal should be positive and as high as possible and all terms of the diagonal should be negative and as low as possible. This corresponds to improving correct predictions of the digits and reducing the incorrect predictions. Somewhat unsurprisingly, the ensemble model outperforms the average quite severely but it also shows significant improvement over the PyTorch Model.

![Differences of Confusion Matrices](images/difference_of_confusion_matrices.png)

It is clear that the ensemble model performs better than the PyTorch model however it takes much longer to train. However, this experiment shows that this library can be levereged in different ways. In some use cases it may reduce runtime and in some others it may be used to increase model performance.

| Model       | Training Time (s) | Accuracy  | Precision | Recall    | F1-Score  |
| ----------- | ----------------- | --------- | --------- | --------- | --------- |
| PyTorch     | **43.62**         | 92.9%     | 92.8%     | 92.8%     | 92.8%     |
| Turba (Avg) | 124.79            | 93.8%     | 93.7%     | 93.7%     | 93.7%     |
| Turba (Ens) | ----------------- | **94.4%** | **94.4%** | **94.4%** | **94.4%** |

# Time Series Stock Prediction
The final performance analysis was to try the capability of TurbaNet to train model simultaneously on different data to learn different trends. Take for example the problem of predicting a stock price over time. This is a well studied problem and LSTMs tend to be an acceptable solution, so network architecture was used for this example. The problem comes when it is desired to predict 10s, 100s, or even 1000s of different stocks. With PyTorch, one would have to train each model individually but with TurbaNet, the models can be trained simultaneously leading to a massive performance boost.

The chart below displays the distribution of the average errors across the testing data of the PyTorch and TurbaNet models for the 100 stocks that they were trained on. The analysis is admittedly rather qualitative but it shows that the TurbaNet models achieve relatively similar performance to the PyTorch models.
![Average Stock Prediction Error](images/average_stock_prediction_errors.png)

Where the results really get interesting is when the runtime of both these training sessions is examined. In this case PyTorch takes roughly 40x longer to train all of the models as compared to TurbaNet. This experiment didn't seem to be stressing the limits of the GPU so it implies that this could be extended even further to train on even more stocks at once.
| Model       | Training Time (s) | Time per Stock (s) |
| ----------- | ----------------- | ------------------ |
| PyTorch     | 9560.44            | 19.12088          |
| Turba       | **224.51**         | **0.44902**       |


## Conclusion
Overall, the exact bounds of where TurbaNet hits its boundaries and where to exactly define a "large network" is left intentionally vague as it depends almost entirely on the hardware being trained on. Future experiments could explore the impact of higher VRAM or larger bandwidth GPUs to determine the more significant factors. The parametric sweep has undoubtedly identified that TurbaNet is capable of outperforming state of the art libraries like PyTorch when it comes to training large numbers of relatively small models. When applying TurbaNet to real life problems, key observations were made. It can be used to improve model accuracy through common bootstrapping/ensemble approaches perhaps at the cost of increased training time, and it can be used to drastically reduce runtime when training on data that requires many independent models that are trained on different datasets. The impact is with this library, the full power of accelerator hardware can be accessed even for more simple problems that do not require large model architectures. The effectiveness of this methodology theoretically scales proportionally with hardware advances (memory and compute) so there is good reason to believe that this will only become more effective as GPUs and other accelerators advance technologically.

[^1]: NVIDIA RTX 2070 Super (8 GB VRAM)
[^2]: AMD Ryzen 7 3700X 8-Core Processor

References
[^3]: Yi, X. (2024). A study of performance programming of CPU, GPU accelerated computers and SIMD architecture. arXiv. https://doi.org/10.48550/arXiv.2409.10661
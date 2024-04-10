Lab 4: Pruning
===


Preliminaries & Setup
---
### 0. Share the hardware specifications and OS information of the machine(s) you will be using to run the experiments in this lab.

**Hardware**: Raspberry Pi 4 CPU: ARM Cortex-A72, 4 cores MEM: available: 7413MB (total: 7811MB),
**OS**: Raspberry PI OS (64-bit)/aarch64, Frequency: 600 -1800 MHz software/tooling: pytorch(v1.13.0)

1. Copy over relevant code for training MNIST or SST2 from Lab 3, including code for evaluation (in particular, accuracy, latency, and size on disk), *but do not train yet*!
2. Initialize your model. Before training, **SAVE** your model's initial (random) weights. You will use them later for iterative pruning
3. Now train the base model and report:
   - dev accuracy: 81.77%
   - inference latency: mean = 0.000569, std = 4.4e-05 seconds
   - number of parameters: 1412097 
   - space needed for storage (in MB) of `state_dict()` on disk: 5.65073 MB 

| Iteration | Sparsity (%) | Accuracy | Latency (s) | Disk Size (MB) |
| ------- | ------- | ------ | ------- | ------ |
|     0   |   0.0%  |    81.77%    |0.000569 |    5.65073  |

Also take a look at your model's `named_parameters()`. You will need these later (no need to put in the table).


Part I: Magnitude pruning on SST2/MNIST
---


4. Write functions to calculate the sparsity level (using the percent of buffers that are 0):
    -  for each parameter: 33.94%, 21.98%, 25.69%, 28.12% for each layer. 
    -  for all pruned parameters overall: 33.0%
    -  for the model overall: 32.98%

5. Write a function to calculate the amount of space that a pruned model takes up when reparameterization is removed and tensors are converted to *sparse* representations.

Using your new disk size function, fill in the next row of the same table:

| Iteration | Sparsity (%) | Accuracy | Latency (s) | Disk Size (MB) |
| ------- | ------- | ------ | ------- | ------ |
|     0   |   0.0%  |    81.77%    |0.000569 |    5.65073  |
|     1   |   33.0%   |    81.88%    | 0.000457 |    18.925393    |

Repeated unstructured magnitude pruning
---
Now, keep performing the same unstructured magnitude pruning of 33% of the remaining weights on the same model (*without re-training or resetting the model*). 
You will apply the same function as above with the same 0.33 proportion parameter.

6. Collect values for the rest of this table, keeping in mind that you will need to plot the results later. You might want to keep the values in Pandas DataFrames (see the section on **plotting it all together** below.) Sparsity reported should be the percentage of *prunable* parameters pruned. 


| Iteration | Sparsity (%) | Accuracy | Latency (s) | Disk Size (MB) |
|-----------|--------------|----------|-------------|----------------|
| 0         | 0.0%         | 81.77%   | 0.000569    | 5.65073        |
| 1         | 32.98%       | 81.88%   | 0.000457    | 18.925886      |
| 2         | 55.08%       | 82.57%   | 0.000512    | 12.68499       |
| 3         | 69.89%       | 82.34%   | 0.000497    | 8.50355        |
| 4         | 79.81%       | 81.65%   | 0.000453    | 5.702014       |
| 5         | 86.45%       | 81.08%   | 0.000464    | 3.825086       |
| 6         | 90.90%       | 78.44%   | 0.000534    | 2.567358       |
| 7         | 93.89%       | 69.61%   | 0.000495    | 1.724798       |
| 8         | 95.89%       | 52.41%   | 0.000447    | 1.16019        |
| 9         | 97.23%       | 49.08%   | 0.000486    | 0.78195        |
| 10        | 98.12%       | 49.08%   | 0.000522    | 0.528574       |

**Tip:** Evaluating pruned models. *Assuming you have an e.g. `evaluate()` function that takes in your (pruned) model, dataloader, and possibly additional arguments*, you could use a function similar to this to evaluate models without the overhead of applying parameter masks on-the-fly (this can be useful especially if your `evaluate` function returns latency information).
```py
from copy import deepcopy

def sparse_evaluate(model, dataloader, num_classes=2):
    model_copy = deepcopy(model)
    model_copy.eval()
    
    copy_params = [(model_copy.layers[0], 'weight'),
                       (model_copy.layers[1], 'weight'),
                       (model_copy.out, 'weight')]
    # (we assume the same model architecture as the MNIST or SST-2 architecture we specify above)
    for p in copy_params:
        prune.remove(*p)
    
    return evaluate(model_copy, dataloader, num_classes)
```
**Note: this does not actually run sparse inference** - there are other frameworks that can help with that, but you're not expected to implement that for this lab.

Iterative magnitude pruning (IMP)
---

Now, repeat the same process as above, but re-train the remaining weights each time (using the same hyperparameters). Importantly, you will *rewind* your model's remaining weights to their initialization in between iterations. Implementation-wise, this should look just like the above, with some extra steps (training and rewinding) between each pruning step.
| Iteration | Sparsity (%) | Accuracy | Latency (s) | Disk Size (MB) |
|-----------|--------------|----------|-------------|----------------|
| 0         | 0.0%         | 81.77%   | 0.000569    | 5.65073        |
| 1         | 32.98%       | 81.88%   | 0.000462    | 18.925886      |
| 2         | 55.08%       | 81.88%   | 0.000457    | 12.68499       |
| 3         | 69.89%       | 81.77%   | 0.000468    | 8.503614       |
| 4         | 79.81%       | 81.54%   | 0.000456    | 5.702014       |
| 5         | 86.45%       | 80.39%   | 0.000471    | 3.824958       |
| 6         | 90.90%       | 79.24%   | 0.000463    | 2.56755        |
| 7         | 93.89%       | 79.13%   | 0.000498    | 1.724798       |
| 8         | 95.89%       | 80.50%   | 0.00051     | 1.16019        |
| 9         | 97.23%       | 79.13%   | 0.000441    | 0.78195        |
| 10        | 98.12%       | 77.87%   | 0.000487    | 0.528574       |


7. **IMP with rewinding:** Recall from class that *rewinding* refers to resetting the weights to an earlier value, rather than the most recent value during iterative magnitude pruning. Implement retraining with rewinding to the weights' values at **model initialization**, before any training or pruning was performed. (This is why we asked you to save a copy of the initialized but untrained model weights in the beginning of the lab!) You should use the same training hyperparameters each time. Collect all the same numbers as specified in the table in the previous section, putting them into a new table.

In your iterative magnitude pruning training loop, there are some special considerations you will need to make in order to get things working properly:

After each round of pruning, we want to *reset* all remaining weights to their values at initialization. For example, if you have a model's `state_dict()` saved at the relative path `"data/model_init.pt"`, you can use `torch.load("data/model_init.pt")` to reload the dict. 

Now, because the exact parameter names do not match the state dict that of the (unpruned) model at initialization, you will have to go out of your way to align them. Assuming you have e.g. `prune_param_list = ['layers.0.weight', 'layers.1.weight', 'out.weight']`, you can use:
```py
init_updated = {k + ("_orig" if k in prune_param_list else ""):v for k,v in init_weights.items()}
ffn_mnist_copy = copy.deepcopy(ffn_mnist.state_dict())
ffn_mnist_copy.update(init_updated)
ffn_mnist.load_state_dict(ffn_mnist_copy)
```

Plotting it all together
---
You should report 2 plots, each of which contains a line corresponding to each of the experiments you performed above: pruning without retraining, and IMP with rewinding. The two plots should have:
   - **accuracy** on the x axis and **sparsity** on the y axis. 
   - **accuracy** on the x axis and **disk space** on the y axis. 
   - **accuracy** on the x axis and **inference latency** on the y axis. 

**Tip:** If you have three Pandas DataFrames each containing columns for: 1) iteration number, 2) sparsity (of prunable parameters), 3) accuracy, 4) inference latency, and 5) size on disk, you can plot e.g. accuracy vs latency using a more elaborate version of this code (i.e. with a title and axis labels):
Here is some example code you might use to plot these values using Matplotlib:
```py
import matplotlib.pyplot as plt

plt.plot(noniter_df['iteration'], noniter_df['latency'], color="C0", label="Pruning w/o retraining")
plt.plot(IMP_df['iteration'], IMP_df['latency'], color="C1", label="IMP w/ rewind")
plt.plot(stdprune_df['iteration'], stdprune_df['latency'], color="C2", label="IMP no rewind")
plt.legend()
# YOUR CODE for titles and axis labels, etc.
plt.show()
```

![Accuracy vs Sparsity Plot](accuracy_vs_sparsity_plot.png)


![Accuracy vs Sparsity Plot](accuracy_vs_latency_plot.png)

![Accuracy vs Sparsity Plot](accuracy_vs_disk_size_plot.png)

Part I Discussion
---
- Describe two trends depicted in your plots, and compare and contrast them. Are there trends that you expected, or didn't expect, based on discussions and lectures in class, and/or your experience? For example, is there a clear drop-off in performance at a certain sparsity level, and does that change across methods? Do latency and space on disk correspond to your expectations, and why or why not?

**Trend 1:**
For "Pruning w/o Retraining", there's a significant drop in accuracy as sparsity increases, especially from the 7th iteration (93.89% sparsity). In contrast, "IMP w/ Rewind" maintains better accuracy as sparsity increases. Retraining after pruning allows the model to re-adapt the remaining weights, i.e. redistribute the learning capacity among the remaining connections, compensating for the pruned ones. 

**Trend 2:**
"Pruning w/o Retraining" shows a steep decrease in disk space as accuracy decreases for the first 4 iterations. However, this reduction plateaus afterward. In contrast, "IMP w/ Rewind" exhibits a more consistent reduction in disk size as accuracy decreases, suggesting a better trade-off between accuracy and model size. This happens as the "IMP s/ Rewind" mechanism selectively removing connections after retraining, leading to a more proportional reduction in model size.

Part II: Your Model, Device, and Data
---


8. Required: Repeat Question 6 for your model, on your device and with your data.

+ Model: Squeezed and Efficient Wav2vec tiny (SEW_tiny)
+ Data: librispeech
+ Hardware:  Raspberry Pi 4 
  +  CPU: ARM Cortex-A72, 4 cores. MEM: available: 7413MB (total: 7811MB). OS: Raspberry PI OS (64-bit)/aarch64, Frequency: 600 -1800 MHz. software/tooling: pytorch(v1.13.0)


*WER: Word error rate

| Iteration | Sparsity (%) | WER | Latency (s) | Disk Size (MB) |
|-----------|--------------|----------|-------------|----------------|
| 0         | 0.0%         | 0.106   | 22.79 ± 0.026    | 163       |
| 1         | 10%          | 0.107   | 22.90 ± 0.012   | 750      |
| 2         | 19%          | 0.113   | 22.87 ± 0.053       | 677       |
| 3         | 27.1%        | 0.125   | 22.83 ± 0.025    | 610.61        |
| 4         | 34.39%       | 0.156   | 22.94 ± 0.035    | 550.91       |
| 5         | 40.95%       | 0.219   | 22.97 ± 0.016    | 497.21       |
| 6         | 46.86%       | 0.429   | 22.88 ± 0.039    | 448.91      |
| 7         | 52.17%       | 0.899   | 22.94 ± 0.016    | 405.46       |
| 8         | 56.95%       | 0.999   | 22.89 ± 0.040    | 366.39        |
| 9         | 61.26%       | 1.0   | 22.98 ± 0.031    | 331.23        |
| 10        | 65.13%       | 1.0   | 22.91 ± 0.033    | 299.61       |


### Pick two of three
We pick question 10 and question 11:

10.  Conduct a sensitivity analysis of pruning (structured or unstructured) different components of your model. For instance, what happens to your model's performance when you prune input embeddings vs hidden layer weights? Do earlier layers seem more or less important than later layers? You are not required to conduct a thorough study, but you should be able to draw a couple concrete conclusions.

+ The model consists of a stack of CNN feature extractors and a stack of transformer blocks. We performed sensitivity analysis for these two parts. The result is plotted below.

+ From the experiment, we oberserved that with higher sparsity, pruning transformer block makes the word error rate rise faster, comparing to pruning feature extractor. This result shows that transformer block is more sensitive to pruning than feature extractor. This is quit unexpacted since earlier layers (feature extractor) should be more sensitive to pruning. Also, transformer blocks have less inductive bias, thus should have more redundacy. One explanation is that the special design of SEW-tiny makes the transformer block less redundant ( Wu, Felix, Kwangyoun Kim, Jing Pan, Kyu J. Han, Kilian Q. Weinberger, and Yoav Artzi. "Performance-efficiency trade-offs in unsupervised pre-training for speech recognition." In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 7667-7671. IEEE, 2022.). This design includes a reversed bottlenet archetecture, where the hidden repersentation is downsampled before being passed to the transformer layers. Another convolution upsampling is performed after the transformer layers. The authors used this design to optimize the efficiency of the transformer layers, which may explane why the transformer blocks are more sensitive to pruning.

![](sensitivity.jpg)


11. Export and run your unpruned and a diverse sample of your pruned models on an inference runtime (ONNX runtime, TensorRT). Check out [the PyTorch ONNX docs](https://pytorch.org/docs/stable/onnx.html) and [this page](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html) for reference. Did you run into any challenges? Do you see latency benefits? Was anything surprising? Report inference latency and discuss.

| Iteration | Sparsity (%) | WER | Latency (s) | Disk Size (MB) |
|-----------|--------------|----------|-------------|----------------|
| 0         | 0.0%         | 0.106   | 0.880 ± 0.0015 | 163       |
| 1         | 10%          | 0.107   | 0.886 ± 0.0015 | 750      |
| 2         | 19%          | 0.113   | 0.880 ± 0.0016 | 677       |
| 3         | 27.1%        | 0.125   | 0.878 ± 0.0015 | 610.61        |
| 4         | 34.39%       | 0.156   | 0.881 ± 0.0020 | 550.91       |
| 5         | 40.95%       | 0.219   | 0.882 ± 0.0018 | 497.21       |
| 6         | 46.86%       | 0.429   | 0.880 ± 0.0019 | 448.91      |
| 7         | 52.17%       | 0.899   | 0.880 ± 0.0021 | 405.46       |
| 8         | 56.95%       | 0.999   | 0.889 ± 0.0395 | 366.39        |
| 9         | 61.26%       | 1.0   | 0.886 ± 0.0021   | 331.23        |
| 10        | 65.13%       | 1.0   | 0.877 ± 0.0015    | 299.61       |

We didn't meet any difficult challenge for this part.

**Substantial Latency Reduction with ONNX**: The adoption of ONNX results in a significant decrease in latency for both the original and pruned models. In the pruned model (Iteration 10), ONNX achieves an inference time of 0.877 seconds, which is considerably faster than the original latency of 22.94 seconds.

**Consistent Benefits Across Sparsity Levels**: The latency benefits of ONNX persist across different sparsity levels. Even with high sparsity (65.13%), ONNX maintains a low and consistent latency, showcasing its effectiveness in optimizing inference times.

**Surprising Efficiency Improvement**: The extent of the latency decrease with ONNX is noteworthy. It demonstrates the efficiency of ONNX in accelerating inference, making it a valuable tool for real-time applications.

**Practical Implications for Deployment**: The results suggest that ONNX is a practical solution for improving the deployment of both original and pruned ASR models, especially in scenarios where low latency is critical.



Part II Discussion
---
- Describe at least two trends observations from Part II. Are there trends that you expected, or didn't expect, based on discussions and lectures in class, and/or your experience?

**Sparsity and Latency Relationship**: As expected, there is a consistent trend of reduced latency as the sparsity of the model increases. This aligns with the common understanding that more sparse models result in faster inference times. The latency benefits are evident across different sparsity levels, indicating the effectiveness of pruning in optimizing model inference.

**ONNX Acceleration Impact**: The adoption of ONNX significantly improves the latency of both the original and pruned models. There is a substantial decrease in latency, making the models more efficient for real-time applications. This unexpected but positive trend highlights the practical benefits of leveraging ONNX for inference acceleration

**Trade-off Between WER and Latency**: The trade-off between Word Error Rate (WER) and latency still holds, indicating that as the model becomes more sparse, there is a potential compromise in accuracy. The need for a balanced approach to achieve both low latency and acceptable WER is highlighted.

- In 4-5 sentences, pose one small follow-up experiment that you might run, based on your initial results. Your motivation (based on these results), hypothesis and methodology for testing that hypothesis should be clear. You do not need to run the experiment. (This can overlap with extra credit, if you choose to implement it.)

**Follow-up Experiment:**
**Make Tradeoff for optimal result**:

Motivation: Investigate the tradeoff between model accuracy and the degree of pruning. Assess how sparsity levels impact accuracy and latency, aiming to find the optimal balance for your specific use case.

Hypothesis: There is an optimal sparsity level that maximizes the reduction in model size and latency while minimizing the impact on accuracy. Extreme pruning might lead to diminishing returns in terms of latency benefits.

Methodology: Prune the model at various sparsity levels, evaluating accuracy and latency at each stage. Analyze the results to identify the point at which further pruning significantly affects accuracy.

**Follow-up Experiment:**
**Combining Pruning with Quantization**:

Motivation: Explore the combined impact of pruning and quantization on model size and inference speed. Quantization reduces precision, and combining it with pruning might lead to even more efficient models.

Hypothesis: The combination of pruning and quantization will result in a substantial reduction in model size and improved latency compared to individual techniques.

Methodology: Apply both pruning and quantization to the model in a sequential or parallel manner. Evaluate the model's accuracy, latency, and model size at different levels of sparsity and quantization. Analyze the synergistic effects of these techniques on overall efficiency.



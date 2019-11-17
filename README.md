# Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks

### BEFORE YOU RUN OUR CODE
We appreciate your interest in our work and trying out our code. We've noticed several cases where incorrect configuration leads to poor performance of detection and mitigation. If you also observe low detection performance far away from what we presented in the paper, please feel free to open an issue in this repo or contact any of the authors directly. We are more than happy to help you debug your experiment and find out the correct configuration. Also feel free to take a look at previous issues in this repo. Someone might have ran into the same problem, and there might already be a fix.

### ABOUT

This repository contains code implementation of the paper "[Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks](https://cs.ucsb.edu/~bolunwang/assets/docs/backdoor-sp19.pdf)", at *IEEE Security and Privacy 2019*. The slides are [here](https://www.shawnshan.com/publication/backdoor-sp19-slides.pdf). 

### DEPENDENCIES

Our code is implemented and tested on Keras with TensorFlow backend. Following packages are used by our code.

- `keras==2.2.2`
- `numpy==1.14.0`
- `tensorflow-gpu==1.10.1`
- `h5py==2.6.0`

Our code is tested on `Python 2.7.12` and `Python 3.6.8`

### HOWTO

#### Injecting Backdoor 

For the GTSRB model, the backdoor injection code is under the [injection repo](https://github.com/bolunwang/backdoor/tree/master/injection). 
You will need to download the training data from [here](https://drive.google.com/file/d/1kcveaJC3Ra-XDuaNqHzYeomMvU8d1npj/view?usp=sharing). 
#### Reverse Engineering

We include a sample script demonstrating how to perform the reverse engineering technique on an infected model. There are several parameters that need to be modified before running the code, which could be modified [here](gtsrb_visualize_example.py#L25-L27).

- **GPU device**: if you are using GPU, specify which GPU you would like to use by setting the [DEVICE](gtsrb_visualize_example.py#L29) variable
- **Data/model/result folder**: if you are using the code on your own models and datasets, please specify the path to the data/model/result files. They are specified by variables [here](gtsrb_visualize_example.py#L31-L37).
- **Meta info**: if you are testing it on your own model, please specify the correct meta information about the task, including [input size](gtsrb_visualize_example.py#L40-L42), [preprocessing method](gtsrb_visualize_example.py#L48), [total # of labels](gtsrb_visualize_example.py#L45), and [infected label](gtsrb_visualize_example.py#L46) (optional).
- **Configuration of the optimization**: There are [several parameters](gtsrb_visualize_example.py#L50-L67) you could configure for the optimization process, including learning rate, batch size, # of samples per iteration, total # of iterations, initial value for weight balance, etc. Most parameters fit all models we tested, and you should be able to use the same configuration for your task as well.

To execute the python script, simply run

```bash
python gtsrb_visualize_example.py
```

We already included a sample of [infected model](models/gtsrb_bottom_right_white_4_target_33.h5) for traffic sign recognition in the repo, along with the [testing data](data/gtsrb_dataset_int.h5) used for reverse engineering. The sample code uses this model/dateset by default. The entire process for examining all labels in the traffic sign recognition model takes roughly 10 min. All reverse engineered triggers (mask, delta) will be stored under RESULT_DIR. You can also specify which labels you would like to focus on. You could configure it yourself by changing the [following code](gtsrb_visualize_example.py#L200-L201).

#### Anomaly Detection

We use an anomaly detection algorithm that is based MAD (Median Absolute Deviation). A very useful explanation of MAD could be found [here](https://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers/). Our implementation reads all reversed triggers and detect any outlier with small size. Before you execute the code, please make sure the following configuration is correct.

- **Path to reversed trigger**: you can specify the location where you put all reversed triggers [here](mad_outlier_detection.py#L19-L20). Filename format in the sample code is consistent with previous code for reverse engineering. Our code only checks if there is any anomaly among reversed triggers under the specified folder. So be sure to include all triggers you would like to analyze in the folder.
- **Meta info**: configure the correct meta information about the task and model correctly, so our analysis code could load reversed triggers with the correct shape. You need to specify the [input shape](mad_outlier_detection.py#L23-L25) and the [total # of labels](mad_outlier_detection.py#L28) in the model.

To execute the sample code, simple run

```bash
python mad_outlier_detection.py
```

Below is a snippet of the output of outlier detection, in the infected GTSRB model (traffic sign recognition).

```
median: 64.466667, MAD: 13.238736
anomaly index: 3.652087
flagged label list: 33: 16.117647
```

Line #2 shows the final anomaly index is 3.652, which suggests the model is infected. Line #3 shows the outlier detection algorithm flags only 1 label (label 33), which has a trigger with L1 norm of 16.1.






































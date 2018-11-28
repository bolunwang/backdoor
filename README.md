# Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks
### ABOUT

This repository contains code implementation of the paper "[Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks]()", at *IEEE Security and Privacy 2019*.

### DEPENDENCIES

Our code is implemented and tested on Keras with TensorFlow backend. Following packages are used by our code.

- `keras==2.2.2`
- `numpy==1.14.0`
- `tensorflow-gpu==1.10.1`
- `h5py==2.6.0`

Our code is tested on `Python 2.7.12`

### HOWTO

#### Reverse Engineering

We include a sample script demonstrating how to perform the reverse engineering technique on an infected model. There are several parameters that need to be modified before running the code, which could be modified [here](gtsrb_visualize_example.py#L25-L27).

- **GPU device**: if you are using GPU, specify which GPU you would like to use by setting the [DEVICE](gtsrb_visualize_example.py#L29) variable
- **Data/model/result folder**: if you are using the code on your own models and datasets, please specify the path to the data/model/result files. They are specified by variables [here](gtsrb_visualize_example.py#L31-L37).
- **Meta info**: if you are testing it on your own model, please specify the correct meta information about the task, including [input size](gtsrb_visualize_example.py#L40-L42), [total # of labels](gtsrb_visualize_example.py#L45), and [infected label](gtsrb_visualize_example.py#L46) (optional).
- **Configuration of the optimization**: There are [several parameters]() you could configure for the optimization process, including learning rate, 

To execute the python script, simply run

```bash
python gtsrb_visualize_example.py
```






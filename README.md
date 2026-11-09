# ANOLE

### The environment setup
Anaconda is suggested to be installed to manage the test environments.

### Prerequisites
- Linux or macOS
- Python >=3.6
- numpy, pandas
- CPU or NVIDIA GPU + CUDA CuDNN
- tensorflow >=1.15

### Overview
The reinforcement learning method corresponding to ANOLE is located in the ```ANOLE/src/simulator/``` directory.
Among them, there are many auxiliary files, such as ```env.py```, 
which is the code for simulating ABR virtual playback, and ```constants.py```, 
which contains various parameter settings.
The main logic files for training the ANOLE algorithm are located in the ```ANOLE/``` directory.

### Baseline algorithm
```Genet-main```,```merina-master```,```Netllm-master```,```pensieve-master``` are the code directories for the Genet, Merina,  Netllm and pensieve algorithms, respectively.

### Network throughput traces
Public bandwidth tracking is placed in this directory ```ANOLE/data/```


### Usage
When running ANOLE on its own, you can execute it using the ```ANOLE/src/driver/abr/ANOLE.sh``` script by running 
```
bash ANOLE.sh
```
In this script, you can set various training parameters, 
such as specifying the directory to save the results in using ```--save-dir``` , 
specifying the training set using ```--train-trace-dir``` , 
and specifying the validation set using ```--val-trace-dir``` .

Plot a example result
```
cd plt
python F_1.py
```




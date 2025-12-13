In this README file we discuss the uses of each python script, and how to run them yourself as well as the required libraries  

Required Libraries:  
1: pytorch  
  `pip install torch`  
2: numpy  
  `pip install numpy`  
3: pandas  
  `pip install pandas`  
4: psycopg2  
  `pip install psycopg2`  
 
The main scripts in our code are as follows  
1) normalize\_data.py  
  This script is used to normalize the training, validation, and test data. After this is done, rows and colummns will be deleted to create sparse data.  
  To tune this script, there are parameters at the top of the script in lines 9-16. These can be used to tune your input and output file paths, as well as the rate at which data is deleted.  
2) ConvLSTM.py  
  This script is how we implement our Convolutional LSTM. No tuning is required for this file, and you need not run it. This file is imported into training and testing scripts to make use of our model.  
3) train.py  
  This script is one of our two main training scripts. It uses a DBMS connection to save computations as well as training run metrics, such as epochs, training and validation loss, and runtime. It has many command line arguments which tune how it runs.  
`--train-csv` is a path, is required, and takes the path to the training csv you wish to use for training.  
`--val-csl` is a path, it is also requied, and takes the path to the validation csv.  
`--epochs` is an INT, it is not required, tunes the number of epochs you with to run the training for. Default value is 2  
`--future-steps` is an INT, it is not required, it sets the number of future steps you wish to train the neural network to predict. Default value is 3  
`--seq-len-in` is an INT, it is not required, it sets the number of timesteps which are inputted into the neural network to use for predicting the furture steps Default value is 9  
`--checkpoint` takes in a path. It is not required. This is the path where your model will be saved after training Default value is `checkpoint.pth`  
`--load-checkpoint` also takes in a path. It is not required, if used it will load the model given in the value for this argument and use it for further training  
`--reset-db` is an optional parameter that will drop all of the tables in the database  
`--max-ram-bytes` is an INT. There is not default. If set it will limit the amount of memory you can read from the CSV at once (only one of these chunks can be held in main memory at a time) and also limit the size of the dense tensor that is created from this csv  
`--max-vram-bytes` is an INT. There is not default. It limits the size of the dense tensor that you can pass into the GPU at once (only one chunk can be held in VRAM at a time).  
`--time-start` is a string. It is an optional parameter with no default that takes in a datetime string of the format "YYYY-MM-DD HH:MM:SS".  
When used, this argument trains only on data with timesteps >= to the value given. `--time-start` must be <= `time-end`  
`--time-end` is a string. It is an optional parameter with no default that works similarly to `--time-start`, except that it only trains on timesteps <= to the value given. `--time-end` must be >= `--time-start`  
A run of this script might look like:  
`python3 train.py --train-csv data/training_data.csv --val-csv data/validation_data.csv --epochs 4 --max-ram-bytes 134217728 --time-start "2024-01-01 00:00:00" --time-end "2024-02-01 00:00:00"`  
4: train\_no\_checkpoint\_no\_DBMS.py  
This file takes in similar arguments to train.py, with the following changes. Everything not specified is exactly the same as described in the train.py section  
`--reset-db` is not a command line argument in this script  
`--checkpoint` has a default path of checkpoint/checkpoint.pth  
There are two new command line arguments:  
`--log-file` takes in a path. It describes the path to the log .json file created. The default value is training\_log.json  
`--save-computations` takes in a path. It describes the path to the .json file that stores the computations done inside the training phase. It has no default value and if not selected computations will not be saved.

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
`--epochs` is an INT, it is not required, tunes the number of epochs you with to run the training for.  
  Default value is 2  
`--future-steps` is an INT, is not required, it sets the number of future steps you wish to train the neural network to predict  
  Default value is 3  
`--seq-len-in` is an INT, is not required, it sets the number of timesteps which are inputted into the neural network to use for predicting the furture steps  
	Default value is 9  
`--checkpoint` takes in a path, this is the path where your model will be saved after training  
 	Default value is 
  

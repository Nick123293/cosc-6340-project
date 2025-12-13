In this README file we discuss the uses of each python script, and how to run them yourself as well as the required libraries

Required Libraries:
- pytorch
  - `pip install torch`
- numpy
  - `pip install numpy`
- pandas
  - `pip install pandas`
- psycopg2
  - `pip install psycopg2`
 
The main scripts in our code are as follows
1) normalize\_data.py  
  This script is used to normalize the training, validation, and test data. After this is done, rows and colummns will be deleted to create sparse data.  
  To tune this script, there are parameters at the top of the script in lines 9-16. These can be used to tune your input and output file paths, as well as the rate at which data is deleted.
2) ConvLSTM.py  
  This script is how we implement our Convolutional LSTM. No tuning is required for this file, and you need not run it. This file is imported into training and testing scripts to make use of our model.
3) train.py  
  This script is one of our two main training scripts. It uses a DBMS connection to save computations as well as training run metrics, such as epochs, training and validation loss, and runtime. It has many command line arguments which tune how it runs.
- "--train-csv" is required, and takes the path to the training csv you wish to use for training.
- "--val-csl" is also requied, and takes the path to the validation csv.
- "--epochs"


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
  This script is used to normalize the training, validation, and test data. After this is done, rows and colummns will be deleted to create sparse data
  To tune this script, there are parameters at the top of the script in lines 9-16. These can be used to tune your input and output file paths, as well as the rate at which data is deleted.


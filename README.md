# cosc-6340-project
To-Do List:  
~~1: Create an RNN, implement into SQL~~  
2: Put input data into SQL  
3: Train our NN on the dataset, create output buffer for SQL inserts.  
~~4: Create a python script, that uses pyTorch and postgreSQL. Pytorch for computation, postgreSQL for data management in secondary storage. We want to have to query postgres for data as few times as possible, these are analagous to our I/O operations.~~  
5: Show that we can query certain parts of the database at a certain time.  
6: Show ACID properties using postgres as a log mechanism.  
7: Analyze parallelization, time complexity, create report.  

Test Dataset:  
https://huggingface.co/datasets/BUPT-PRIS-727/Weather2K  


Notes:  
1: Cluster physically reorders data in secondary storage to be physically closer based on index. https://www.postgresql.org/docs/current/sql-cluster.html  
2: We can assume that the input data will fit in main memory 
3: Maybe use
Matrix multiplication is generally vectors
Give an Hybrid and mixed data representation with vectors
Maybe sparse RNN in noSQL or any other format
We want to store tensor computations themselves in noSQL or any other format(perhaps JSON or better)

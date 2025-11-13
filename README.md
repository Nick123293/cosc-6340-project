# cosc-6340-project
To-Do List:  
1: Create an RNN, implement into SQL  
2: Put input data into SQL  
3: Create a python script, that uses pyTorch and postgreSQL. Pytorch for computation, postgreSQL for data management in secondary storage. We want to have to query postgres for data as few times as possible, these are analagous to our I/O operations. 



Notes:  
1: Cluster physically reorders data in secondary storage to be physically closer based on index. https://www.postgresql.org/docs/current/sql-cluster.html  
2: We can assume that the input data will fit in main memory  

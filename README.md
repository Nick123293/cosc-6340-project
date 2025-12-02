# cosc-6340-project
To-Do List:  
1) ~~Create an RNN, implement into SQL~~
2) Find good input data, make it sparse and put into SQL
3) Train NN on input data, getting data in chunks through SQL queries (as if the data cannot fit in main memory)
4) Output computations to SQL database in chunks (to create checkpoints, mostly done??)
5) Show how we can query SQL database to retreive checkpoints
6) Compress NN using tensor decomposition??
7) Turn Tensor products to matrix multiplication??
8) Analyze parallelization, time complexity
9) Create report


Notes:  
1) Cluster physically reorders data in secondary storage to be physically closer based on index. https://www.postgresql.org/docs/current/sql-cluster.html
2) Maybe use Matrix multiplication is generally vectors Give an Hybrid and mixed data representation with vectorsMaybe sparse RNN in noSQL or any other formatWe want to store tensor computations themselves in noSQL or any other format(perhaps JSON or better)

# IR Assignment 2
## Part A
We have successfully implemented the PageRank Algorithm in this part. 
The PageRank algorithm was implemented using Python and Numpy. We created the probability transition matrix both with and without random teleportations. 
In case of random teleportations, we used the probability of teleportations as 0.1.


### Running
It is recommended to use a virtual environment to install the dependencies. Do so using pipenv:
```
$ pipenv shell
```
Install the dependencies:
```
$ pipenv install
```

To run the code without using random teleportation, do the following:
```
$ python pagerank.py
Number of nodes: 4
Number of edges: 6
Mention each edge as a space separated list of two vertices: 
0 1
1 0
1 2
2 1
2 3
3 2

The final probabilty transition matrix will be: 
[[0.  1.  0.  0. ]
 [0.5 0.  0.5 0. ]
 [0.  0.5 0.  0.5]
 [0.  0.  1.  0. ]]
Probabilities using Numpy Function: 
[-1.  -0.5  1.   0.5]
[[-0.31622777 -0.5         0.31622777 -0.5       ]
 [ 0.63245553  0.5         0.63245553 -0.5       ]
 [-0.63245553  0.5         0.63245553  0.5       ]
 [ 0.31622777 -0.5         0.31622777  0.5       ]]
[[0.16666667 0.33333333 0.33333333 0.16666667]]
Probabilities using Power Iteration: 
[[0.16666794 0.33333206 0.33333206 0.16666794]]
```

To run the code with random teleportation, use the `--use-random-teleportation` flag while running the file:
```
IRIR ❯ python pagerank.py --use-random-teleportation                                                                             main
Using random teleportation
Number of nodes: 4
Number of edges: 6
Mention each edge as a space separated list of two vertices: 
0 1
1 0
1 2
2 1
2 3
3 2

The final probabilty transition matrix will be: 
[[0.03333333 0.9        0.03333333 0.03333333]
 [0.45       0.05       0.45       0.05      ]
 [0.05       0.45       0.05       0.45      ]
 [0.03333333 0.03333333 0.9        0.03333333]]
Probabilities using Numpy Function: 
[ 1.          0.42182527 -0.82182527 -0.43333333]
[[-0.33391096 -0.48654992  0.30945578  0.5       ]
 [-0.62330046 -0.51309763 -0.63579645 -0.5       ]
 [-0.62330046  0.51309763  0.63579645 -0.5       ]
 [-0.33391096  0.48654992 -0.30945578  0.5       ]]
[[0.1744186 0.3255814 0.3255814 0.1744186]]
Probabilities using Power Iteration: 
[[0.17441717 0.32558283 0.32558283 0.17441717]]
```

**Note that all the nodes are zero-indexed. If the number of nodes passed is 4, the maximum value of a node can be 3.**


## Part B
HITS algorithm was implemented on a directed networkx graph with web content, which contains 100 nodes and 256 directed edges.
The implementation used numpy and python to calculate the authority and hub scores by calculating the left eigenvector, i.e. the eigenvector corresponding to the largest eigenvalue.
We have used 2 methods to find the eigenvectors – using numpy.linalg.eig(), and using the power iteration method



### Running
It is recommended to use a virtual environment to install the dependencies. Do so using pipenv:
```
$ pipenv shell
```
Install the dependencies:
```
$ pipenv install
```

To run the code, do the following:
```
python hits.py
Enter query word: >? pension
Top 3 authority scores are:
  Node    Authority Score
------  -----------------
     0           0.320012
     3           0.21121
    61           0.21121
Top 3 Hub scores are:
  Node    Hub Score
------  -----------
    10     0.320012
    75     0.320012
     0     0.204815
 List of all scores:
  Node    Authority Score    Hub Score
------  -----------------  -----------
     0           0.320012     0.204815
     3           0.21121      0
    10          -0            0.320012
    75           0.128785     0.320012
    87           0.128785     0.155162
    61           0.21121      0
Authority score sum = 1.0
Hub score sum = 1.0
```

To measure the running time, do the following:
```
python hits_tca.py
Enter query word: >? world
world
base set size = 82
0.022000789642333984
```
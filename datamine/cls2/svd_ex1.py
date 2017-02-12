# exmample from https://www.youtube.com/watch?v=P5mlg91as1c
# start at https://youtu.be/P5mlg91as1c?t=9m42s
#               values taken at time:  https://youtu.be/P5mlg91as1c?t=11m22s
# users (u<n>) x movies (f<n>) ['f' for film ]
# i.e.
#    f1 f2 f3 f4 f5
# u1 1  1  1  0  0
# u5 0  2  0  4  4
A_mat = 
[
    [1,1,1,0,0],
    [3,3,3,0,0],
    [4,4,4,0,0],
    [5,5,5,0,0],
    [0,2,0,4,4],
    [0,0,0,5,5],
    [0,1,0,2,2],
    ]

# verification values
# Note: third category not really exmplained, seen as 'noise' because strength is much lower
# The U Matrix
# E.g. "user-to-concept" matrix
U_mat
[
    [0.13, 0.02,-0.01],
    [0.41, 0.07,-0.03],
    [0.55, 0.09,-0.04],
    [0.68, 0.11,-0.05],
    [0.15,-0.59, 0.65],
    [0.07,-0.73,-0.67],
    [0.07,-0.29, 0.32],
    ]

# The SIGMA matrix
# "strength" of the concepts
SIG_mat = [
    [12.4,   0,   0],
    [   0, 9.5,   0],
    [   0,   0, 1.3],
    ]

# e.g. "movie-to-concept" similarity matrix
VT_mat = [
    [0.56,  0.59,  0.56,  0.09,  0.09],
    [0.12, -0.02,  0.12, -0.69, -0.69],
    [0.40, -0.80,  0.40,  0.09,  0.09],
    ]

# thingy?
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Example of SVD from MMDS book
# see also visual explanation at:  https://www.youtube.com/watch?v=P5mlg91as1c

## VALUES from the VIDEO .... DAWG!!!
# start at https://youtu.be/P5mlg91as1c?t=9m42s
#               values taken at time:  https://youtu.be/P5mlg91as1c?t=11m22s
# users (u<n>) x movies (f<n>) ['f' for film ]
# i.e.
#    f1 f2 f3 f4 f5
# u1 1  1  1  0  0
# u5 0  2  0  4  4
def _mmds_p421():
  A_mat = np.array([
      [1,1,1,0,0],
      [3,3,3,0,0],
      [4,4,4,0,0],
      [5,5,5,0,0],
      [0,2,0,4,4],
      [0,0,0,5,5],
      [0,1,0,2,2],
      ])

  # verification values
  # Note: third category not really exmplained, seen as 'noise' because strength is much lower
  # The U Matrix
  # E.g. "user-to-concept" matrix
  U_mat = np.array([
      [0.13, 0.02,-0.01],
      [0.41, 0.07,-0.03],
      [0.55, 0.09,-0.04],
      [0.68, 0.11,-0.05],
      [0.15,-0.59, 0.65],
      [0.07,-0.73,-0.67],
      [0.07,-0.29, 0.32],
      ])

  # The SIGMA matrix
  # "strength" of the concepts
  SIG_mat = np.array([
      [12.4,   0,   0],
      [   0, 9.5,   0],
      [   0,   0, 1.3],
      ]) 

  # e.g. "movie-to-concept" similarity matrix
  VT_mat = np.array([
      [0.56,  0.59,  0.56,  0.09,  0.09],
      [0.12, -0.02,  0.12, -0.69, -0.69],
      [0.40, -0.80,  0.40,  0.09,  0.09],
      ]) 
  return(A_mat,U_mat,SIG_mat,VT_mat)
def _mmds_p420():
  A_mat = [
  [1 , 1 , 1 , 0 , 0],
  [3 , 3 , 3 , 0 , 0],
  [4 , 4 , 4 , 0 , 0],
  [5 , 5 , 5 , 0 , 0],
  [0 , 0 , 0 , 4 , 4],
  [0 , 0 , 0 , 5 , 5],
  [0 , 0 , 0 , 2 , 2],
  ]
  # =
  U_mat = [
  [.14 , 0],
  [.42 , 0],
  [.56 , 0],
  [.70 , 0],
  [0 , .60],
  [0 , .75],
  [0 , .30],
  ]
  SIG_mat = [
  [12.4 ,   0],
  [   0 , 9.5],
    ]
  U_mat = [
  [.58 , .58 , .58 ,   0 ,  0 ],
  [  0 ,   0 ,   0 , .71 , .71],
  ]
  return(
    np.array(A_mat),
    np.array(U_mat),
    np.array(SIG_mat),
    np.array(U_mat),
    )
#def compare_np_linalg(A_mat,U_mat,SIG_mat,VT_mat):
# pass method: http://stackoverflow.com/a/706735
def compare_np_linalg(runmethod):
  print("########################################")
  print("# comparing and verifying np.linalg output for %s", runmethod)
  A_mat,U_mat,SIG_mat,VT_mat = runmethod()
  """
  # numpy.linalg.svd
   numpy.linalg.svd(a, full_matrices=1, compute_uv=1)[source]
      Singular Value Decomposition.
      Factors the matrix a as u * np.diag(s) * v, where u and v are unitary and s is a 1-d array of a's singular values.
   # for this script:
    A_mat   == 'a'  # input
    U_mat   == 'u'  # unitary
    SIG_mat == 'np.diag(s)' # s is a 1-d array of a's singular values.
    VT_mat  == 'v'  # unitary
  """

  # getting started
  u,s,v = np.linalg.svd(A_mat)
  u2,s2,v2 = np.linalg.svd(A_mat, full_matrices=False)

  # re-calculate input matrix
  print("# Original: A")
  print(A_mat)
  print("# Restored: A2 = [U dot SIGMA] dot V:")
  # src: http://stackoverflow.com/a/24914785
  A2_mat = np.dot(np.dot(u2,np.diag(s2)),v2)
  # filter out values close to zero
  #   [src: http://stackoverflow.com/a/33306495 (isclose) ; src: http://stackoverflow.com/a/28279557 (mask and =0) ]
  A2_mat[np.isclose(A2_mat,0)] = 0
  print(A2_mat)
  print("# Original Matrix vs np.linalg with full_matrices=true and full_matrices=false")
  print("_type_  %s\tvs %s\t vs %s  " % ("U" , "VT" , "S" ))
  print("[ref]   %s vs %s vs %s  " % (U_mat.shape, VT_mat.shape, SIG_mat.shape ))
  print("[fm=1]  %s vs %s vs %s  " % (u.shape , v.shape , s.shape ))
  print("[fm=0]  %s vs %s vs %s  " % (u2.shape , v2.shape , s2.shape ))
  print("")


compare_np_linalg(_mmds_p420)
compare_np_linalg(_mmds_p421)

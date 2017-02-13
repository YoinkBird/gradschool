# thingy?
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.misc

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

def round_to_zero(A2_mat):
  # filter out values close to zero
  #   [src: http://stackoverflow.com/a/33306495 (isclose) ; src: http://stackoverflow.com/a/28279557 (mask and =0) ]
  A2_mat[np.isclose(A2_mat,0)] = 0
  return A2_mat

file_relpath = "mona_lisa.png"
#load_img_to_arr(file_relpath)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imread.html
img_arr = scipy.misc.imread(file_relpath, flatten=True)

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
# create U,s,V and then re-create the original matrix for verification
U,s,V = np.linalg.svd(img_arr, full_matrices=False)
S = np.diag(s)
img2_arr = np.dot(np.dot(U,np.diag(s)),V)
print("Reconstruction from Partial SVD? %s" % np.allclose(img_arr,img2_arr))
print("expect:[%s]M  = [(%d, x)]U [(x,)]S [(x, %d)]VT" % (img_arr.shape, img_arr.shape[0], img_arr.shape[1]))
print("[fm=0] [%s]M2 = [%s]U [%s]S [%s]VT  " % (img2_arr.shape, U.shape , s.shape , V.shape ))

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
# create U,s,V and then re-create the original matrix for verification
U,s,V = np.linalg.svd(img_arr, full_matrices=True)
S = np.zeros((img_arr.shape), dtype=complex)
S[:len(s), :len(s)] = np.diag(s)
SV = np.dot(S,V)
np.dot(U, SV)
img2_arr = np.dot(U, np.dot(S,V))
print("Reconstruction from Full SVD? %s" % np.allclose(img_arr,img2_arr))
print("expect:[%s]M  = [(%d, x)]U [(x,)]S [(x, %d)]VT" % (img_arr.shape, img_arr.shape[0], img_arr.shape[1]))
print("[fm=0] [%s]M2 = [%s]U [%s]S [%s]VT  " % (img2_arr.shape, U.shape , s.shape , V.shape ))

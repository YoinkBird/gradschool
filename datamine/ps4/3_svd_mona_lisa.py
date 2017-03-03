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

def round_to_zero(matrix):
  # filter out values close to zero
  #   [src: http://stackoverflow.com/a/33306495 (isclose) ; src: http://stackoverflow.com/a/28279557 (mask and =0) ]
  matrix[np.isclose(matrix,0)] = 0
  return matrix

def expand_s_to_S(arr, s, full_matrices=True):
  S = np.zeros((arr.shape))#, dtype=complex)
  if(full_matrices):
    S[:len(s), :len(s)] = np.diag(s)
  else:
    S = np.diag(s)
  return S

def svd_reconstruct(U, S, V, full_matrices=True):
  SV = np.dot(S,V)
  np.dot(U, SV)
  arr2 = np.dot(U, np.dot(S,V))
#  if(full_matrices):
#  else:
  return arr2

# 'blocksize' in bits
# 'krank' is the rank
def svd_size(U, krank, V, blocksize=16):
  size = 0
  size += (blocksize * krank)
  size += (blocksize * krank * U.shape[0])
  size += (blocksize * krank * V.shape[1])
  return size

def print_partial_svd_reconstruction(arr1):
  # partial SVD
  U,s,V = np.linalg.svd(arr1, full_matrices=False)
  S = np.zeros((arr1.shape), dtype=complex)
  S = np.diag(s)
  SV = np.dot(S,V)
  np.dot(U, SV)
  arr2 = np.dot(U, np.dot(S,V))
  print("Reconstruction from Partial SVD? %s" % np.allclose(arr1,arr2))
  print("expect:[%s]M  = [(%d, _)]U [(_,)]S [(_, %d)]VT" % (arr1.shape, arr1.shape[0], arr1.shape[1]))
  print("[fm=0] [%s]M2 = [%s]U [%s]S [%s]VT  " % (arr2.shape, U.shape , s.shape , V.shape ))
  return(U,s,V)

def print_full_svd_reconstruction(arr1):
  # Full SVD
  U,s,V = np.linalg.svd(arr1, full_matrices=True)
  S = np.zeros((arr1.shape), dtype=complex)
  S[:len(s), :len(s)] = np.diag(s)
  SV = np.dot(S,V)
  np.dot(U, SV)
  arr2 = np.dot(U, np.dot(S,V))
  print("Reconstruction from Full SVD? %s" % np.allclose(arr1,arr2))
  print("expect:[%s]M  = [(%d, _)]U [(_,)]S [(_, %d)]VT" % (arr1.shape, arr1.shape[0], arr1.shape[1]))
  print("[fm=0] [%s]M2 = [%s]U [%s]S [%s]VT  " % (arr2.shape, U.shape , s.shape , V.shape ))
  return(U,s,V)


file_subject = "mona_lisa"
detail = "" # original file, mona_lisa.png
file_relpath = "%s%s.png" % (file_subject, detail)
#load_img_to_arr(file_relpath)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imread.html
img_arr = scipy.misc.imread(file_relpath, flatten=True)

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
# create U,s,V and then re-create the original matrix for verification
# Partial SVD
U,s,V = print_partial_svd_reconstruction(img_arr)

# Full SVD
U,s,V = print_full_svd_reconstruction(img_arr)
dumpExtra = False
from PIL import Image
if(dumpExtra):
  im_u = Image.fromarray(U).convert('RGB')
  im_v = Image.fromarray(V).convert('RGB')
  im_u.save("svd_concept_u.png")
  im_v.save("svd_concept_v.png")

#  Show the best rank k = 2, k = 5 and k = 10 approximation to Mona Lisa.
kvals = [2,5,10,20,100,200,len(s)]
for kval in kvals:
  size = svd_size(U,kval,V)
  S2 = expand_s_to_S(img_arr, s[:kval])
  arrk = svd_reconstruct(U,S2,V)
  if(dumpExtra):
    im_s = Image.fromarray(S2).convert('RGB')
    im_s.save("out/" + "svd_concept_s_%s.png" % kval)


  print("array: %d ; rank: %d ; size: %d ;" % (len(s),kval,size))
  # plt.imshow(im_s)
  dumpImg = True
  if(dumpImg):
    # have to RGB in order to save. src: http://stackoverflow.com/a/18879396
    im  = Image.fromarray(arrk).convert('RGB')
    img_out_relpath = "svd_%s%d%s.png" % (file_subject,kval,detail)
    img_out_relpath = "out/" + img_out_relpath
    im.save(img_out_relpath)

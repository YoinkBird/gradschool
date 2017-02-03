import numpy,pandas,matplotlib
import numpy as np
import matplotlib.pyplot as plt
from pandas import *

"""
1. Jaccard Similarity. Compute the Jaccard Similarity of the three sets S1 = {'nike',
'running', 'shoe'}, S2 = {'nike', 'black', 'running', 'shoe'}, and S3 = {'nike',
'blue', 'jacket', 'adidas'}, using the intersection and union of lists in Python.
"""
# list
S1 = {'nike', 'running', 'shoe'}
S2 = {'nike', 'black', 'running', 'shoe'}
S3 = {'nike', 'blue', 'jacket', 'adidas'}
# set
set1 = set(S1)
set2 = set(S2)
set3 = set(S3)

def jaccard(set_S, set_T):
  # ensure both are sets
  set_inter = set_S.intersection(set(set_T))
  set_union = set_S.union(set_T)
  total = len(set_S) + len(set_T)
  similarity = len(set_inter) / len(set_union)
  return similarity


'''
Calibrate jaccard function based on Ex3.1 from MMDS CH3
'''
def jaccard_calibrate():
  print("jaccard function: reticulating splines");
  S1a = {1,2,3,4,5}
  S1b = {3,4,5,6,7,8}
  print(jaccard(S1a,S1b) == float(3/8))

def problem1():
  print(jaccard(S1,S2))
  print(jaccard(S1,S3))
  print(jaccard(S2,S3))


jaccard_calibrate()
problem1()

"""
2. Minhash
(a) For the example above, create the characteristic matrix where the alphabet is taken to be
the seven words {'nike', 'running', 'shoe','black','blue','jacket','adidas'}.
(b) For a random permutation of the seven alphabet elements, find a way to compute the
first non-zero element of each column (i.e., of each set), under the permutation.
(c) Now do the same where instead of choosing a random permutation, you use the
hash function: h(x) = 3x + 2 (mod 7).
(d) Generate your own hash functions of the form h(x) = a * x + b (mod 7) by choosing a
and b at random from {0,1,...,6}. Doing this 20 times, estimate the Jaccard Similarity
of the three sets. How closely do you approximate the true values, computed in the
previous exercise?
"""
import ipdb;ipdb.set_trace();

## convert sets to dict and add missing values
# dict : https://docs.python.org/2/library/stdtypes.html#dict
def dict_from_set(keywords,set_s):
  # create dict with all values set to 1
  set_dict = dict.fromkeys(set_s,1)
  # set missing values to 0
  for word in keywords:
    # idempotent set
    set_dict.setdefault(word,0)
  return set_dict

# create char matrix from sets
def char_matrix_1():
  # set of all keywords
  set_keywords = set.union(S1,S2,S3)
  dict_s1 = dict_from_set(set_keywords,S1)
  dict_s2 = dict_from_set(set_keywords,S2)
  dict_s3 = dict_from_set(set_keywords,S3)
  matrix_dict = {}
  matrix_dict['S1'] = dict.fromkeys(S1,1)
  matrix_dict['S2'] = dict.fromkeys(S2,1)
  matrix_dict['S3'] = dict.fromkeys(S3,1)
  import ipdb;ipdb.set_trace();
  # src: http://stackoverflow.com/a/10628728
  df = DataFrame(matrix_dict).T.fillna(0).transpose()


  print()
  import ipdb;ipdb.set_trace();
  return df

# a: characteristic matrix
char_matrix_df = char_matrix_1()
print(char_matrix_df)

# b: 
  
    




"""
3. Implementing Minhash
Repeat the above exercise where instead of permuting the entire characteristic matrix, you
implement the algorithm described in Chapter 3 of MMDS, for implementing Minhash.
"""
"""
4. More MinHash: Shingling
(a) Figure out how to load the 5 article excerpts in HW3articles-5.txt.
1(b) Use stopwords from the natural language processing module nltk.corpus to strip the
stopwords from the five articles.
(c) Compute the k-shingles of the documents, for k = 2, where you shingle on words, not
letters.
(d) Compute the k-shingles of the documents, for k = 3, where you shingle on characters,
not words.
"""
"""
5. Even More MinHash: Document Similarity
(a) For each of the documents above, and for the shingles generated in both ways (words,
characters), generate MinHash signatures using 30 hash functions. As above, each hash
function takes the form h(x) = ax+b(mod p). As explained in the book, it is important
to choose p to be a prime number. Thus, set p = 4;294;967;311, which is a prime number
larger than 232-1 (and thus sufficiently large for our purposes). Uniformly sample n = 30
values of (a; b) 2 f0;1;2; : : : ; p-1g and compute the corresponding MinHash signatures.
Note that to do this, multiplication has to be defined. You will therefore need to map
each of the k-shingles to integers. One way to do this is, for example, using the CRC32
hash.
(b) Which of the five documents is most similar to the first (t121)? And which worked
better: shingling words or characters?
"""
"""
6. (Bonus) LSH. So far, we have only been exploring comparisons between very few documents.
What if we wish to compare a very large number of documents? In particular, suppose we
wish to compare a new document to many many previous ones. If we follow the above flow,
the time to do this will scale linearly in the number of documents. LSH (Locality Sensitive
Hashing), as explained in MMDS Chapter 3, is a data structure that allows us to do this:
we have to work to create the data structure, but once that is in place, we can very quickly
find close matches without having to go through the documents one by one. This idea is
similar in spirit (very different in the details), to the example of a dictionary, as discussed
in class: if you want to find if a word is in the dictionary, the time to do so does not
scale linearly in the number of words in the dictionary. Explore LSH using the 1;000 texts
contained in HW3articles-1000.txt. The key question: how does the time of comparing 1
to N documents scale, as N grows? Can you get it to scale sublinearly?
"""

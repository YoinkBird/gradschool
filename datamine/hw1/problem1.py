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
  prob1_jaccard_dict = {}
  prob1_jaccard_dict['S1_S2'] = jaccard(S1,S2)
  prob1_jaccard_dict['S1_S3'] = jaccard(S1,S3)
  prob1_jaccard_dict['S2_S3'] = jaccard(S2,S3)
  print(prob1_jaccard_dict)
  return prob1_jaccard_dict


jaccard_calibrate()
prob1_jaccard_dict = problem1()

"""
2. Minhash
(a) For the example above, create the characteristic matrix where the alphabet is taken to be
the seven words {'nike', 'running', 'shoe','black','blue','jacket','adidas'}.
(b) For a random permutation of the seven alphabet elements, find a way to compute the
first non-zero element of each column (i.e., of each set), under the permutation.
(c) Now do the same where instead of choosing a random permutation, you use the hash function: h(x) = 3x + 2 (mod 7).
(d) Generate your own hash functions of the form h(x) = a * x + b (mod 7) by choosing a
and b at random from {0,1,...,6}. Doing this 20 times, estimate the Jaccard Similarity
of the three sets. How closely do you approximate the true values, computed in the previous exercise?
"""

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
  # src: http://stackoverflow.com/a/10628728
  df = DataFrame(matrix_dict).T.fillna(0).transpose()


  print()
  return df

# a: characteristic matrix
char_matrix_df = char_matrix_1()
print(char_matrix_df)

# create permutation
# src: http://stackoverflow.com/a/13401681
def permutate_rand(df):
  # (b) For a random permutation of the seven alphabet elements, find a way to compute the
  # number of rows = df.shape[0]
  permute = np.random.permutation(df.shape[0])
  df2 = df.take(permute)
  return df2

# hash function
# hash function of the form h(x) = a * x + b (mod 7)
def hash_fn(x,a,b,modulo):
  value = (a * x + b) % modulo
  return value

# permuate with hash
# use the hash function: h(x) = 3x + 2 (mod 7).
# src: http://stackoverflow.com/a/13401681
def permutate_hash1_fixed(df):
  # (b) For a random permutation of the seven alphabet elements, find a way to compute the
  # number of rows = df.shape[0]
  # (c) Now do the same where instead of choosing a random permutation, you use the hash function: h(x) = 3x + 2 (mod 7).
  permute = []
  rnd_a = 3
  rnd_b = 2
  for i in range(0,df.shape[0]):
    # value = (3 * i + 2) % df.shape[0]
    value = hash_fn(i,rnd_a,rnd_b,df.shape[0])
    permute.append(value)
  # src http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.take.html
  #     https://docs.scipy.org/doc/numpy/reference/generated/numpy.take.html
  # Take elements from an array along an axis.
  df2 = df.take(permute)
  return df2

# permuate with hash
# use the hash function: h(x) = ax + b (mod 7).
# (d) Generate your own hash functions of the form h(x) = a * x + b (mod 7) by choosing a
# and b at random from {0,1,...,6}. Doing this 20 times, estimate the Jaccard Similarity
# of the three sets. How closely do you approximate the true values, computed in the previous exercise?
# params: df: dataframe, randint_range: range for generating integers, modulo: moduloe for hash function
# PURPOSE: permutate hash as input for computation of minhash signature
def permutate_hash2_rand(df,randint_range,**kwargs):
  permute = []
  modulo = df.shape[0]
  if("modulo" in kwargs):
    modulo = kwargs["modulo"]
  for i in range(0,modulo):
    rnd_a = np.random.randint(0,randint_range)
    rnd_b = np.random.randint(1,randint_range)
    value = hash_fn(i,rnd_a,rnd_b,modulo)
    permute.append(value)
  #DEBUG print("# rand 0,6: " + str(permute))
  df2 = df.take(permute)
  return df2

def first_nonzero(needle,haystack):
  # get first occurence
  for index in range(0,len(haystack)):
    if(haystack[index] >= needle):
      return index

# compute the first non-zero element of each column (i.e., of each set), under the permutation.
# i.e. return the row-index of the first non-zero element
# src: pandas indexing - http://pandas.pydata.org/pandas-docs/stable/indexing.html
def first_nonzero_df_dict(df):
  non_zero_mat={}
  # iterate over column series TODO: iterate directly using series 
  for ser_col in df.columns:
    # get list of values based on df[ser_col] TODO: use series iterator instead of lookup
    vals = list(df[ser_col])
    # find first value '1' 
    index = first_nonzero(1.0,vals)
    #non_zero_mat[ser.name] = df.index[index]
    # store name of col (the set) and look up the row name (df index)
    # src: http://stackoverflow.com/questions/18327624/find-elements-index-in-pandas-series
    # TODO: use iterator, e.g. ser_col.name
    non_zero_mat[df[ser_col].name] = df[ser_col].index[index]

  return non_zero_mat


# print first occurence
print("""
# (b) For a random permutation of the seven alphabet elements, find a way to compute the
first non-zero element of each column (i.e., of each set), under the permutation.
""")
df_rand = permutate_rand(char_matrix_df)
first_nonzero_rand = first_nonzero_df_dict(df_rand)
print(df_rand)
print(first_nonzero_rand)

# first_nonzero_matrix={}
# first_nonzero_matrix['rand'] = first_nonzero_df_dict(df_rand)

print("""
# (c) Now do the same where instead of choosing a random permutation, you use the hash function: h(x) = 3x + 2 (mod 7).
""")
df_hash1 = permutate_hash1_fixed(char_matrix_df)
first_nonzero_hash1 = first_nonzero_df_dict(df_hash1)
print(df_hash1)
print(first_nonzero_hash1)
  
    
print("""
# (d) Generate your own hash functions of the form h(x) = a * x + b (mod 7) by choosing a
# and b at random from {0,1,...,6}. Doing this 20 times, estimate the Jaccard Similarity
# of the three sets. How closely do you approximate the true values, computed in the previous exercise?
""")
def problem2():
  print("jaccard similarity for 20 random hash functions")
  print("hash fn\tS1,S2\tS1,S3\tS2,S3")
  print("#true\t%02.2f (fit)\t%02.2f (fit)\t%02.2f (fit)" % (prob1_jaccard_dict['S1_S2'], prob1_jaccard_dict['S1_S3'],prob1_jaccard_dict['S2_S3']))
  print("---------------------------")
  for i in range(20):
    # get minhashed values
    df_hash2_rand = permutate_hash2_rand(char_matrix_df, 6)
    # TODO: convert to function
    # convert to sets
    set_mat = {}
    for set_name in df_hash2_rand.columns:
      # get key-value
      hash2_rand_dict = df_hash2_rand[set_name].to_dict()
      # correlate key with entry, i.e. convert series to set
      tmpset = set()
      for key in hash2_rand_dict:
        if hash2_rand_dict[key] >= 1.0:
          tmpset.add(key)
      set_mat[set_name] = tmpset

    # calculate jaccard
    jaccard_dict = {}
    jaccard_dict['S1_S2'] = jaccard( set_mat['S1'],set_mat['S2'])
    jaccard_dict['S1_S3'] = jaccard( set_mat['S1'],set_mat['S3'])
    jaccard_dict['S2_S3'] = jaccard( set_mat['S2'],set_mat['S3'])
    # weird - calculate percent difference to true values
    # S1_S2_err = 100 * (abs(prob1_jaccard_dict['S1_S2'] - jaccard_dict['S1_S2']) / prob1_jaccard_dict['S1_S2'])
    # S1_S3_err = 100 * (abs(prob1_jaccard_dict['S1_S3'] - jaccard_dict['S1_S3']) / prob1_jaccard_dict['S1_S3'])
    # S2_S3_err = 100 * (abs(prob1_jaccard_dict['S2_S3'] - jaccard_dict['S2_S3']) / prob1_jaccard_dict['S2_S3'])
    # simple percentage
    S1_S2_err = 100 * (jaccard_dict['S1_S2'] / prob1_jaccard_dict['S1_S2'])
    S1_S3_err = 100 * (jaccard_dict['S1_S3'] / prob1_jaccard_dict['S1_S3'])
    S2_S3_err = 100 * (jaccard_dict['S2_S3'] / prob1_jaccard_dict['S2_S3'])

    # print results
    print("#%02s\t%02.2f (%d%%)\t%02.2f (%d%%)\t%02.2f (%d%%)" % (i, jaccard_dict['S1_S2'], S1_S2_err, jaccard_dict['S1_S3'], S1_S3_err, jaccard_dict['S2_S3'], S2_S3_err))


  return
problem2()




"""
3. Implementing Minhash
Repeat the above exercise where instead of permuting the entire characteristic matrix, you
implement the algorithm described in Chapter 3 of MMDS, for implementing Minhash.
"""
"""
4. More MinHash: Shingling
(a) Figure out how to load the 5 article excerpts in HW3articles-5.txt.
(b) Use stopwords from the natural language processing module nltk.corpus to strip the stopwords from the five articles.
(c) Compute the k-shingles of the documents, for k = 2, where you shingle on words, not
letters.
(d) Compute the k-shingles of the documents, for k = 3, where you shingle on characters,
not words.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



import nltk
# nltk.download('english')
# nltk.download(\"stopwords\")
stop = set(nltk.corpus.stopwords.words('english'))
# (a) Figure out how to load the 5 article excerpts in HW3articles-5.txt.
# TODO: super wrong, whatever who gives a crap
data = pd.read_csv("files_pset3/HW3articles-5.txt",delimiter="\t")

# verify


# Use stopwords from the natural language processing module nltk.corpus to strip the
# stopwords from the five articles.
def strip_stopwords_nope(df):
    df.apply(lambda x: [item for item in x if item not in stop], axis=0)
    
    return df

word_list = []
#import ipdb;ipdb.set_trace();
# stupid line is adding '...Name:4,dtype:object'
for line in data.iterrows():
  # print(line);
  # src: http://stackoverflow.com/a/15057966
  word_list += nltk.word_tokenize(str(line[1]))
  # return(word_list)

# strip stopwords
def strip_stopwords(word_list):
  small_list = []
  for item in word_list:
    if item not in stop:
      small_list.append(item)
  return(small_list)

# shingling
def shingle_words(tkn_list,k_len):
  shingles = []
  # range is exclusive limit
  lim = (len(tkn_list) - k_len) + 1
  for i in range( lim ):
    shingles.append( tkn_list[i:i+k_len] )
  return shingles

#import ipdb;ipdb.set_trace();
# (b) Use stopwords from the natural language processing module nltk.corpus to strip the stopwords from the five articles.
tkns = strip_stopwords(word_list)
tkns_char_str = ''.join(tkns)
# (c) Compute the k-shingles of the documents, for k = 2, where you shingle on words, not letters.
word_shingles = shingle_words(tkns,2)
# (d) Compute the k-shingles of the documents, for k = 3, where you shingle on characters, not words.
char_shingles = shingle_words(tkns_char_str,3)

# print("# data before")
# print(data)
# data2 = strip_stopwords_nope(data)
# print("# data after")
# print(data2)

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

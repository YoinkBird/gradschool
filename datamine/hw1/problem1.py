import numpy,pandas,matplotlib
import numpy as np
import matplotlib.pyplot as plt
from pandas import *

probprint_dict = {}
probprint_dict[1] = 0
probprint_dict[2] = 0
probprint_dict[3] = 0
probprint_dict[4] = 1
probprint_dict[5] = 0

"""
1. Jaccard Similarity. Compute the Jaccard Similarity of the three sets S1 = {'nike',
'running', 'shoe'}, S2 = {'nike', 'black', 'running', 'shoe'}, and S3 = {'nike',
'blue', 'jacket', 'adidas'}, using the intersection and union of lists in Python.
"""
print(" 1. Jaccard Similarity. Compute the Jaccard Similarity of the three sets S1,S2,S3")
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
  return prob1_jaccard_dict


prob1_jaccard_dict = problem1()
if(probprint_dict[1]):
  jaccard_calibrate()
  print(prob1_jaccard_dict)

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
print("2. Minhash")

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
if(probprint_dict[2]):
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
dbprint_dict = {}
dbprint_dict['hash_fn'] = 0
def hash_fn(x,a,b,modulo):
  if(dbprint_dict['hash_fn']):
    print("(%d * %d + %d) %% %d)" % (a, x, b, modulo))
  value = (a * x + b) % modulo
  return value

# compute one random hash value
# intended to return the hash value used to generate one row of sig matrix
def randhash_fn(x_val, randint_range,modulo,**kwargs):
  rnd_a = np.random.randint(0,randint_range)
  rnd_b = np.random.randint(1,randint_range)
  value = hash_fn(x_val,rnd_a,rnd_b,modulo)
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
  for i in range(0,df.shape[0]):
    rnd_a = np.random.randint(0,randint_range)
    rnd_b = np.random.randint(1,randint_range)
    value = hash_fn(i,rnd_a,rnd_b,modulo)
    permute.append(value)

  df2 = DataFrame()
  df2 = df.take(permute)
  # TODO: set 'df2' on permute
  # df2.index=permute
  return df2

# inputs: char_matrix_df: characteristic matrix, hash range for random values: randint_range,
def calculate_minhash_sig_matrix_permute__df(char_matrix__df, num_hash_fn, randint_range,**kwargs):
  # for problem 5
  modulo = char_matrix__df.shape[0]
  if("modulo" in kwargs):
    modulo = kwargs["modulo"]
  minhash_sig_matrix__df = DataFrame(columns=char_matrix_df.columns)
  for i in range(num_hash_fn):
    # get signature matrix
    df_hash2_rand = permutate_hash2_rand(char_matrix_df, 6)
    # calculate hash signatures
    first_nonzero_hash2_rand = first_nonzero_df_dict(df_hash2_rand)
    #  PANDAS
    # convert dict to pd Series to add to the dataframe
    hash_i_pdseries = Series(first_nonzero_hash2_rand, name=i)
    ### could also create dataframe
    ### hash_i_df = DataFrame(first_nonzero_hash2_rand, index=[i])
    # add Series to dataframe
    #  src: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.append.html
    #  have to re-assign each time or else parent 'df' stays empty
    minhash_sig_matrix__df = minhash_sig_matrix__df.append(hash_i_pdseries)
    ### could also add DataFrame to DataFrame
    ### minhash_sig_matrix__df = minhash_sig_matrix__df.append(hash_i_df)
  return minhash_sig_matrix__df

# MMDS CH3 page 84
# However, if c has 1 in row r, then for each i = 1, 2, . . . , n
#   set SIG(i, c) to the smaller of the current value of SIG(i, c) and h_i(r).
# i.e. if at r,c there is a 1, compute h_i(r)
# if h_i(r) < SIG(i,c): SIG(i,c) = h_i(r)
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
if(probprint_dict[2]):
  print("""
# (b) For a random permutation of the seven alphabet elements, find a way to compute the
first non-zero element of each column (i.e., of each set), under the permutation.
""")
df_rand = permutate_rand(char_matrix_df)
first_nonzero_rand = first_nonzero_df_dict(df_rand)
if(probprint_dict[2]):
  print(df_rand)
  print(first_nonzero_rand)

# first_nonzero_matrix={}
# first_nonzero_matrix['rand'] = first_nonzero_df_dict(df_rand)

if(probprint_dict[2]):
  print("""
# (c) Now do the same where instead of choosing a random permutation, you use the hash function: h(x) = 3x + 2 (mod 7).
""")
df_hash1 = permutate_hash1_fixed(char_matrix_df)
first_nonzero_hash1 = first_nonzero_df_dict(df_hash1)
if(probprint_dict[2]):
  print(df_hash1)
  print(first_nonzero_hash1)
  
    
if(probprint_dict[2]):
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
    print("#%02s\t%02.2f [%3d%%]\t%02.2f [%3d%%]\t%02.2f [%3d%%]" % (i, jaccard_dict['S1_S2'], S1_S2_err, jaccard_dict['S1_S3'], S1_S3_err, jaccard_dict['S2_S3'], S2_S3_err))


  return
if(probprint_dict[2]):
  problem2()




"""
3. Implementing Minhash
Repeat the above exercise where instead of permuting the entire characteristic matrix, you
implement the algorithm described in Chapter 3 of MMDS, for implementing Minhash.
"""
"""
1. generate matrix of signatures
2. process matrix of signatures
"""
print("3. Implementing Minhash")

# compute one minhash signature for a characteristic matrix
# return values for one "row" for the signature matrix
def calculate_mh_sig__df_dict(df,randint_range,**kwargs):
  permute = []
  modulo = df.shape[0]
  if("modulo" in kwargs):
    modulo = kwargs["modulo"]
  for i in range(0,df.shape[0]):
    rnd_a = np.random.randint(0,randint_range)
    rnd_b = np.random.randint(1,randint_range)
    value = hash_fn(i,rnd_a,rnd_b,modulo)
    permute.append(value)

  df2 = DataFrame()
  df2 = df.take(permute)
  # TODO: set 'df2' on permute
  # df2.index=permute
  return df2
# MMDS CH3 page 84
# However, if c has 1 in row r, then for each i = 1, 2, . . . , n
#   set SIG(i, c) to the smaller of the current value of SIG(i, c) and h_i(r).
# i.e. if at r,c there is a 1, compute h_i(r)
# if h_i(r) < SIG(i,c): SIG(i,c) = h_i(r)
import math
def calculate_minhash_sig_matrix__df(char_matrix__df, numhashes, randint_range,**kwargs):
  # dbprint_dict['hash_fn'] = 1
  modulo = char_matrix__df.shape[0]
  if("modulo" in kwargs):
    modulo = kwargs["modulo"]
  # init sighash as NaN (equivalent to infinity)
  # needs one row per hash function, and one column per set from the characteristic matrix
  sighash = DataFrame(index=range(numhashes),columns=char_matrix__df.columns)
  # generate n hash functions
  hashvalues = {}
  for i in range(numhashes):
    ## print("#hash number: %d" % i)
    printflag = 0
    # get random parameters
    rnd_a = np.random.randint(0,randint_range)
    rnd_b = np.random.randint(1,randint_range)
    # loop through characteristic matrix one row at a time
    for rownum,row in char_matrix__df.iterrows():
      # rownum may need conversion from string to int
      # TODO: this may need to be CRC23'd later
      rownum = char_matrix__df.index.get_loc(rownum)
      ## print("#rownum: %d" % rownum)
      # compute hash value of current row
      # error: different hash function for each row, not good!
      cur_hash_val = hash_fn(rownum, rnd_a, rnd_b, modulo)
      # loop each row of characteristic matrix
      for colname in row.index:
        # if characteristic matrix value == 1
        if(row[colname] == 1):
        #if( sighash.ix[i][colname] == 1 or (math.isnan(sighash.ix[i][colname])) ):
          # if hash value of current row,col smaller than value at sighash of i'th hash-row and col
          # i.e. set the sighash to the smallest value
          if(cur_hash_val < sighash.ix[i][colname] or (math.isnan(sighash.ix[i][colname])) ):
            sighash.ix[i][colname] = int(cur_hash_val) # int() just to be safe
            # print("new hash smaller")
        else:
          if(0 and printflag):
            print("skipping %d" % row[colname])
    if(printflag):
      print(sighash)
  # dbprint_dict['hash_fn'] = 0
  return sighash
def problem3():
  numhashes = 20
  randvals = 6
  minhash_sig_matrix__df = calculate_minhash_sig_matrix__df(char_matrix_df,numhashes,randvals)
  print(minhash_sig_matrix__df)
  return
if(probprint_dict[3]):
  problem3()
"""
4. More MinHash: Shingling
(a) Figure out how to load the 5 article excerpts in HW3articles-5.txt.
(b) Use stopwords from the natural language processing module nltk.corpus to strip the stopwords from the five articles.
(c) Compute the k-shingles of the documents, for k = 2, where you shingle on words, not
letters.
(d) Compute the k-shingles of the documents, for k = 3, where you shingle on characters,
not words.
"""
print("4. More MinHash: Shingling")
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
if(probprint_dict[4]):
  print("# (b) Use stopwords from the natural language processing module nltk.corpus to strip the stopwords from the five articles.")
  print("before: %d" % len(word_list))
  tkns = strip_stopwords(word_list)
  tkns_char_str = ''.join(tkns)
  print("after: %d" % len(tkns))
  print(tkns_char_str[-1])
  print("# (c) Compute the k-shingles of the documents, for k = 2, where you shingle on words, not letters.")
  word_shingles = shingle_words(tkns,2)
  print(len(word_shingles))
  print(word_shingles[-1])
  print("# (d) Compute the k-shingles of the documents, for k = 3, where you shingle on characters, not words.")
  char_shingles = shingle_words(tkns_char_str,3)
  print(len(char_shingles))
  print(char_shingles[-1])


# print("# data before")
# print(data)
# data2 = strip_stopwords_nope(data)
# print("# data after")
# print(data2)

"""
5. Even More MinHash: Document Similarity
(a) For each of the documents above, and for the shingles generated in both ways (words,
characters), generate MinHash signatures using 30 hash functions.
As above, each hash function takes the form h(x) = ax+b(mod p).
As explained in the book, it is important to choose p to be a prime number.
Thus, set p = 4;294;967;311, which is a prime number larger than 232-1 (and thus sufficiently large for our purposes).
Uniformly sample n = 30 values of (a, b) element {0;1;2;...; p-1} and compute the corresponding MinHash signatures.
Note that to do this, multiplication has to be defined.
You will therefore need to map each of the k-shingles to integers.
One way to do this is, for example, using the CRC32 hash.
(b) Which of the five documents is most similar to the first (t121)?
And which worked better: shingling words or characters?
"""
# about "Note that to do this, multiplication has to be defined."
# previous examples used the size as the modulo, so the 'x' was simply the loop index
# in this example, it could be any value from the large primes
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

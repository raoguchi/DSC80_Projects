# project.py


import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------

def get_book(url):
    time.sleep(0.5)
    book = requests.get(url)
    book_text = book.text
    start = re.search(r"\*{3} START.*\*", book_text).end()
    end = re.search(r"\*{3} END.*\*", book_text).start()
    book_text = book_text[start:end]
    book_text = book_text.replace("\r\n", "\n")
    return book_text


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    strings_book = '\x02' + book_string.strip() + '\x03'
    strings_book = re.sub(r"(\n){2,}", "\nx03 \x02\n", strings_book)
    word_list = re.findall(r'\x02|\x03|\w+|[^\w\s]', strings_book)
    return word_list


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):


    def __init__(self, tokens):

        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        token_unique = set(tokens)
        token_series = pd.Series(data = 1, index = token_unique)
        return token_series / len(token_series)
    
    def probability(self, words):
        if words[0] in self.mdl.index:
            prop_one = self.mdl[words[0]]
        else:
            prop_one = 0
        if len(words) == 2:
            if (words[1] in self.mdl.index):
                prop_two = self.mdl[words[1]]
            else:
                prop_two = 0
            prop = prop_one * prop_two
        else:
            prop = prop_one
        return prop
        
    def sample(self, M):
        word_sample = self.mdl.sample(M, replace = True).index
        sample_string =  " ".join(word_sample)
        return sample_string


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):

        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        token_list = list(tokens)
        token_counts = pd.Series(token_list).value_counts()
        token_counts = token_counts / token_counts.sum()
        return token_counts
    
    def probability(self, words):
        if words[0] in self.mdl.index:
            prop_one = self.mdl[words[0]]
        else:
            prop_one = 0
        if len(words) == 2:
            if (words[1] in self.mdl.index):
                prop_two = self.mdl[words[1]]
            else:
                prop_two = 0
            prop = prop_one * prop_two
        else:
            prop = prop_one
        return prop
        
    def sample(self, M):
        word_sample = self.mdl.sample(M, replace = True).index
        sample_string =  " ".join(word_sample)
        return sample_string


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.tokens = tokens
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

            
    def create_ngrams(self, tokens):
        tuple_size = self.N
        ngram_list = []
        for i in np.arange(0, len(tokens)):
            partition = tuple(tokens[i:i+tuple_size])
            if len(partition) == tuple_size:
                ngram_list.append(partition)
        return ngram_list
        
            
        
    def train(self, ngrams):
        token_string = " ".join(self.tokens)
        
        # N-Gram counts C(w_1, ..., w_n)
        ngram_counts = pd.DataFrame({'ngram': ngrams})
        ngram_counts['ngram_freq'] = ngram_counts.groupby('ngram')['ngram'].transform("count")
        
        # (N-1)-Gram counts C(w_1, ..., w_(n-1))
        n1grams = [ngram[0:self.N-1] for ngram in ngrams]
        ngram_counts['n1gram'] = n1grams
        ngram_counts['n1gram_freq'] = ngram_counts.groupby('n1gram')['n1gram'].transform("count")

        # Create the conditional probabilities
        ngram_counts['prob'] = ngram_counts['ngram_freq'] / ngram_counts['n1gram_freq']
        ngram_counts = ngram_counts.drop(columns=['ngram_freq', 'n1gram_freq'])
       
        return ngram_counts.drop_duplicates(['ngram'])
    
    def probability(self, words):
        ngrams = self.mdl['ngram']
        
        gram_tokens = self.create_ngrams(words)
        
        prob = 1
        ngram_probs = ([self.mdl[self.mdl['ngram'] == gram]['prob'].iloc[0] 
                        if self.mdl['ngram'].isin([gram]).any()
                        else 0
                        for gram in gram_tokens])
        prob *= np.prod(ngram_probs)
          
       
        for i in range(self.N, 1, -1):
            smaller_n = words[0:i-1]
            smaller_n1 = tuple(words[0:i-2])
            last_mdl = NGramLM(i, self.tokens).prev_mdl
            if len(smaller_n) > 1:
                last_row = last_mdl.mdl[(last_mdl.mdl['ngram'] == smaller_n) & (last_mdl.mdl['n1gram'] == smaller_n1)]
                last_prob = last_row.iloc[0]['prob']
                prob *= last_prob
            else:
                last_row = last_mdl.mdl
                last_prob = last_row.loc[smaller_n[0]]
                prob *= last_prob
           
                
        
        return prob
    

    def sample(self, M):
        # Use a helper function to generate sample tokens of length `length`
        def gen_wd(gram_lvl, first_words):
            gram_mdl = NGramLM(gram_lvl, self.tokens).mdl
                
            grams = gram_mdl[gram_mdl['n1gram'].apply(lambda x: x == first_words)]
                
            if len(grams) == 0:
                 return ("\x03",)
            else:
                word = grams.sample(1, weights = grams['prob'], replace = True)
                return word['ngram'].iloc[0]
        
        # Transform the tokens to strings
        first_iter = ("\x02",)
        fst_two = gen_wd(2, first_iter)
        word_list = " ".join(fst_two)
        for i in range(3, M+1):
            if self.N > i:
                fst_two = fst_two[-(self.N-1):]
                fst_two = gen_wd(i, fst_two)
            else:
                fst_two = fst_two[-(self.N-1):]
                fst_two = gen_wd(self.N, fst_two)
            word_list += " " + fst_two[-1]
        word_list += " \x03"
        return word_list
                
                    
                    
                    
                    
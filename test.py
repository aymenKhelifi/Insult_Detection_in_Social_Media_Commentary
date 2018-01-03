import nltk
from nltk.util import ngrams

def word_grams(words, min=1, max=4):
    s = []
    for n in range(min, max):
        print('n : '+str(n)+'\n')
        for ngram in ngrams(words, n):
            print(ngram)
            #print(' '.join(str(i) for i in ngram))
            #print(' \n')
            s.append(' '.join(str(i) for i in ngram))
    print(s)
    return s

print(word_grams('one two three four'))
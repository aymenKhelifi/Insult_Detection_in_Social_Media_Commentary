import pandas as pd
import numpy as np
from nltk.util import ngrams
import re
import queue as Q
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score

## Manipulating Files

DIR_PATH = "./../resources/"

TRAIN_FILE      = DIR_PATH + "train.csv"
TEST_SOL_FILE   = DIR_PATH + "test_with_solutions.csv"   # This is also used for training, together with TRAIN_FILE
BADWORDS_FILE   = DIR_PATH + "badwords.txt"              # attached with submission

TEST_FILE       = DIR_PATH + "verification.csv"          # set this to the new test file name
PREDICTION_FILE = DIR_PATH + "test_results.csv"                 # predictions will be written here
STOPWORDS_FILE = DIR_PATH + "stopwords.csv"
NGRAMS_RANK_FILE = DIR_PATH + "ngrams_rank.csv"
NGRAMS_VALUES_FILE = DIR_PATH + "ngrams.csv"
CHAR_NGRAMS_RANK_FILE = DIR_PATH + "char_ngrams_rank.csv"
CHAR_NGRAMS_VALUES_FILE=DIR_PATH + "char_ngrams.csv"
RESULT_FILE = DIR_PATH + "result.csv"




def openFiles():
    train_data=pd.read_csv(TRAIN_FILE)
    train_labels=train_data.Insult.tolist()
    train_data=train_data.Comment.tolist()
    test_sol_data=pd.read_csv(TEST_SOL_FILE)
    test_sol_labels=test_sol_data.Insult.tolist()
    test_sol_data=test_sol_data.Comment.tolist()
    test_data = pd.read_csv(TEST_FILE)
    test_data=test_data.Comment.tolist()
    stop_words=pd.read_csv(STOPWORDS_FILE)
    stop_words=stop_words.stopwords.tolist()
    return train_data, train_labels, test_data, test_sol_data,test_sol_labels,stop_words

def preprocessing(train_data, test_data, test_sol_data, stop_words):
    train_data, test_data, test_sol_data=removeNan(train_data, test_data, test_sol_data)
    train_data=normalize(train_data, stop_words);
    test_data=normalize(test_data, stop_words);
    test_sol_data=normalize(test_sol_data, stop_words);
    return train_data, test_data, test_sol_data

def removeNan(train_data, test_data, test_sol_data):
    # train_data=train_data[np.isfinite(train_data['Insult'])]
    # train_data=train_data[np.isfinite(train_data['Comment'])]
    #
    # train_data=train_data[np.isfinite(train_data['Date'])]
    #
    # test_data=test_data[np.isfinite(test_data['Insult'])]
    # test_data=test_data[np.isfinite(test_data['Comment'])]
    # test_sol_data=test_sol_data[np.isfinite(test_sol_data['Insult'])]
    # test_sol_data=test_sol_data[np.isfinite(test_sol_data['Comment'])]
    return train_data, test_data, test_sol_data
def normalize(data, stop_words):
    ##stemming
    data = [x.replace(" u ", " you ") for x in data]
    data = [x.replace(" yo ", " you ") for x in data]
    data = [x.replace(" your ", " you ") for x in data]
    data = [x.replace(" ur ", " you ") for x in data]

    data = [x.replace(" em ", " them ") for x in data]
    data = [x.replace(" da ", " the ") for x in data]

    data = [x.replace("won't", "will not") for x in data]
    data = [x.replace("can't", "cannot") for x in data]
    data = [x.replace("i'm", "i am") for x in data]
    data = [x.replace(" im ", " i am ") for x in data]
    data = [x.replace("ain't", "is not") for x in data]
    data = [x.replace("'ll", " will") for x in data]
    data = [x.replace("'t", " not") for x in data]
    data = [x.replace("'ve", " have") for x in data]
    data = [x.replace("'s", " is") for x in data]
    data = [x.replace("'re", " are") for x in data]
    data = [x.replace("'d", " would") for x in data]

    ## removing special charchters
    data = [x.lower() for x in data]
    data = [x.replace("\\n", " ") for x in data]
    data = [x.replace("\\t", " ") for x in data]
    data = [x.replace("\\xa0", " ") for x in data]
    data = [x.replace("\\xc2", " ") for x in data]

    data = [x.replace(","," ").replace("."," ").replace(" ", "  ") for x in data]
    data = [x.replace('"','') for x in data]


    data = [re.subn("ies( |$)", "y ", x)[0].strip() for x in data]
    # f = [re.subn("([abcdefghijklmnopqrstuvwxyz])s( |$)", "\\1 ", x)[0].strip() for x in f]
    data = [re.subn("s( |$)", " ", x)[0].strip() for x in data]
    data = [re.subn("ing( |$)", " ", x)[0].strip() for x in data]
    data = [x.replace("tard ", " ") for x in data]

    ##stemming
    data = [re.subn(" [*$%&#@][*$%&#@]+", " xexp ", x)[0].strip() for x in data]
    data = [re.subn(" [0-9]+ ", " DD ", x)[0].strip() for x in data]
    data = [re.subn("<\S*>", "", x)[0].strip() for x in data]


    ##handling spaces
    data=[x.strip() for x in data]
    data=[" ".join(x.split()) for x in data]


    data=[x.lower() for x in data]
    data=remove_stopwords(data,stop_words)

    return data

def remove_stopwords(data,stop_words):
    result=[]
    for comment in data:
        comment=comment.split(' ')
        comment=[x for x in comment if (x not in stop_words) ]
        result.append(' '.join(comment))
    return result

def lexicon_score():

    pass

###to change
#def feature_extraction_selecttion(k):
def feature_extraction_selecttion(train_data,labels,k):
    # res=[]
    # ### Feature extraction
    # ##TODO: Build charchter n-grams
    print("extracting character 3-grams\n")
    all_char_3grams = []
    for comment in train_data:
        all_char_3grams = all_char_3grams + word_grams(comment, 3, 4)
    all_char_3grams = [x for x in all_char_3grams if x]
    all_char_3grams = list(set(all_char_3grams))

    print("extracting character 4-grams\n")

    all_char_4grams = []
    for comment in train_data:
        all_char_4grams = all_char_4grams + word_grams(comment, 4, 5)
    all_char_4grams = list(set(all_char_4grams))

    print("extracting character 5-grams\n")

    all_char_5grams = []
    for comment in train_data:
        all_char_5grams = all_char_5grams + word_grams(comment, 5, 6)
    all_char_5grams = list(set(all_char_5grams))

    selected_char_features=char_ngrams_select(train_data,labels,all_char_3grams,all_char_4grams,all_char_5grams,k)
    #selected_char_features=char_ngrams_select(k)
    #
    #
    ### Saving all ngrams in a file for Future uses
    all_ngrams = all_char_5grams + all_char_4grams + all_char_3grams
    df = pd.DataFrame({'id': range(len(all_ngrams))})
    df['ngrams'] = pd.Series(all_ngrams, index=df.index[:len(all_ngrams)])
    df.to_csv(CHAR_NGRAMS_VALUES_FILE)
    df=[]

    ##Build word n-grams
    all_word_1grams=[]
    for comment in train_data:
        all_word_1grams=all_word_1grams+word_grams(comment.split(' '),1,2)
    all_word_1grams=[x for x in all_word_1grams if x]
    all_word_1grams=list(set(all_word_1grams))

    all_word_2grams = []
    for comment in train_data:
        all_word_2grams = all_word_2grams + word_grams(comment.split(' '), 2, 3)
    all_word_2grams=list(set(all_word_2grams))

    all_word_3grams = []
    for comment in train_data:
        all_word_3grams = all_word_3grams + word_grams(comment.split(' '), 3, 4)
    all_word_3grams=list(set(all_word_3grams))

    ### Saving all ngrams in a file for Future uses
    all_ngrams=all_word_1grams+all_word_2grams+all_word_3grams
    df = pd.DataFrame({'id': range(len(all_ngrams))})
    df['ngrams'] = pd.Series(all_ngrams, index=df.index[:len(all_ngrams)])
    df.to_csv(NGRAMS_VALUES_FILE)

###end of feature extraction


    ##TODO: Features Selection:
    ###to change
    #selected_features=ngrams_select(k)
    selected_features=ngrams_select(train_data,labels,all_word_1grams,all_word_2grams,all_word_3grams,k)
    #selected_features=ngrams_select(k)

    return selected_char_features, selected_features


def word_grams(comment, min=1, max=4):
    s = []
    for n in range(min, max):
        if(len(comment)>=min):
            for ngram in ngrams(comment, n):
                s.append(' '.join(str(i) for i in ngram))
    s=list(set(s))
    return s

def char_grams(comment, min=1, max=4):
    s = []
    for n in range(min, max):
        if(len(comment)>=min):
            for ngram in ngrams(comment, n):
                s.append(' '.join(str(i) for i in ngram))
    s=list(set(s))

    return s

#def char_ngrams_select(k):
def char_ngrams_select(data, labels, char_3grams, char_4grams , char_5grams,k):

    l1 =len(char_3grams)
    l2 = len(char_4grams)
    l3 =len(char_5grams)
    l=l1+l2+l3
    N00=[1]*l
    N01=[1]*l
    N11=[1]*l
    N10=[1]*l
    print("starting ngrams selection")
    i=0;
    for comment in data:
        if(i%100==0):
            print("Feature Selection : processed Comment : ",i)
        comment_3grams=word_grams(comment,3,4)
        comment_4grams=word_grams(comment,4,5)
        comment_5grams=word_grams(comment,5,6)

        if (labels[i] == 0):
            N00=[(x+1) for x in N00]
            for ngram in comment_3grams:
                N10[char_3grams.index(ngram)]+=1
                N00[char_3grams.index(ngram)]-= 1
            for ngram in comment_4grams:
                N10[l1+char_4grams.index(ngram)] += 1
                N00[l1+char_4grams.index(ngram)] -= 1
            for ngram in comment_5grams:
                N10[l1+l2+char_5grams.index(ngram)]+=1
                N00[l1+l2+char_5grams.index(ngram)]-= 1

        if (labels[i] == 1):
            N01 = [(x + 1) for x in N01]
            for ngram in comment_3grams:
                #print(word_1grams.index(ngram))
                N11[char_3grams.index(ngram)] += 1
                N01[char_3grams.index(ngram)]-= 1
            for ngram in comment_4grams:
                N11[l1+char_4grams.index(ngram)] += 1
                N01[l1+char_4grams.index(ngram)]-= 1
            for ngram in comment_5grams:
                N11[l1+l2+char_5grams.index(ngram)] += 1
                N01[l1+l2+char_5grams.index(ngram)]-= 1
        i+=1
    ## ranking features
    ranked_ngrams=[]
    chi2_ngrams=[chi2(N00[i],N01[i],N11[i],N10[i]) for i in range(l)]
    q = Q.PriorityQueue()
    for i in range(l):
       q.put((-chi2_ngrams[i],i))
    ngrams_rank=open(CHAR_NGRAMS_RANK_FILE,"w")
    print("best features : ")
    ngrams_rank.write("rank,id\n")
    while not q.empty():
        a=tuple(map(abs,q.get()))
        a=tuple(map(str,a))
        ngrams_rank.write(','.join(a))
        ngrams_rank.write('\n')
    ngrams_rank.close()
    comment_3grams=[]
    comment_4grams=[]
    comment_5grams=[]

### end of features selection


    ranks=pd.read_csv(CHAR_NGRAMS_RANK_FILE)
    ngrams=pd.read_csv(CHAR_NGRAMS_VALUES_FILE)

    idies=ranks[:k].id.tolist()
    selectedKngrams=ngrams[ngrams['id'].isin(idies)].ngrams.tolist()
    return selectedKngrams



###to change
#def ngrams_select(k):

#def ngrams_select(k):
def ngrams_select(data, labels, word_1grams, word_2grams , word_3grams,k):

    l1 =len(word_1grams)
    l2 = len(word_2grams)
    l3 =len(word_3grams)
    l=l1+l2+l3
    N00=[1]*l
    N01=[1]*l
    N11=[1]*l
    N10=[1]*l
    print("starting ngrams selection")
    i=0;
    for comment in data:
        print("Feature Selection : processing Comment : ",i)
        comment_1grams=word_grams(comment.split(' '),1,2)
        comment_2grams=word_grams(comment.split(' '),2,3)
        comment_3grams=word_grams(comment.split(' '),3,4)
        if (labels[i] == 0):
            N00=[(x+1) for x in N00]
            for ngram in comment_1grams:
                N10[word_1grams.index(ngram)]+=1
                N00[word_1grams.index(ngram)]-= 1
            for ngram in comment_2grams:
                N10[l1+word_2grams.index(ngram)] += 1
                N00[l1+word_2grams.index(ngram)] -= 1
            for ngram in comment_3grams:
                N10[l1+l2+word_3grams.index(ngram)]+=1
                N00[l1+l2+word_3grams.index(ngram)]-= 1

        if (labels[i] == 1):
            N01 = [(x + 1) for x in N01]
            for ngram in comment_1grams:
                #print(word_1grams.index(ngram))
                N11[word_1grams.index(ngram)] += 1
                N01[word_1grams.index(ngram)]-= 1
            for ngram in comment_2grams:
                N11[l1+word_2grams.index(ngram)] += 1
                N01[l1+word_2grams.index(ngram)]-= 1
            for ngram in comment_3grams:
                N11[l1+l2+word_3grams.index(ngram)] += 1
                N01[l1+l2+word_3grams.index(ngram)]-= 1
        i+=1
    ## ranking features
    ranked_ngrams=[]
    chi2_ngrams=[chi2(N00[i],N01[i],N11[i],N10[i]) for i in range(l)]
    q = Q.PriorityQueue()
    for i in range(l):
       q.put((-chi2_ngrams[i],i))
    ngrams_rank=open(NGRAMS_RANK_FILE,"w")
    print("best features : ")
    ngrams_rank.write("rank,id\n")
    while not q.empty():
        a=tuple(map(abs,q.get()))
        a=tuple(map(str,a))
        ngrams_rank.write(','.join(a))
        ngrams_rank.write('\n')
    ngrams_rank.close()
    word_1grams=[]
    word_2grams=[]
    word_3grams=[]

# ### end of features selection


    ranks=pd.read_csv(NGRAMS_RANK_FILE)
    ngrams=pd.read_csv(NGRAMS_VALUES_FILE)

    idies=ranks[:k].id.tolist()
    selectedKngrams=ngrams[ngrams['id'].isin(idies)].ngrams.tolist()
    return selectedKngrams



def chi2(N00, N01, N11, N10):
    N=N00+N01+N11+N10
    res=(N*pow(((N11*N00)+(N10*N01)),2))/( (N11+N01)*(N11+N10)*(N10+N00)*(N01+N00))
    return res


def train(train_data,labels,selected_char_features, selected_features):
    finalfeatures=selected_features+selected_char_features
    print(selected_features)
    occurences=[[] for _ in range(len(train_data))]
    i=0
    for comment in train_data:
        #print(i)
        occurences[i]=([comment.count(x) for x in finalfeatures])

        i+=1

    print("training SVM\n")
    svc = svm.SVC(kernel='linear', C=1, gamma='auto').fit(occurences, labels)

    return svc

def test(test_data,selected_char_features, selected_features,svc):
    occurences_test = [[] for _ in range(len(test_data))]
    finalfeatures=selected_features+selected_char_features

    i = 0
    for comment in test_data:
        #print(i)
        occurences_test[i]=([comment.count(x) for x in finalfeatures])
        i += 1
    y_predicted = svc.predict(occurences_test)

    print(y_predicted)
    y_predicted=y_predicted.tolist()
    df = pd.DataFrame({'id': range(len(y_predicted))})
    df['prediction'] = pd.Series(y_predicted, index=df.index[:len(y_predicted)])
    df.to_csv(RESULT_FILE)
    return y_predicted

def evaluate(y_true,y_pred):
    print("\n\n\n Evalution :\n")
    print("ROC Area Under the Curve Score: \n")
    print(roc_auc_score(y_true, y_pred))
    print("Acuarracy :\n")
    print(accuracy_score(y_true, y_pred))
    print("Confusion Matrix :\n")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:\n")
    target_names = ['Not Insulting', 'Insulting']
    print(classification_report(y_true, y_pred, target_names=target_names))



def run():
    print("oppenning files")
    train_data, train_labels, test_data, test_sol_data, test_sol_labels, stop_words=openFiles()
    print("Files oppened\n")

    print("Preprocessing The Data")
    train_data, test_data, test_sol_data = preprocessing(train_data, test_data, test_sol_data, stop_words)
    print("Data Preprocessed\n")
    print(len(train_data))
    print("Feature extractin and selection")
    ###to change
    #final_features = feature_extraction_selecttion(350)
    selected_char_features, selected_features=feature_extraction_selecttion(train_data,train_labels,1000)
    svc=train(train_data,train_labels,selected_char_features, selected_features)
    print("Testing\n")
    predicted=test(test_sol_data,selected_char_features, selected_features,svc)
    print("Evaluating\n")
    evaluate(test_sol_labels,predicted)


if __name__ == '__main__':
    run()
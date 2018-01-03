from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, f1_score
import pandas as pd

import scikitplot as skplt
import matplotlib.pyplot as plt



DIR_PATH = "./../resources/"
TEST_SOL_FILE   = DIR_PATH + "test_with_solutions.csv"
PREDICTIONS_FILE = DIR_PATH + "result.csv"
TRAIN_FILE      = DIR_PATH + "train.csv"


def openFiles():
    test_sol_data=pd.read_csv(TEST_SOL_FILE)
    test_sol_labels=test_sol_data.Insult.tolist()
    predicted=pd.read_csv(PREDICTIONS_FILE)
    predicted=predicted.prediction.tolist()
    train_data=pd.read_csv(TRAIN_FILE)
    train_labels=train_data.Insult.tolist()
    return test_sol_labels, predicted, train_labels


def evaluate(y_true,y_pred):
    print("Acuarracy :\n")
    print(accuracy_score(y_true, y_pred))
    print("ROC Area Under the Curve Score: \n")
    print(roc_auc_score(y_true, y_pred))
    print("Confusion Matrix :\n")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:\n")
    target_names = ['Not Insulting', 'Insulting']
    print(classification_report(y_true, y_pred, target_names=target_names))

    print("Plotting\n")
    ## Confussion matrix
    skplt.metrics.plot_confusion_matrix(y_true,y_pred)
    plt.show()

    ## ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    # Print ROC curve
    plt.plot(fpr, tpr)
    plt.show()

def stats_on_data(labels):
    df_train = pd.DataFrame({'id': range(len(labels))})
    df_train['label'] = pd.Series(labels, index=df_train.index[:len(labels)])
    df_train.label.hist()
    plt.show()




def run():
    print("Reading file :predictions and test")
    test_sol_labels, predicted, train_labels=openFiles()
    print("Data loaded")
    print("Plotting Stastics on Training Data")
    stats_on_data(train_labels)

    print("Plotting Stastics on Testing Data")
    stats_on_data(test_sol_labels)

    print("Launching Evalution")
    evaluate(test_sol_labels,predicted)
if __name__ == '__main__':
    run()
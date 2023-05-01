import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io
import scipy.interpolate
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline, Pipeline
from scipy.stats import gaussian_kde

mpl.use('TkAgg')
DATA_FILE_PATH = "./DataFiles/feaSubE"
DATA_TYPE = "Overt"


def readDataFile(name):
    dataF = pd.read_csv(DATA_FILE_PATH + name + ".csv", header=None)
    return dataF


# This function shows the location of one channel on head.
#
# INPUTS:
#   chanVal is a vector with the weight of the channels to be plotted.
def show_chanWeights(chanVal_orig, title):
    chanVal = abs(chanVal_orig)
    matlab_offset = 1
    selNum = np.asarray(range(1, 306))
    cortIX = np.where(np.mod(selNum, 3) != 0)
    selNum = selNum[cortIX]
    resolution = 200
    # Load sensor location
    # load sensors102.mat
    mat = scipy.io.loadmat('./DataFiles/sensors102.mat')
    c102 = mat['c102']
    x = c102[:, 2 - matlab_offset]
    y = c102[:, 3 - matlab_offset]
    xlin = np.linspace(min(x), max(x) + 35, resolution)
    ylin = np.linspace(min(y), max(y), resolution)
    r = 5

    MinChanVal = min(chanVal)
    z = np.ones(len(x)) * MinChanVal

    selSen = np.ceil(selNum / 3)

    maxSen = int(max(selSen))
    for senIX in range(1, maxSen):
        currVal = np.zeros(2)
        for chanIX in range(1, 2):
            chanInd = (senIX - 1) * 3 + chanIX
            tmp = np.where(selNum == chanInd)
            if len(tmp) != 0:
                currVal[chanIX - matlab_offset] = chanVal[tmp]
        z[senIX] = max(currVal)

    X, Y = np.meshgrid(xlin, ylin)
    Z = scipy.interpolate.griddata((x, y), z, (X, Y), method='cubic')
    # pcm = plt.pcolor([X, Y], Z)
    plt.pcolor(Z)
    plt.axis('equal')  # ax.axis('equal')
    plt.axis('off')
    plt.colorbar()
    plt.title(title)
    plt.savefig("./Results/" + DATA_TYPE + "_brain_weights.png")
    plt.show()


def plotVector(cVec, title):
    xVals = np.arange(0, len(cVec), 1)
    plt.plot(xVals, cVec)
    plt.xlabel("Weight Index")
    plt.ylabel("Weight Value")
    plt.grid()
    plt.title(title)
    plt.savefig("./Results/Checkpoint1/" + title.replace(" ", "_") + ".png")


def generateFold1Plots(clf):
    coefs = clf.named_steps['svm'].coef_.T
    plt.stem(coefs)
    plt.title("Model Tested on Fold 1 Electrode Channel Weights - " + DATA_TYPE)
    plt.grid()
    plt.savefig("./Results/" + DATA_TYPE + "_stem_plot.png")
    plt.show()

    show_chanWeights(coefs, DATA_TYPE + " Model Tested on Fold 1 Channel Weights")

    # Get the indices of the sorted array in ascending order
    coefs = np.concatenate(coefs, axis=0)

    sorted_indices = np.argsort(np.abs(coefs))[::-1]
    data = []
    for i in range(6):
        data.append([sorted_indices[i], coefs[sorted_indices[i]]])
    # Create a figure and axis object
    fig, ax = plt.subplots()
    # Hide axis
    ax.axis('off')
    table = ax.table(cellText=data, loc='center')
    table.set_fontsize(14)

    # Set the size of the table
    table.scale(1, 2)
    plt.title("Model Tested on Fold 1 Top 6 Electrode Weights - " + DATA_TYPE)
    plt.savefig("./Results/top_electrode_weights_table_" + DATA_TYPE + ".png")
    plt.show()


def crossVal(X, y):
    # shuffle data
    X['C'] = y
    X = X.sample(frac=1).reset_index(drop=True)
    y = X['C'].values
    X = X.drop(columns='C')
    # layer 1 CV
    cv = KFold(n_splits=6)
    finalModel = None
    clfScores = []
    clfList = []
    alphasRes = []
    ROCDataframes = []
    count = 0
    for train_index, test_index in cv.split(X):
        count += 1
        layer_2 = KFold(n_splits=5)
        curr_train_X = X.iloc[train_index]
        curr_train_y = y[train_index]
        curr_test_X = X.iloc[test_index]
        curr_test_y = y[test_index]
        clf_curr_fold = None
        current_alpha_val = None
        for lay_2_train_idx, lay_2_test_idx in layer_2.split(curr_train_X):
            alphas = np.logspace(-15, 15, num=13 * 3 + 1, base=10)
            scores = []
            for alpha in alphas:
                clf = make_pipeline(LinearSVC(dual=False, max_iter=100000))
                clf.set_params(linearsvc__C=alpha)
                curr_train_X_l2 = curr_train_X.iloc[lay_2_train_idx]
                curr_train_y_l2 = curr_train_y[lay_2_train_idx].T
                curr_test_X_l2 = curr_train_X.iloc[lay_2_test_idx]
                curr_test_y_l2 = curr_train_y[lay_2_test_idx].T
                clf.fit(curr_train_X_l2, curr_train_y_l2)
                # make predictions to determine alpha Pcd
                preds = clf.predict(curr_test_X_l2)
                correct = 0
                for i in range(len(preds)):
                    if (preds[i] == curr_test_y_l2[i]):
                        correct += 1
                pCd = correct / len(preds)
                scores.append(pCd)
            maxpcd = 0
            idx = -1
            for i in range(len(scores)):
                if (scores[i] > maxpcd):
                    maxpcd = scores[i]
                    idx = i
            optimalAlpha = alphas[idx]
            clf_curr_fold = Pipeline([
                ('svm', LinearSVC(dual=False, C=optimalAlpha, max_iter=100000))
            ])
            current_alpha_val = optimalAlpha
        clf_curr_fold.fit(curr_train_X, curr_train_y)
        alphasRes.append(current_alpha_val)
        if (count == 1):
            generateFold1Plots(clf_curr_fold)
        clfList.append(clf_curr_fold)
        preds = clf_curr_fold.predict(curr_test_X)
        dec_stats = clf_curr_fold.decision_function(curr_test_X)
        single_fold_ROC_df = pd.DataFrame(np.column_stack((curr_test_y, dec_stats)))
        ROCDataframes.append(single_fold_ROC_df)
        correct = 0
        for i in range(len(preds)):
            if preds[i] == curr_test_y[i]:
                correct += 1
        pCd = correct / len(preds)
        clfScores.append(pCd)
    # select final model
    idx = -1
    finalScore = 0
    for i in range(len(clfScores)):
        if (clfScores[i] > finalScore):
            finalScore = clfScores[i]
            idx = i

    plotHolisticROC(ROCDataframes)
    return clfList[idx], alphasRes


def trainSVM(data0, data1, data0tes, data1tes):
    data0 = data0.T
    data1 = data1.T
    data0.insert(loc=0, column='', value=0)
    data1.insert(loc=0, column='', value=1)

    data0tes = data0tes.T
    data1tes = data1tes.T
    data0tes.insert(loc=0, column='', value=0)
    data1tes.insert(loc=0, column='', value=1)

    # Prepare Data
    X_test = pd.concat([data0tes, data1tes], ignore_index=True)
    y_test = X_test.iloc[:, 0]
    X_test = X_test.iloc[:, 1:]

    # Prepare Data
    X = pd.concat([data0, data1], ignore_index=True)
    y = X.iloc[:, 0]
    X = X.iloc[:, 1:]

    # clf, alphas = crossVal(X, y)
    # x = np.arange(1, 7, 1)

    # alphas = np.log10(alphas)
    # plt.stem(x, alphas)
    # plt.ylim(-20, 20)
    # plt.grid()
    # plt.title(DATA_TYPE + " Log10 of Regularization Parameter Values By Fold")
    # plt.savefig("./Results/" + DATA_TYPE + "_alpha_values.png")
    # plt.show()
    clf1 = LinearSVC(max_iter=100000, dual=False)
    clf1.fit(X, y)
    # predict testing data with decision function
    preds1 = clf1.decision_function(X_test)
    testROC = pd.DataFrame(preds1)
    testROC.insert(loc=0, column='', value=y_test)
    # Fit other model
    clf2 = LinearSVC(max_iter=100000, dual = False)
    clf2.fit(X_test, y_test)
    preds2 = clf2.decision_function(X)
    testROC2 = pd.DataFrame(preds2)
    testROC2.insert(loc=0, column='', value=y_test)

    # Fit inces model
    clf3 = LinearSVC(max_iter=100000, dual=False)
    clf3.fit(X, y)
    preds3 = clf3.decision_function(X)
    testROC3 = pd.DataFrame(preds3)
    testROC3.insert(loc=0, column='', value=y)

    # Fit inces model
    clf4 = LinearSVC(max_iter=100000, dual=False)
    clf4.fit(X_test, y_test)
    preds4 = clf4.decision_function(X_test)
    testROC4 = pd.DataFrame(preds4)
    testROC4.insert(loc=0, column='', value=y_test)

    # generate ROC
    fig, ax = plt.subplots()
    other = "Overt" if DATA_TYPE == "Img" else "Img"
    plotROC(testROC, ax, "blue", "Trained " + DATA_TYPE + " Tested " + other)
    plotROC(testROC2, ax, "orange", "Trained " + other + " Tested " + DATA_TYPE)
    plotROC(testROC3, ax, "pink", "Trained " + DATA_TYPE + " Tested " + DATA_TYPE)
    plotROC(testROC4, ax, "green", "Trained " + other + " Tested " + other)


    ax.set_xlabel('Probability False Alarm')
    ax.set_ylabel('Probability Detection')
    ax.grid()
    fig.suptitle("Cross-Tested ROCs")
    plt.legend()
    plt.savefig("./Results/cross_tested_ROC.png")

    plt.show()


def plotHolisticROC(dfList):
    finalDf = pd.DataFrame()
    for df in dfList:
        finalDf = pd.concat([finalDf, df])
    dfList.append(finalDf)
    fig, ax = plt.subplots()
    accuracies = []
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
    for i in range(7):
        label = "Full ROC" if (i == 6) else ("ROC Testing `Fold` " + str(i + 1))
        accuracies.append(plotROC(dfList[i], ax, colors[i], label=label))
    ax.set_xlabel('Probability False Alarm')
    ax.set_ylabel('Probability Detection')
    ax.legend()
    ax.grid()
    fig.suptitle(DATA_TYPE + " Cross-Validated ROCs")
    plt.savefig("./Results/" + DATA_TYPE + "_cross_validated_ROCS.png")
    plt.show()

    data = []
    # accuracies[6] = round((np.mean(accuracies[:6])),2)
    for i in range(7):
        label = "Full ROC" if (i == 6) else ("ROC Testing Fold " + str(i + 1))
        data.append([label, str(round(accuracies[i], 2)) + "%"])
    # Create a figure and axis object
    fig, ax = plt.subplots()
    # Hide axis
    ax.axis('off')
    table = ax.table(cellText=data, loc='center')
    table.set_fontsize(14)

    # Set the size of the table
    table.scale(1, 2)
    plt.title("Fold Accuracies P(cd) - " + DATA_TYPE)
    plt.savefig("./Results/fold_accuracies_" + DATA_TYPE + ".png")
    plt.show()


def plotROC(dataFrame, ax, color, label):
    # true class stored in first column
    trueClass = dataFrame.iloc[:, 0]
    # compute desicion stats and store in col 2
    decisionStats = dataFrame.iloc[:, 1]
    thresholds = dataFrame.iloc[:, 1:].values.tolist()
    if 0 not in thresholds:
        thresholds.append([0])
    thresholds = sorted(sum(thresholds, []))

    tdr = []
    fpr = []
    pcdLis = []
    accuracy = []
    for threshold in thresholds:
        threshPredict = (decisionStats >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(trueClass, threshPredict).ravel()
        pcd = (tn + tp) / (tn + fp + fn + tp)
        if threshold == 0:
            accuracy = pcd * 100
        pcdLis.append(pcd)
        tdr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))
    ax.plot(fpr, tdr, color=color, label=label)
    return accuracy


def KDEPdf(class0, class1):
    # Fit a KDE to both classes
    kde0 = gaussian_kde(class0.T)
    kde1 = gaussian_kde(class1.T)

    # evaluate the KDE at a range of values
    x_vals = np.linspace(np.min(class0), np.max(class1), 100)
    y_vals0 = kde0(x_vals)
    y_vals1 = kde1(x_vals)

    # plot the Estimated PDFs
    plt.plot(x_vals, y_vals0, color="blue", label="Class 0")
    plt.plot(x_vals, y_vals1, color="orange", label="Class 1")
    plt.grid()
    plt.xlabel("SVM Value")
    plt.ylabel("Density")
    plt.title(DATA_TYPE + " KDE PDFs")
    plt.legend()
    plt.savefig("./Results/Checkpoint2/" + DATA_TYPE + "_KDE.png")
    plt.show()
    return kde0, kde1


overtClass0 = readDataFile("Overt_1")
overtClass1 = readDataFile("Overt_2")

imgClass0 = readDataFile("Img_1")
imgClass1 = readDataFile("Img_2")

trainSVM(overtClass0, overtClass1, imgClass0, imgClass1)

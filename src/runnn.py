import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import mlrose_hiive
from sklearn.metrics import accuracy_score, f1_score
import time

act = 'sigmoid'
def calc_mean(optlist):
    rhcarr = np.array(optlist)
    rhcmax = np.max([len(a) for a in rhcarr])
    rhcarr = np.asarray([np.pad(a, (0, rhcmax - len(a)), 'edge',) for a in rhcarr])
    rhcmean = np.mean(rhcarr,axis=0)
    return rhcmean

def runnn(nn,xTrain, yTrain, xTest, yTest):
    start = time.time()
    nn.fit(xTrain, yTrain)
    runtime = time.time() - start
    # Predict labels for train set and assess accuracy
    y_train_pred = nn.predict(xTrain)
    y_train_accuracy = accuracy_score(yTrain, y_train_pred)
    trainf1 = f1_score(yTrain,y_train_pred)
    #print(kind + ' train: ' +  str(y_train_accuracy))
    # Predict labels for test set and assess accuracy
    y_test_pred = nn.predict(xTest)

    y_test_accuracy = accuracy_score(yTest, y_test_pred)
    testf1 = f1_score(yTest, y_test_pred)

    #print(kind + ' test: ' +  str(y_test_accuracy))
    return nn.fitness_curve,y_train_accuracy, y_test_accuracy, trainf1, testf1,runtime

def rhctune(x1, y1,  average, arch):
    #restarts
    #learning_rate:
    learning_rates = [ .1, 1.5 , 3, 5]
    restarts = [1, 4, 8, 20]
    curves = []
    testaccuracy = []
    trainaccuracy = []
    trainf1 = []
    testf1 = []
    times = []
    avg = 1
    if average == True:
        avg = 3
    #plt.figure()
    print('rhc ' + str(len(arch)) + ' layer(s) ' + str(arch[0]))
    print('lr rand time Train Accuracy Test Accuracy f1 train f1 test ')
    for l in learning_rates:
        for r in restarts:
            curves = []
            testaccuracy = []
            trainaccuracy = []
            trainf1 = []
            testf1 = []
            times = []
            for i in range(avg):

                x1Train, x1Test, y1Train, y1Test = train_test_split(x1, y1, test_size=0.22,random_state=i)
                scaler = StandardScaler()
                scaler.fit(x1Train)
                x1Train = scaler.transform(x1Train)
                # apply same transformation to test data
                x1Test = scaler.transform(x1Test)
                nnrhc = mlrose_hiive.NeuralNetwork(hidden_nodes = arch, activation =act,
                                 algorithm ='random_hill_climb', curve=True,
                                 max_iters = 1000, bias = True, is_classifier = True,
                                 learning_rate = l, early_stopping = True, clip_max=5,
                                 max_attempts = 100, random_state = i, restarts=r)
                curve, trainacc, testacc, f1train, f1test,traintime = runnn(nnrhc,x1Train,y1Train,x1Test,y1Test)
                curves.append(curve[:,0])
                trainaccuracy.append(trainacc)
                testaccuracy.append(testacc)
                trainf1.append(f1train)
                testf1.append(f1test)
                times.append(traintime)

            avgcurve = np.negative(calc_mean(curves))
            avgtime = np.mean(times)
            avgtrainacc = np.mean(trainaccuracy)
            avgtestacc = np.mean(testaccuracy)
            avgf1tr = np.mean(trainf1)
            avgf1te = np.mean(testf1)
            plt.plot(avgcurve,label=str('res ' + str(r) + ' lr ' + str(l)))
            print ("%.2f %d %.5f %.5f %.5f %.5f %.5f" % (l, r, avgtime, avgtrainacc, avgtestacc,avgf1tr,avgf1te))
    plt.title('rhc ' + str(arch[0]) + ' layers of ' + str(len(arch)) + ' nodes tuning ')
    plt.legend(loc='best')
    plt.savefig('./nn/rhc' + str(len(arch)) + 'layers-' + str(arch[0]) + 'cm5.PNG')
    plt.close()
#def satune(xTrain, yTrain, xTest, yTest):
    #temperatuve
    #constant
    #learning_rate

def satune(x1, y1,  average, arch):
    learning_rates = [ .1, 1.5 , 3, 5]
    temps = [.1, 1, 10, 100, 300]
    curves = []
    testaccuracy = []
    trainaccuracy = []
    trainf1 = []
    testf1 = []
    times = []
    avg = 1
    if average == True:
        avg = 3
    #plt.figure()
    print('sa ' + str(len(arch)) + ' layer(s) ' + str(arch[0]))
    print('lr temp time Train Accuracy Test Accuracy f1 train f1 test ')
    for l in learning_rates:
        for t in temps:
            curves = []
            testaccuracy = []
            trainaccuracy = []
            trainf1 = []
            testf1 = []
            times = []
            for i in range(avg):
                x1Train, x1Test, y1Train, y1Test = train_test_split(x1, y1, test_size=0.22,random_state=i)
                scaler = StandardScaler()
                scaler.fit(x1Train)
                x1Train = scaler.transform(x1Train)
                # apply same transformation to test data
                x1Test = scaler.transform(x1Test)
                nnrhc = mlrose_hiive.NeuralNetwork(hidden_nodes = arch, activation =act,
                                 algorithm ='simulated_annealing', curve=True,
                                 max_iters = 1000, bias = True, is_classifier = True,
                                 learning_rate = l, early_stopping = True, clip_max=5,
                                 max_attempts = 100, random_state = i, schedule=mlrose_hiive.ExpDecay(init_temp=t,exp_const=.1))
                curve, trainacc, testacc, f1train, f1test,traintime = runnn(nnrhc,x1Train,y1Train,x1Test,y1Test)
                curves.append(curve[:,0])
                trainaccuracy.append(trainacc)
                testaccuracy.append(testacc)
                trainf1.append(f1train)
                testf1.append(f1test)
                times.append(traintime)

            avgcurve = np.negative(calc_mean(curves))
            avgtime = np.mean(times)
            avgtrainacc = np.mean(trainaccuracy)
            avgtestacc = np.mean(testaccuracy)
            avgf1tr = np.mean(trainf1)
            avgf1te = np.mean(testf1)
            plt.plot(avgcurve,label=str('temp ' + str(t) + ' lr ' + str(l)))
            print ("%.2f %.1f %.5f %.5f %.5f %.5f %.5f" % (l, t, avgtime, avgtrainacc, avgtestacc,avgf1tr,avgf1te))
    plt.title('sa ' + str(arch[0]) + ' layers of ' + str(len(arch)) + ' nodes tuning ')
    plt.legend(loc='best')
    plt.savefig('./nn/sa/' + str(len(arch)) + 'layers-' + str(arch[0]) + 'cm5.PNG')
    plt.close()

def gatune(x1, y1, average, arch):
    learning_rates = [ .001, .05, .1, .3, 1.5 , 3]
    mutations = [.01, .05, .1, .33, .66, .99]
    curves = []
    testaccuracy = []
    trainaccuracy = []
    trainf1 = []
    testf1 = []
    times = []
    avg = 1
    if average == True:
        avg = 3
    #plt.figure()
    print('ga ' + str(len(arch)) + ' layer(s) ' + str(arch[0]))
    print('lr mut time Train Accuracy Test Accuracy f1 train f1 test ')
    for l in learning_rates:
        for m in mutations:
            curves = []
            testaccuracy = []
            trainaccuracy = []
            trainf1 = []
            testf1 = []
            times = []
            for i in range(avg):
                x1Train, x1Test, y1Train, y1Test = train_test_split(x1, y1, test_size=0.22,random_state=i)
                scaler = StandardScaler()
                scaler.fit(x1Train)
                x1Train = scaler.transform(x1Train)
                # apply same transformation to test data
                x1Test = scaler.transform(x1Test)
                nnrhc = mlrose_hiive.NeuralNetwork(hidden_nodes = arch, activation =act,
                                 algorithm ='genetic_alg', curve=True,
                                 max_iters = 100, bias = True, is_classifier = True,
                                 learning_rate = l, early_stopping = True, clip_max=5,
                                 max_attempts = 10, random_state = i, mutation_prob=m)
                curve, trainacc, testacc, f1train, f1test,traintime = runnn(nnrhc,x1Train,y1Train,x1Test,y1Test)
                curves.append(curve[:,0])
                trainaccuracy.append(trainacc)
                testaccuracy.append(testacc)
                trainf1.append(f1train)
                testf1.append(f1test)
                times.append(traintime)

            avgcurve = np.negative(calc_mean(curves))
            avgtime = np.mean(times)
            avgtrainacc = np.mean(trainaccuracy)
            avgtestacc = np.mean(testaccuracy)
            avgf1tr = np.mean(trainf1)
            avgf1te = np.mean(testf1)
            plt.plot(avgcurve,label=str('mutp ' + str(m) + ' lr ' + str(l)))
            print ("%.3f %.3f %.5f %.5f %.5f %.5f %.5f" % (l, m, avgtime, avgtrainacc, avgtestacc,avgf1tr,avgf1te))
    plt.title('ga ' + str(arch[0]) + ' layers of ' + str(len(arch)) + ' nodes tuning ')
    plt.legend(loc='best')
    plt.savefig('./nn/ga/' + str(len(arch)) + 'layers-' + str(arch[0]) + 'cm5.PNG')
    plt.close()

def gdtune(x1, y1, average, arch):
    learning_rates = [ .0001, .0005,.001, .0015, .002, .003, .004]
    #mutations = [.1, .33, .66, .99]
    curves = []
    testaccuracy = []
    trainaccuracy = []
    trainf1 = []
    testf1 = []
    times = []
    avg = 1
    if average == True:
        avg = 3
    #plt.figure()
    print('gd ' + str(len(arch)) + ' layer(s) ' + str(arch[0]))
    print('time Train Accuracy Test Accuracy f1 train f1 test ')
    for l in learning_rates:
            curves = []
            testaccuracy = []
            trainaccuracy = []
            trainf1 = []
            testf1 = []
            times = []
            for i in range(avg):
                x1Train, x1Test, y1Train, y1Test = train_test_split(x1, y1, test_size=0.22,random_state=i)
                scaler = StandardScaler()
                scaler.fit(x1Train)
                x1Train = scaler.transform(x1Train)
                # apply same transformation to test data
                x1Test = scaler.transform(x1Test)
                nnrhc = mlrose_hiive.NeuralNetwork(hidden_nodes = arch, activation =act,
                                 algorithm ='gradient_descent', curve=True,
                                 max_iters = 3000, bias = True, is_classifier = True,
                                 learning_rate = l, early_stopping = True, clip_max=5,
                                 max_attempts = 100, random_state = i)
                curve, trainacc, testacc, f1train, f1test,traintime = runnn(nnrhc,x1Train,y1Train,x1Test,y1Test)
                curves.append(curve)
                trainaccuracy.append(trainacc)
                testaccuracy.append(testacc)
                trainf1.append(f1train)
                testf1.append(f1test)
                times.append(traintime)

            avgcurve = calc_mean(curves)
            avgtime = np.mean(times)
            avgtrainacc = np.mean(trainaccuracy)
            avgtestacc = np.mean(testaccuracy)
            avgf1tr = np.mean(trainf1)
            avgf1te = np.mean(testf1)
            plt.plot(avgcurve,label=str('lr ' + str(l)))
            print ("%.5f %.5f %.5f %.5f %.5f %.5f" % (l, avgtime, avgtrainacc, avgtestacc,avgf1tr,avgf1te))
    plt.title('gd ' + str(arch[0]) + ' layers of ' + str(len(arch)) + ' nodes tuning ')
    plt.legend(loc='best')
    plt.savefig('./nn/gd/' + str(len(arch)) + 'layers-' + str(arch[0]) + 'cm5.PNG')
    plt.close()


file1 = "original.csv" #credit data
#file1 = "drug200.csv" #drug identification set
#file2 = "cardio_train.csv"
data1 = pd.read_csv("../data/" + file1)
#data2 = pd.read_csv("../data/telecom_churn.csv")
#data2 = pd.read_csv("../data/cardio_train.csv")
#print(data[data.columns[0:5]].head())
data1 = data1.dropna()
x1 = data1[data1.columns[1:4]]
y1 = data1['default']
'''
data = pd.read_csv("../data/telecom_churn.csv")
x1 = data2[data2.columns[2:12]] #telecom
y1 = data2['Churn']
'''

x1Train, x1Test, y1Train, y1Test = train_test_split(x1, y1, test_size=0.22,random_state=1)
scaler = StandardScaler()
scaler.fit(x1Train)
x1Train = scaler.transform(x1Train)
# apply same transformation to test data
x1Test = scaler.transform(x1Test)
arch = [12]
act = 'sigmoid'
lr = .0001
#archs = [[12], [4,4,4]]
#for arch in archs:
#rhctune(x1, y1,True,arch)
#satune(x1,y1,True,arch)
gatune(x1,y1,True,arch)
#gdtune(x1,y1,True,arch)


nnrhc = mlrose_hiive.NeuralNetwork(hidden_nodes = arch, activation =act,
                                 algorithm ='random_hill_climb', curve=True,
                                 max_iters = 2000, bias = True, is_classifier = True,
                                 learning_rate = 0.0001, early_stopping = True,
                                 clip_max = 5, max_attempts = 100, random_state = 3)
runnn(nnrhc,x1Train,y1Train,x1Test,y1Test,'rhc')
nngd = mlrose_hiive.NeuralNetwork(hidden_nodes = arch, activation =act,
                                 algorithm = 'gradient_descent', curve=True,
                                 max_iters = 2000, bias = True, is_classifier = True,
                                 learning_rate = 0.0001, early_stopping = True,
                                 clip_max = 5, max_attempts = 100, random_state = 3)
runnn(nngd,x1Train,y1Train,x1Test,y1Test,'gd')


nnsa = mlrose_hiive.NeuralNetwork(hidden_nodes = arch, activation =act,
                                 algorithm = 'simmulated_annealing', curve=True,
                                 max_iters = 2000, bias = True, is_classifier = True,
                                 learning_rate = 0.0001, early_stopping = True,
                                 clip_max = 5, max_attempts = 100, random_state = 3)
runnn(nnsa,x1Train,y1Train,x1Test,y1Test,'gd')

nnga = mlrose_hiive.NeuralNetwork(hidden_nodes = arch, activation =act,
                                 algorithm = 'genetic_alg', curve=True,
                                 max_iters = 2000, bias = True, is_classifier = True,
                                 learning_rate = 0.0001, early_stopping = True,
                                 clip_max = 5, max_attempts = 100, random_state = 3)
runnn(nnsa,x1Train,y1Train,x1Test,y1Test,'gd')
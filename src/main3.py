import mlrose_hiive
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time



def hypertuning(prob,probname,probsize):
    tunerhc(prob, probname,probsize)
    tunesa(prob,probname,probsize)
    tunega(prob,probname,probsize)
    tunemim(prob,probname,probsize)
    '''
    plt.figure()
    plt.plot(rhcmean,label='randomized hill climbing')
    plt.plot(samean,label='simmulated annealing')
    plt.plot(gamean,label='genetic algorithm')
    plt.plot(mimmean,label='mimic algorithm')
    plt.title('probname')
    plt.legend(loc='best')
    plt.savefig(probname + '.PNG')
    plt.close()
    '''

def calc_mean(optlist):
    rhcarr = np.array(optlist)
    rhcmax = np.max([len(a) for a in rhcarr])
    rhcarr = np.asarray([np.pad(a, (0, rhcmax - len(a)), 'edge',) for a in rhcarr])
    rhcmean = np.mean(rhcarr,axis=0)
    return rhcmean

def run_problem(prob,probname,probsize):
    results=[]
    rhclist = []
    salist = []
    galist = []
    mimlist = []
    rhctimelist = []
    satimelist = []
    gatimelist = []
    mimtimelist = []
    iters=1000
    restarts = 5
    mutprob = .3
    mate = .3
    pop = 200
    temp = .1
    expc = .01
    keepp = .25
    noise = .01
    if probname == 'om':
        mutprob = .3
        mate = .3
        pop = 100
        iters = 1000
        keepp = .5
        noise = .005
        temp = .1
        expc = .01
        restarts = 5
    for i in range(4):
        #print(i)
        probi = prob.generate(i,probsize)
        probi.set_mimic_fast_mode(fast_mode=True)
        start = time.time()
        rhcres = mlrose_hiive.random_hill_climb(probi,restarts=restarts,curve=True,max_attempts=iters*1000,max_iters=iters)
        runtime = time.time() - start
        rhclist.append(rhcres[2][:,0])
        rhctimelist.append(runtime)
        #print('RHC %d %d %.2f %d'  % (probsize, rhcres[1],runtime,len(rhcres[2])))
        start = time.time()
        sares = mlrose_hiive.simulated_annealing(probi,mlrose_hiive.ExpDecay(init_temp=temp,exp_const=expc),curve=True, max_attempts=iters*1000,max_iters=iters)
        runtime = time.time() - start
        salist.append(sares[2][:,0])
        satimelist.append(runtime)
        #print('SA %d %d %.2f %d'  % (probsize, sares[1],runtime,len(sares[2])))
        start = time.time()
        gares = mlrose_hiive.genetic_alg(probi,pop_size=20,mutation_prob=mutprob,pop_breed_percent=mate,curve=True, max_attempts=iters*1000,max_iters=1000)
        runtime = time.time() - start
        galist.append(gares[2][:,0])
        gatimelist.append(runtime)
        #print('GA %d %d %.2f %d'  % (probsize, gares[1],runtime,len(gares[2])))
        start = time.time()
        mimres = mlrose_hiive.mimic(probi,pop_size=pop,keep_pct=keepp,noise=.01,curve=True,max_attempts=iters*1000,max_iters=500)
        runtime = time.time() - start
        mimlist.append(mimres[2][:,0])
        mimtimelist.append(runtime)

    rhcmean = calc_mean(rhclist)
    samean = calc_mean(salist)
    gamean = calc_mean(galist)
    mimmean = calc_mean(mimlist)

    plt.figure()
    plt.plot(rhcmean,label='randomized hill climbing')
    plt.plot(samean,label='simmulated annealing')
    plt.plot(gamean,label='genetic algorithm')
    plt.plot(mimmean,label='mimic algorithm')
    plt.title(probname + ' results ' + str(probsize))
    plt.legend(loc='best')
    plt.savefig('./' + probname+'/results/' + probname + str(probsize) + '.PNG')
    plt.close()
    rhctimemean = np.mean(np.array(rhctimelist))
    satimemean = np.mean(np.array(satimelist))
    gatimemean = np.mean(np.array(gatimelist))
    mimtimemean = np.mean(np.array(mimtimelist))
    print('rhc mean %.2f' % rhctimemean)
    print('sa mean %.2f' % satimemean)
    print('ga mean %.2f' % gatimemean)
    print('mim mean %.2f' % mimtimemean)


def tunerhc(prob,probname,probnumber):
    prob = prob.generate(4,probnumber)
    rhclist = []
    iters = 1000

    if probname == 'tsp':
        mutprob = .10
        mate = .50
        pop = 100
        iters = 2500
        keepp = .5
        temp = 100
        expc = .01
        restarts = 6

    for i in range(5):
        rhcres = mlrose_hiive.random_hill_climb(prob,restarts=((i+1)*5),curve=True,max_attempts=iters*1000,max_iters=iters)
        #numpyres = np.append(numpyres,[rhcres[2]],axis=0)
        rhclist.append(rhcres[2][:,0])
    rhcarr = np.array(rhclist)
    print(rhcarr)
    rhcmax = 0
    rhcmax = np.max([len(a) for a in rhcarr])
    rhcarr = np.asarray([np.pad(a, (0, rhcmax - len(a)), 'edge',) for a in rhcarr])
        #mlrose_hiive.simulated_annealing(
    plt.figure()
    for i in range(5):
        plt.plot(rhcarr[i],label=str((i+1)*5))

    #plt.plot(sares[2],label='simmulated annealing')
    #plt.plot(gares[2],label='genetic algorithm')
    #plt.plot(mimres[2],label='mimic algorithm')
    plt.title(probname + ' tuning rhc')
    plt.legend(loc='best')
    plt.savefig('./'+probname+ '/hyp/' + str(probnumber) + 'mrhc.PNG')
    plt.close()

def tunesa(prob,probname,probnumber,average):
    #prob = prob.generate(4,probnumber)
    reslist = []
    if probname == 'om':
        mutprob = .10
        mate = .50
        pop = 100
        iters = 1000
        keepp = .5
        temp = 10
        expc = .01
        restarts = 30
    expc = .01
    iters = 1000
    temps = [.1, 1, 10, 100, 1000]
    avg = 1
    if average == True:
        avg = 4
    for t in temps:
        resrun = []
        for i in range(avg):
            probi = prob.generate(i,probnumber)
            res = mlrose_hiive.simulated_annealing(probi,schedule=mlrose_hiive.ExpDecay(init_temp=t,exp_const=expc),curve=True,max_attempts=iters*1000,max_iters=iters)
        #numpyres = np.append(numpyres,[rhcres[2]],axis=0)
            resrun.append(res[2][:,0])
        reslist.append(calc_mean(resrun))
    resarr = np.array(reslist)
        #mlrose_hiive.simulated_annealing(
    plt.figure()
    for i in range(len(temps)):
        plt.plot(resarr[i],label=str(temps[i]))

    #plt.plot(sares[2],label='simmulated annealing')
    #plt.plot(gares[2],label='genetic algorithm')
    #plt.plot(mimres[2],label='mimic algorithm')
    plt.title(probname + ' tuning sa')
    plt.legend(loc='best')
    plt.savefig('./' + probname + '/hyp/' + str(probnumber) + 'msa.PNG')
    plt.close()

def tunega(prob,probname,probnumber, average):
    #prob = prob.generate(4,probnumber)
    reslist = []
    if probname == 'tsp':
        mutprob = .05
        mate = .50
        pop = 100
        iters = 1000
        keepp = .5
        temp = 100
        expc = .01
        restarts = 30
    mutprobs = [.1, .25, .4, .55, .70, .85]
    mutprob = .30
    mates = [.10, .20, .30, .50, .75]
    iters=1000
    avg = 1
    if average == True:
        avg = 4
    for m in mates:
        resrun = []
        for i in range(avg):
            probi = prob.generate(i,probnumber)
            res = mlrose_hiive.genetic_alg(probi,pop_size=20,mutation_prob=mutprob,pop_breed_percent=m,curve=True,max_attempts=iters*1000,max_iters=iters)
        #numpyres = np.append(numpyres,[rhcres[2]],axis=0)
            resrun.append(res[2][:,0])
        reslist.append(calc_mean(resrun))
    resarr = np.array(reslist)
        #mlrose_hiive.simulated_annealing(
    plt.figure()
    for i in range(len(mates)):
        plt.plot(resarr[i],label=str(mates[i]))

    #plt.plot(sares[2],label='simmulated annealing')
    #plt.plot(gares[2],label='genetic algorithm')
    #plt.plot(mimres[2],label='mimic algorithm')
    plt.title(probname + ' tuning ga')
    plt.legend(loc='best')
    plt.savefig('./' + probname + '/hyp/' + str(probnumber) + 'mga.PNG')
    plt.close()

def tunemim(prob,probname,probnumber, average):
    #seed = np.random.randint(100)
    #prob = prob.generate(4,probnumber)
    #prob.set_mimic_fast_mode(fast_mode=True)
    reslist = []
    if probname == 'om':
        mutprob = .10
        mate = .50
        pop = 100
        iters = 2500
        keepp = .5

        temp = 10
        expc = .01
        restarts = 30
    avg = 1
    if average == True:
        avg = 4
    #noises = [.1, .25, .5, .75, .9]
    #keeps = [.25, .5, .75]
    #keep = .005
    keep = .5
    noises = [.001, .0025, .005, .007, .009]
    iters = 100
    pop = 100
    print("keepp fitness time iters")

    for n in noises:
        resrun = []
        for i in range(avg):
            probi = prob.generate(i,probnumber)
            probi.set_mimic_fast_mode(fast_mode=True)
            start = time.time()
            mimres = mlrose_hiive.mimic(probi,pop_size=pop,noise=n,keep_pct=keep,curve=True,max_attempts=iters*1000,max_iters=500)
            runtime = time.time() - start
            print('MIM %.2f %.2f %.2f %d '  % (n, mimres[1],runtime,len(mimres[2])))
            #numpyres = np.append(numpyres,[rhcres[2]],axis=0)
            resrun.append(mimres[2][:,0])
        reslist.append(calc_mean(np.array(resrun)))
    #print(resarr)
        #resmax = 0
        #resmax = np.max([len(a) for a in resarr])
        #resarr = np.asarray([np.pad(a, (0, resmax - len(a)), 'edge',) for a in resarr])
        #mlrose_hiive.simulated_annealing(s

    resarr = np.array(reslist)
    plt.figure()
    for i in range(len(noises)):
        plt.plot(resarr[i],label=str(noises[i]))
    plt.title(probname + ' tuning mim')
    plt.legend(loc='best')
    plt.savefig('./' + probname + '/hyp/lowpopn' + str(probnumber) + 'keepmim2.PNG')
    plt.close()

def run_size_var(prob, probname, probtest):
    results=[]
    rhclist = []
    salist = []
    galist = []
    mimlist = []
    iters=1000
    restarts = 5
    mutprob = .3
    mate = .3
    pop = 200
    temp = .1
    expc = .01
    keepp = .25
    noise = .01
    if probname == 'tsp':
        mutprob = .15
        mate = .50
        pop = 100
        iters = 2500
        keepp = .6
        temp = 300
        expc = .01
        restarts = 20
    if probname == 'om':
        mutprob = .7
        mate = .50
        pop = 100
        iters = 1000
        keepp = .6
        temp = 10
        expc = .01
        restarts = 20

    print('opt fitness time iters')
    for p in range(4):
        #print(i)
        if probname == 'flipflops':
            probsize = 2**(probtest + i)
        if probname == 'tsp':
            probsize = 25 * (i+2)
        if probname == 'om':
            probsize = 35 * (i+1)
        if probname == 'np':
            probsize = (p+1)*50
        avgscore = []
        avgtime = []
        curves = []
        testaccuracy = []
        trainaccuracy = []
        trainf1 = []
        testf1 = []
        rhtimes = []
        satimes = []
        gatimes = []
        mimtimes= []
        rhavg = []
        saavg = []
        gaavg = []
        mimavg = []
        for i in range(4):
            probi = prob.generate(i,probsize)
            probi.set_mimic_fast_mode(fast_mode=True)
            start = time.time()
            rhcres = mlrose_hiive.random_hill_climb(probi,restarts=restarts,curve=True,max_attempts=iters*1000,max_iters=iters)
            runtime = time.time() - start
            rhavg.append(rhcres[1])
            rhtimes.append(runtime)
            start = time.time()
            sares = mlrose_hiive.simulated_annealing(probi,mlrose_hiive.ExpDecay(init_temp=temp,exp_const=expc),curve=True, max_attempts=iters*1000,max_iters=iters)
            runtime = time.time() - start
            saavg.append(sares[1])
            satimes.append(runtime)
            start = time.time()
            gares = mlrose_hiive.genetic_alg(probi,pop_size=20,mutation_prob=mutprob,pop_breed_percent=mate,curve=True, max_attempts=iters*1000,max_iters=1000)
            runtime = time.time() - start
            gaavg.append(gares[1])
            gatimes.append(runtime)
            start = time.time()
            mimres = mlrose_hiive.mimic(probi,pop_size=pop,keep_pct=keepp,curve=True,max_attempts=iters*1000,max_iters=500)
            runtime = time.time() - start
            mimavg.append(mimres[1])
            mimtimes.append(runtime)


        print('RHC %d %.2f %.2f %d'  % (probsize, np.mean(rhavg),np.mean(rhtimes),iters))
        print('SA %d %.2f %.2f %d'  % (probsize, np.mean(saavg),np.mean(satimes),iters))
        print('GA %d %.2f %.2f %d'  % (probsize, np.mean(gaavg),np.mean(gatimes),iters))
        print('MIM %d %.2f %.2f %d'  % (probsize, np.mean(mimavg),np.mean(mimtimes),iters))

#probsizeff = 100
probsize=10
nop = mlrose_hiive.NoisyPatternGenerator
#hypertuning(flip_flops, 'ff', 1000)
#tunemim(nop, 'np', 100, True)
tunega(nop, 'np', 100, True)
#tunesa(nop, 'np', 100, True)
#run_problem(nop,'np',200)
#run_size_var(nop, 'np', 50)
#base = 5
#run_size_var(flip_flops, '/flips/sizes/' + str(base),base)
#tunemim(flip_flops, './flips/mim/64' , 64)
#tunemim(flip_flops,'flipflops',64)
#tunesa(flip_flops,'flipflops',64)
#plt.figure()
#plt.plot(results4,label='feval')
#plt.title('rhc')
#plt.legend(loc='best')
#plt.savefig('RHCfeval.PNG')
#plt.close()

#nop = mlrose_hiive.NoisyPatternGenerator
#run_size_var(np, 'np', 40)
#run_size_var(om, 'om',0)
#hypertuning(om,'om',70)

#tunemim(nop,'np',20)
#tunemim(nop,'np',30)
#tunemim(nop,'np',30)
#tunega(tsp,'tsp',75)
#hypertuning(tsp,'tsp',75)
'''
run_problem(tsp, 'tsp', 75)

tspnum = 75
tsp = mlrose_hiive.TSPGenerator
run_problem(tsp, 'tsp', 75)

run_size_var(tsp, 'tsp',0)

#hypertuning(tsp,'tsp',75)
#tunemim(tsp,'tsp',75)
tunega(tsp,'tsp',75)
'''
def plot_results(results):
    _, axes = plt.subplots(1, len(results), figsize=(20, 5))
import mlrose_hiive
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time



def hypertuning(prob,probname,probsize):
    #tunerhc(prob, probname,probsize)
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
    if probname == 'sunom':
        mutprob = .85
        mate = .7
        pop = 100
        iters = 1000
        keepp = .7
        temp = .1
        expc = .01
        restarts = 5
    for i in range(2):
        #print(i)
        probi = prob.generate(i,probsize)
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
        gares = mlrose_hiive.genetic_alg(probi,pop_size=pop,mutation_prob=mutprob,pop_breed_percent=mate,curve=True, max_attempts=iters*1000,max_iters=80)
        runtime = time.time() - start
        galist.append(gares[2][:,0])
        gatimelist.append(runtime)
        #print('GA %d %d %.2f %d'  % (probsize, gares[1],runtime,len(gares[2])))
        start = time.time()
        mimres = mlrose_hiive.mimic(probi,pop_size=pop,keep_pct=keepp,curve=True,max_attempts=iters*1000,max_iters=40)
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
        rhcres = mlrose_hiive.random_hill_climb(prob,restarts=(i+1),curve=True,max_attempts=iters*1000,max_iters=iters)
        #numpyres = np.append(numpyres,[rhcres[2]],axis=0)
        rhclist.append(rhcres[2])
    rhcarr = np.negative(np.array(rhclist))
    print(rhcarr)
    rhcmax = 0
    rhcmax = np.max([len(a) for a in rhcarr])
    rhcarr = np.asarray([np.pad(a, (0, rhcmax - len(a)), 'edge',) for a in rhcarr])
        #mlrose_hiive.simulated_annealing(
    plt.figure()
    for i in range(5):
        plt.plot(rhcarr[i],label=str(restarts*(i+2)))

    #plt.plot(sares[2],label='simmulated annealing')
    #plt.plot(gares[2],label='genetic algorithm')
    #plt.plot(mimres[2],label='mimic algorithm')
    plt.title(probname + ' tuning rhc')
    plt.legend(loc='best')
    plt.savefig('./'+probname+ '/hyp/mrhc.PNG')
    plt.close()

def tunesa(prob,probname,probnumber):
    prob = prob.generate(4,probnumber)
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
    temps = [.1, 1, 10, 100, 1000]
    for i in temps:
        res = mlrose_hiive.simulated_annealing(prob,schedule=mlrose_hiive.ExpDecay(init_temp=i,exp_const=expc),curve=True,max_attempts=iters*1000,max_iters=iters)
        #numpyres = np.append(numpyres,[rhcres[2]],axis=0)
        reslist.append(res[2][:,0])
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
    plt.savefig('./' + probname + '/hyp/msa.PNG')
    plt.close()

def tunega(prob,probname,probnumber):
    prob = prob.generate(4,probnumber)
    reslist = []
    if probname == 'tsp':
        mutprob = .55
        mate = .7
        pop = 100
        iters = 1000
        keepp = .5
        temp = 100
        expc = .01
        restarts = 30
    mute = .7
    mutprobs = [.1, .25, .4, .55, .70, .85]
    matepops = [.1, .25, .4, .55, .70, .85]
    mate = .50
    iters=1000
    for i in matepops:
        res = mlrose_hiive.genetic_alg(prob,pop_size=100,mutation_prob=mute,pop_breed_percent=i,curve=True,max_attempts=iters*1000,max_iters=iters)
        #numpyres = np.append(numpyres,[rhcres[2]],axis=0)
        reslist.append(res[2][:,0])
    resarr = np.array(reslist)
        #mlrose_hiive.simulated_annealing(
    plt.figure()
    '''
    for i in range(len(mutprobs)):
        plt.plot(resarr[i],label=str(mutprobs[i]))
    '''
    for i in range(len(mutprobs)):
        plt.plot(resarr[i],label=str(mutprobs[i]))
    #plt.plot(sares[2],label='simmulated annealing')
    #plt.plot(gares[2],label='genetic algorithm')
    #plt.plot(mimres[2],label='mimic algorithm')
    plt.title(probname + ' tuning ga')
    plt.legend(loc='best')
    plt.savefig('./' + probname + '/hyp/matemga.PNG')
    plt.close()

def tunemim(prob,probname,probnumber):
    prob = prob.generate(4,probnumber)
    reslist = []
    if probname == 'om':
        mutprob = .10
        mate = .50
        pop = 100
        iters = 2500
        keepp = .05
        temp = 10
        expc = .01
        restarts = 30
    keeps = [.1, .25, .5, .75, .9]
    iters = 100
    for i in keeps:
        res = mlrose_hiive.mimic(prob,keep_pct=i,curve=True,max_attempts=iters*1000,max_iters=75)
        #numpyres = np.append(numpyres,[rhcres[2]],axis=0)
        reslist.append(res[2][:,0])
    resarr = np.array(reslist)
    print(resarr)
    resmax = 0
    resmax = np.max([len(a) for a in resarr])
    resarr = np.asarray([np.pad(a, (0, resmax - len(a)), 'edge',) for a in resarr])
        #mlrose_hiive.simulated_annealing(s
    plt.figure()
    for i in range(len(keeps)):
        plt.plot(resarr[i],label=str(keeps[i]))
    plt.title(probname + ' tuning mim')
    plt.legend(loc='best')
    plt.savefig('./' + probname + '/hyp/mmim.PNG')
    plt.close()

def run_size_var(prob, probname, probtest):
    results=[]
    rhclist = []
    salist = []
    galist = []
    mimlist = []
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
        mate = .85
        pop = 100
        iters = 1000
        keepp = .6
        temp = 10
        expc = .01
        restarts = 20

    print('opt fitness time iters')
    for i in range(4):
        #print(i)
        if probname == 'flipflops':
            probsize = 2**(probtest + i)
        if probname == 'tsp':
            probsize = 25 * (i+2)
        if probname == 'om':
            probsize = 35 * (i+1)
        probi = prob.generate(i,probsize)
        start = time.time()
        rhcres = mlrose_hiive.random_hill_climb(probi,restarts=restarts,curve=True,max_attempts=iters*1000,max_iters=iters)
        runtime = time.time() - start
        print('RHC %d %d %.2f %d'  % (probsize, rhcres[1],runtime,len(rhcres[2])))
        start = time.time()
        sares = mlrose_hiive.simulated_annealing(probi,mlrose_hiive.ExpDecay(init_temp=temp,exp_const=expc),curve=True, max_attempts=iters*1000,max_iters=iters)
        runtime = time.time() - start
        print('SA %d %d %.2f %d'  % (probsize, sares[1],runtime,len(sares[2])))
        start = time.time()
        gares = mlrose_hiive.genetic_alg(probi,pop_size=pop,mutation_prob=mutprob,pop_breed_percent=mate,curve=True, max_attempts=iters*1000,max_iters=150)
        runtime = time.time() - start
        print('GA %d %d %.2f %d'  % (probsize, gares[1],runtime,len(gares[2])))
        start = time.time()
        mimres = mlrose_hiive.mimic(probi,pop_size=pop,keep_pct=keepp,curve=True,max_attempts=iters*1000,max_iters=100)
        runtime = time.time() - start
        print('MIM %d %d %.2f %d'  % (probsize, mimres[1],runtime,len(mimres[2])))


#probsizeff = 100
#flip_flops = mlrose_hiive.FlipFlopGenerator
#run_problem(flip_flops,'/flip/' str(probsize) + 'flip flop',probsize)
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

om = mlrose_hiive.OneMaxGenerator
run_problem(om, 'sunom', 70)
#run_size_var(om, 'om',0)
#hypertuning(om,'om',70)
#tunemim(tsp,'tsp',75)
#tunega(om,'om',75)
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
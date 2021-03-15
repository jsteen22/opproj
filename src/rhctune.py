def tunerhc(prob,probname,probnumber):
    prob = prob.generate(4,probnumber)
    rhclist = []
    for i in range(5):
        rhcres = mlrose_hiive.random_hill_climb(prob,restarts=2**(i+1),curve=True,fevals=True,max_attempts=probnumber)
        #numpyres = np.append(numpyres,[rhcres[2]],axis=0)
        rhclist.append(rhcres[2])
    rhcarr = np.array(rhclist)
    print(rhcarr)
    rhcmax = 0
    rhcmax = np.max([len(a) for a in rhcarr])
    rhcarr = np.asarray([np.pad(a, (0, rhcmax - len(a)), 'edge',) for a in rhcarr])
        #mlrose_hiive.simulated_annealing(
    plt.figure()
    for i in range(5):
        plt.plot(rhcarr[i],label=str(2**(i+1)))

    #plt.plot(sares[2],label='simmulated annealing')
    #plt.plot(gares[2],label='genetic algorithm')
    #plt.plot(mimres[2],label='mimic algorithm')
    plt.title('probname')
    plt.legend(loc='best')
    plt.savefig('rhc'+probname+'.PNG')
    plt.close()
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.style.use('seaborn-whitegrid')
import numpy as np

import scipy
from scipy.stats import entropy
from scipy.spatial import distance
from scipy import stats
import argparse


# In[2]:



def loadData(path):
    flag = True
    data = np.array([])
    #data = np.array([[0,0]])
    #data = np.append(data, [[0,0]], axis =0)
    with open(path, "r") as filin:
        for line in filin:
            tmp = line[:-1].split("\t")
            if line[0] == "#":
                continue
            elif flag:
                data = np.array([[int(tmp[0]),int(tmp[1])]])
                flag = False
            else:
                data = np.append(data, [[int(tmp[0]),int(tmp[1])]], axis = 0)
    return data

def loadRMSD(path):
    flag = True
    data = []
    #data = np.array([[0,0]])
    #data = np.append(data, [[0,0]], axis =0)
    with open(path, "r") as filin:
        for line in filin:
            data.append(float(line[:-1]))
    return np.array(data)


def RMSDmin(rmsd):
    tmp = rmsd[0]
    minrmsd = [tmp]
    for i in range(1, len(rmsd)):
        if rmsd[i] < tmp:
            tmp = rmsd[i]
        minrmsd.append(tmp)
    return np.array(minrmsd)


def clusterCreate(data):
    sorted_array = data[np.argsort(data[:, 1])]
    clust = []
    nb = []
    tmp=0
    for i in range(len(sorted_array)):
        if i == 0 :
            clust.append(sorted_array[0,0])
            tmp = 1
            nb.append(tmp)
        elif sorted_array[i,0] in clust:
            nb.append(tmp)
        else:
            clust.append(sorted_array[i,0])
            tmp += 1
            nb.append(tmp)
    return nb

def clusterCreate1pourcent(data):
    index_clust = np.unique(data[:,0])
    tot = np.shape(data)[0]*1.0
    cluster = []
    for index in index_clust:
        tmp = np.where(data[:,0]==index)[0]
        if len(tmp)/tot >= 0.01:
            cluster.append(min(data[tmp,1] ))
    cluster.sort()
    tmp = np.zeros(np.shape(data)[0])
    for i in cluster:
        tmp[i:] += 1

    return tmp



def ComputeEntropy(ref, other, rmsd_s, end):
    rmsd_ref = rmsd_s[ref][0:end]
    cpt=0
    rmsd_ref.sort()
    #print(rmsd_ref)
    for i in other:
        #print(rmsd_s[i][0:end])
        rmsd = rmsd_s[i][0:end]
        rmsd.sort()
        val = entropy(rmsd_ref,rmsd)
        #print(val)
        if entropy(rmsd_ref,rmsd) < 0.05:
            cpt += 1
    return cpt



def ComputeJensen(ref, other, rmsd_s, Xmax, end):
    rmsd_ref = rmsd_s[ref][0:end]
    cpt=0
    #Profile de probabilité
    kernel = stats.gaussian_kde(rmsd_ref)
    x = np.arange(0,Xmax,0.01)
    y_ref = []
    for j in x:
        y_ref.append(kernel(j)[0])
    #print(rmsd_ref)
    for i in other:
        rmsd = rmsd_s[i][0:end]
        kernel = stats.gaussian_kde(rmsd)
        x = np.arange(0,Xmax,0.01)
        y = []
        for j in x:
            y.append(kernel(j)[0])
        val = distance.jensenshannon(np.sort(y_ref),np.sort(y))
        #print(val)
        if val < 0.05:
            cpt += 1
    return cpt



def time2converge(path):
    convergence = None
    rmsd = loadRMSD(path)
    flag = True
    kernel = stats.gaussian_kde(rmsd)
    x = np.arange(0,Xmax,0.01)
    y_ref = []
    for j in x:
        y_ref.append(kernel(j)[0])
    for i in range(1,100):
        time = round(len(rmsd)*(i+1)/100)
        tmp_rmsd = rmsd[0:time]
        kernel = stats.gaussian_kde(tmp_rmsd)
        x = np.arange(0,Xmax,0.01)
        y = []
        for j in x:
            y.append(kernel(j)[0])
        tmp = distance.jensenshannon(np.sort(y_ref),np.sort(y))
        #print(tmp)
        #print(time)
        if tmp > 0.05:
            flag = True
        if tmp < 0.05 and flag:
            convergence = time
            flag = False
    return convergence


def JS(rmsd_ref, rmsd=None):
    """
    Make a density profil and compare distribution between with Jenssen Shannon
    divergence between data. If only rmsd_ref, function compute JS between
    all rmsd_ref with itself (with different timestep)
    Arguments:
        rmsd_ref: data used as reference
        rmsd: data compared to reference
    return array with JS values
    """
    if rmsd is None:
        rmsd = rmsd_ref
    rmsd_ref
    flag = True
    kernel = stats.gaussian_kde(rmsd_ref,  bw_method=0.1)
    x = np.arange(0,Xmax,0.01)
    y_ref = []
    tmp = []
    for j in x:
        y_ref.append(kernel(j)[0])
    for i in range(1,100):
        time = round(len(rmsd)*(i+1)/100)
        tmp_rmsd = rmsd[0:time]
        kernel = stats.gaussian_kde(tmp_rmsd,  bw_method=0.1)
        x = np.arange(0,Xmax,0.01)
        y = []
        for j in x:
            y.append(kernel(j)[0])
        tmp.append(distance.jensenshannon(np.sort(y_ref),np.sort(y)))
    return np.array(tmp)



# In[ ]:


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="This script evaluates convergence of simulation")
    parser.add_argument('-peptide', action="store",dest="peptide", type=str, help='peptide\'s sequence.\nExemple: -peptide hPdqsep', default="hPdqsep")
    parser.add_argument('-folder', action="store",dest="folder", type=str, help='folder.\nExemple: -folder mini_map', default="carte")
    parser.add_argument('-nb', action="store",dest="nb", type=str, help='peptide\'s number.\nExemple: --nb 1', default="7.B")
    args = parser.parse_args()
    peptide=args.peptide
    nb=args.nb
    folder = args.folder


    rmsd1 = loadRMSD(folder+"1/rmsd.out")
    rmsd1_min = RMSDmin(rmsd1)
    rmsd2 = loadRMSD(folder+"2/rmsd.out")
    rmsd2_min = RMSDmin(rmsd2)
    rmsd3 = loadRMSD(folder+"3/rmsd.out")
    rmsd3_min = RMSDmin(rmsd3)
    rmsd4 = loadRMSD(folder+"4/rmsd.out")
    rmsd4_min = RMSDmin(rmsd4)
    rmsd5 = loadRMSD(folder+"5/rmsd.out")
    rmsd5_min = RMSDmin(rmsd5)
    Xmax = np.max([np.max(rmsd1), np.max(rmsd2), np.max(rmsd3), np.max(rmsd4), np.max(rmsd5)])

    rmsdAll= np.concatenate([rmsd1,rmsd2, rmsd3,rmsd4, rmsd5])


    # In[17]:


    ##All run###
    #Comparaison des densité de RMSD des run vs l'ensemble des runs
    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel('RMSD $\AA$', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    i = 1
    for i in range(1,6,1):
        rmsd = loadRMSD(folder+str(i)+"/rmsd.out")
        kernel = stats.gaussian_kde(rmsd, bw_method=0.1)
        x = np.arange(0,Xmax,0.01)
        y = []
        for j in x:
            y.append(kernel(j)[0])
        plt.plot(x, y, label='run '+str(i))

    #Courbe de densité du RMSD de l'ensemble des run
    kernel = stats.gaussian_kde(rmsdAll, bw_method=0.1)
    x = np.arange(0,Xmax,0.01)
    yall = []
    for j in x:
        yall.append(kernel(j)[0])
    #kernel = stats.gaussian_kde(rmsdAll)

    plt.plot(x, yall, label='All', color = "black", linewidth = 2)
    plt.title("Peptide "+nb+": "+peptide, fontsize=20)
    plt.legend(fontsize=18)
    #plt.legend(bbox_to_anchor=(1.04,1), loc="lower left", fontsize=14)
    plt.tight_layout()
    #plt.subplots_adjust(right=0.75)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.savefig("Density_all_"+peptide+".png")
    #plt.show()


    # In[6]:


    #plot un run vs courbe de densiré de référence
    kernel = stats.gaussian_kde(rmsdAll, bw_method=0.1)
    x = np.arange(0,Xmax,0.01)
    yall = []
    for j in x:
        yall.append(kernel(j)[0])

    for i in range(1,6,1):
        fig = plt.figure()
        ax = plt.axes()
        plt.xlabel('RMSD $\AA$', fontsize=18)
        plt.ylabel('Density', fontsize=18)
        plt.plot(x, yall, label='All', color = "black", linewidth = 2)
        rmsd = loadRMSD(folder+str(i)+"/rmsd.out")
        kernel = stats.gaussian_kde(rmsd, bw_method=0.1)
        x = np.arange(0,Xmax,0.01)
        y = []
        for j in x:
            y.append(kernel(j)[0])
        plt.plot(x, y, label='run '+str(i))
        plt.title("Peptide "+nb+": "+peptide, fontsize=20)
        plt.legend(fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        fig.savefig("Density_all_run_"+str(i)+".png")
        #plt.show()




    # In[19]:


    # In[7]:


    #Compute
    frames = []
    for indice in range(1,6):
        rmsd = loadRMSD(folder+str(indice)+"/rmsd.out")
        #rmsdAll
        frames.append(JS(rmsdAll, rmsd))


    # In[42]:


    for indice in range(0,5):
        rmsd = loadRMSD(folder+str(indice+1)+"/rmsd.out")
        time = len(rmsd)/100
        x = np.arange(time/100,time-1, time/100)
        len(x)
        tmp = np.where(frames[indice] > 0.05)[0]
        if len(tmp) > 0 and len(x)-1 > tmp[-1]:
            ind = tmp[-1]+1
            print("Replica {} converged at {} ns".format(indice, x[ind]))
        elif len(tmp) > 0 and len(x)-1 == tmp[-1]:
            print("Replica {} did not converge > {} ns".format(indice, x[-1]))
        else:
            print("Replica {} converged at {} ns".format(indice, x[0]))


    # In[5]:



    fig = plt.figure()
    ax = plt.subplot(111)
    #ax = plt.axes()
    plt.xlabel('Temps (ns)', fontsize=18)
    plt.ylabel('Divergence Jensen-Shannon', fontsize=18)

    for indice in range(5):
        rmsd = loadRMSD(folder+str(indice+1)+"/rmsd.out")
        time = len(rmsd)/100
        x = np.arange(time/100,time-1, time/100)
        plt.plot(x, frames[indice], label="run "+str(indice+1), linewidth=2.5)

    plt.hlines(y=0.05, xmin=x[0], xmax=x[-1], label = "Seuil")
    plt.legend(bbox_to_anchor=(1.04,1), fontsize=18)
    plt.title("Peptide "+nb+": "+peptide, fontsize=20)
    fig.savefig("JS_"+peptide+".png")
    #plt.show()



    fig = plt.figure()
    ax = plt.subplot(111)
    #ax = plt.axes()
    plt.xlabel('Temps (ns)', fontsize=18)
    plt.ylabel('Divergence Jensen-Shannon', fontsize=18)

    for indice in range(5):
        rmsd = loadRMSD(folder+str(indice+1)+"/rmsd.out")
        time = len(rmsd)/100
        x = np.arange(time/100,time-1, time/100)
        ax.plot(x, frames[indice], label="run "+str(indice+1), linewidth=2.5)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.hlines(y=0.05, xmin=x[0], xmax=x[-1], label = "Seuil")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=18)
    plt.title("Peptide "+nb+": "+peptide, fontsize=20)
    fig.savefig("JS_"+peptide+".png")
    #plt.show()


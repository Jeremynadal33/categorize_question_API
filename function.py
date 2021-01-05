################################################################################
################################################################################
########### THIS SCRIPT REGROUPS USEFULL HANDMADE FUNCTIONS ####################
################################################################################
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def corr_threshold(df,thresh, root_dir=''):
    upper = df.corr().where(np.triu(np.ones(df.corr().shape), k=1).astype(np.bool)).abs()
    upper = upper[upper>=thresh]
    
    plt.figure(figsize=(10,5))
    hm = sns.heatmap(upper, cmap="coolwarm")
    hm.set_xticklabels(hm.get_xticklabels(), rotation=90,fontsize=10)
    hm.set_yticklabels(hm.get_yticklabels(), rotation=0,fontsize=10) 

    plt.savefig(root_dir+'corr_heat.png',transparent=True)
    plt.show()
    
    upper = upper.stack().sort_values(ascending=False)
    upper = pd.DataFrame(upper).reset_index()
    upper.columns = ['Var 1', 'Var 2', 'correlation']
    
    index =  np.max([ i for i in range(len(upper['correlation'])) if upper['correlation'][i]>=thresh])
    return upper[:index+1]



def missing_data(df,root_dir, name=None): 
    if name :    assert type(name)==str,'Please enter a string name'
    # Overall missing data
    number_missing = df.isnull().sum().sum()
    number_non_missing = df.count().sum()
    
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Données présentes', 'Données manquantes'
    sizes = [number_non_missing, number_missing]
    explode = (0, 0.09)  # only "explode" the 2nd slice (i.e. 'Hogs')
    colors = ['tab:green','tab:red']

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors = colors)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    if name :
        plt.savefig(root_dir + name+'pie_missing_values.png')
    else : 
        plt.savefig(root_dir + 'pie_missing_values.png')
    plt.show()
    
    # Missing data by column
    num_na_cols = df.isnull().sum(axis=0)
    num_na_indv = df.isnull().sum(axis=1)
    
    nb_indv = df.shape[0]

    plt.figure(figsize=(20,5))
    # We create colors from coolwarm to get a gradient of color depending on the rate of missing value
    color = plt.cm.coolwarm(np.linspace(0.,1,11)) # This returns RGBA; convert: 


    colors = []
    for i in range(len(num_na_cols)):
        colors.append(color[int(10*np.round(num_na_cols[i]/nb_indv,1))])

    #plt.title('Freq of NaN for each variable')
    plt.bar(range(len(num_na_cols)),num_na_cols/nb_indv,color=colors)
    #plt.plot([0,nb_var],[1,1],c='red')
    if name:
        plt.savefig(root_dir+name+'bar_missing_value.png')
    else: 
        plt.savefig(root_dir+'bar_missing_value.png')
    plt.show()
    
    for i in range(len(num_na_cols)):
        if num_na_cols[i]!=0:
            print(num_na_cols[i],df.columns[i])
            
            
def display_uniques(array, horiz = False, figsize=(10,10), bins = 10, name = None, save = False,display=True,root_dir=''):
    
    
    if display and not horiz : 
        uniq, count = np.unique(array,return_counts=True)
        for i in range(len(uniq)):
            print(uniq[i],count[i])
        array.hist(figsize=figsize, bins= bins)
        plt.grid(None)
        if name : plt.xlabel(name)
        plt.ylabel('Nombre d individus')
        if save : plt.savefig(root_dir+name+'.png')
        plt.show()
        
        return uniq, count
        
    elif display and horiz :
        counts = array.value_counts()
        plt.figure(figsize=figsize)
        counts.plot(kind='barh')
        if name : plt.xlabel(name)
        plt.ylabel('Nombre d individus')
        if save : plt.savefig(root_dir+name+'.png')
        plt.show()
        return counts


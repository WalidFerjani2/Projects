# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import csv
import numpy as np

import matplotlib.pyplot as plt

import re

def load_dataset(pathname:str):
    """Load a dataset in csv format.

    Each line of the csv file represents a data from our dataset and each
    column represents the parameters.
    The last column corresponds to the label associated with our data.

    Parameters
    ----------
    pathname : str
        The path of the csv file.

    Returns
    -------
    data : ndarray
        All data in the database.
    labels : ndarray
        Labels associated with the data.
    """
    # check the file format through its extension
    if pathname[-4:] != '.csv':
        raise OSError("The dataset must be in csv format")
    # open the file in read mode
    with open(pathname, 'r') as csvfile:
        # create the reader object in order to parse the data file
        reader = csv.reader(csvfile)
        # extract the data and the associated label
        # (he last column of the file corresponds to the label)
        #data = []
        dataset = list(reader)
        #labels = []
        #for row in reader:
        for i in range(len(dataset)):
            #data.append(row[1:])
            #labels.append(row[-1])
            dataset[i] = [ float(x) if re.search('\d',x) else x for x in dataset[i]]
        # converts Python lists into NumPy matrices
        #data = np.array(data, dtype=np.float)
        #labels = np.array(labels)

    # return data with the associated label
    return dataset


def plot_dataset2d(data, labels, theta=None):
    x_aa = data [ labels == 'aa']
    x_ee = data [ labels == 'ee']
    x_eh = data [ labels == 'eh']
    x_eu = data [ labels == 'eu']
    x_ii = data [ labels == 'ii']
    x_oe = data [ labels == 'oe']
    x_oh = data [ labels == 'oh']
    x_oo = data [ labels == 'oo']
    x_uu = data [ labels == 'uu']
    x_yy = data [ labels == 'yy']

    plt.plot(x_aa[:,1],x_aa[:,0],'x',label='aa')
    plt.plot(x_ee[:,1],x_ee[:,0],'k+',label='ee')
    plt.plot(x_eh[:,1],x_eh[:,0],'k+',label='eh')
    plt.plot(x_eu[:,1],x_eu[:,0],'x',label='eu')
    plt.plot(x_ii[:,1],x_ii[:,0],'_',label='ii')
    plt.plot(x_oe[:,1],x_oe[:,0],'|',label='oe')
    plt.plot(x_oh[:,1],x_oh[:,0],'x',label='oh')
    plt.plot(x_oo[:,1],x_oo[:,0],'x',label='oo')
    plt.plot(x_uu[:,1],x_uu[:,0],'x',label='uu')
    plt.plot(x_yy[:,1],x_yy[:,0],'k+',label='yy')
    
    
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()


def plot_loss(h):
    plt.plot(h)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import argparse
import os
import win32api


parser=argparse.ArgumentParser(
    description='''Please read the instructions carefully. 
				   1.The results file should be the same directory as the code.
				   2.Ensure there are no blanks in the spreadsheet. 
				   3.Ensure there is only one sheet in the results document.
				   4.Please never change the format of the spreadsheet. No changes in the columns
				   5.Please change the marks and names only. Don't change the column headers
				   6.This script will create graphs folder if it not present''',
    epilog="""Thank you. please let pathanjali or dheeraj know if you see any issues""")
args=parser.parse_args()

#clean all the folders before the script starts
#if folders are not present, it will create the folders

#---improvise the below code when you find time


#drives = win32api.GetLogicalDriveStrings()
#drives = drives.split('\000')[:-1]
#if len(drives)>1:
#    print ("Below folders will be created in D drive")
#    if not os.path.exists(drives[1]+'gurukulam marks program'):
#        os.makedirs(drives[1]+'gurukulam marks program')
#        print('gurukulam marks program folder created')
#    if not os.path.exists(drives[1]+'gurukulam marks program/graphs'):
#        os.makedirs(drives[1]+'gurukulam marks program/graphs')
#        print('gurukulam marks program/graphs folder created')
#    else:
        

#chart creation tool starts here
maxPieChartSize = 8
maxMarks = 40

#Below code will read data from excel
df = pd.read_csv("D:/gurukulam marks program/results.csv")
df_graphs=df.drop(['total marks'],axis=1)
numGraphs = df_graphs.shape[0]

for x in range(numGraphs):
    ax = plt.axes([0.1,0.1,0.8,0.8], polar=True)
    row = df_graphs.iloc[x]

    

    ######################   IMPORTANT   #############################
    parameters = row["Sruthi":"Service"]
    #parameters=parameters.drop("total marks")
    ######################   IMPORTANT   #############################

    
    theta = []
    radii = []
    width = []
    N = len(parameters)
    for x in range(N):
        theta.append(2*x*np.pi/N)
        radii.append(parameters[x])
        width.append(2*np.pi/N)
    theta.append(0)
    radii.append(25)
    width.append(0)
    bars = plt.bar(theta, radii, width=width, bottom=0.0)

    x = 0
    for r,bar in zip(radii, bars):
        x += 1
        bar.set_facecolor(plt.cm.inferno(x*25))
        bar.set_alpha(0.5)

    plt.xticks(np.arange(0, 2*np.pi, 2*np.pi/N))
    plt.yticks(np.arange(0,25,2))

    plt.setp(ax.get_yticklabels(), fontsize=6)
    
    ax.set_xticklabels(df_graphs.keys()[3:])

    ax.set_ylim(0,25)
    # ax.set_yticklabels(np.arange(0,20,1))
    title="Name:" + str(row["student"]) +"   Year:" + str(row["Year"]) + '   Category: '+ row['category']
    ax.set_title(title)
    
    #plt.suptitle(title,y=1.05)
    plt.savefig("D:/gurukulam marks program/graphs/" + row["student"] +
                row["category"] + ".png", dpi=300, format="png", transparent=True)
    plt.clf()

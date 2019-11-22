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

numGraphs = df.shape[0]

for x in range(numGraphs):
    ax = plt.axes([0.1,0.1,0.8,0.8], polar=True)
    row = df.iloc[x]

    

    ######################   IMPORTANT   #############################
    parameters = row["Sruthi":"Service"]
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
    radii.append(20)
    width.append(0)
    bars = plt.bar(theta, radii, width=width, bottom=0.0)

    x = 0
    for r,bar in zip(radii, bars):
        x += 1
        bar.set_facecolor(plt.cm.inferno(x*20))
        bar.set_alpha(0.5)

    plt.xticks(np.arange(0, 2*np.pi, 2*np.pi/N))
    plt.yticks(np.arange(0,21,2))

    plt.setp(ax.get_yticklabels(), fontsize=6)
    
    ax.set_xticklabels(df.keys()[3:])

    ax.set_ylim(0,20)
    # ax.set_yticklabels(np.arange(0,20,1))
    
    plt.savefig("D:/gurukulam marks program/graphs/" + row["student"] +
                row["category"] + ".png", dpi=300, format="png", transparent=True)
    plt.clf()

##--graph generation code over--#
##-- this code generates images of individual marks--#	
header_one=['Sruthi', 'Laya', 'Melody', 'Diction',
       'Overall performance', 'Total marks']
header_two=['Attendance', 'Concentration',
       'Homework', 'Honest', 'Grit', 'Service', 'Total marks']
#--unique name is combination of name and category(vocal,violin)
def marks_snapshot(marks,uniquename):
    samp_data=marks
    samp_data_one=[samp_data[3:9]]
    samp_data_two=[samp_data[9:16]]
    
    plt.subplots(figsize=(16,6))
    
    the_table = plt.table(cellText=samp_data_one,
                          colLabels=header_one,
                          colWidths=[0.007*len(x) for x in header_one],
                          loc='upper left',cellLoc='center')
    the_table2 = plt.table(cellText=samp_data_two,
                          colLabels=header_two,
                          colWidths=[0.007*len(x) for x in header_two],
                          loc='center left',cellLoc='center')
    plt.axis('off')
    plt.axis('tight')
    plt.tight_layout()
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(24)
    the_table.scale(2, 2)
    the_table2.auto_set_font_size(False)
    the_table2.set_fontsize(24)
    the_table2.scale(2, 2)
    #plt.subplots_adjust(left=1, bottom=1)
    plt.savefig('D:/gurukulam marks program/marks/'+ uniquename +'.png')
    plt.close()
    
#combining the two images and create a pdf/image
def image_combiner(uniquename):    
    list_im = ['D:/gurukulam marks program/graphs/'+uniquename+'.png','D:/gurukulam marks program/marks/'+uniquename+'.png']
    imgs    = [ PIL.Image.open(i) for i in list_im ]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[1][1]
    imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    
    # for a vertical stacking it is simple: use vstack
    imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    imgs_comb.save( 'D:/gurukulam marks program/backpage/'+uniquename+'.png' )
    im1 = Image.open('D:/gurukulam marks program/backpage/'+uniquename+'.png')
    rgb = Image.new('RGB', im1.size, (255, 255, 255))  # white background
    rgb.paste(im1, mask=im1.split()[3])   
    pdf1_filename = 'D:/gurukulam marks program/backpage/'+uniquename+'.pdf'
    
    os.remove('D:/gurukulam marks program/backpage/'+uniquename+'.png')
    rgb.save(pdf1_filename, "PDF" ,resolution=100.0)


#providing input for each record
for i in  range(1,len(df)):
    samp_data=list(df.iloc[i])
    uniquename=str(df.iloc[i][0])+str(df.iloc[i][1])
    marks_snapshot(samp_data,uniquename)
    image_combiner(uniquename)
    print('completed for ',str(df.iloc[i][0]),' ',str(df.iloc[i][1]))
print('exported marks for all students as pictures')

#Grading system now
# above 90- A+
# above 80- A
# above 70- B+
# above 60- B
# above 50- D
# above 40- E
def grader(marks):
    if marks >= 90:
        return 'A+'
    elif marks >= 80:
        return 'A'
    elif marks >=70:
        return 'B+'
    elif marks >=60:
        return 'B'
    elif marks >=50:
        return 'C'
    elif marks >=40:
        return 'D'
    else: return 'E'

df['Grade']=df['total marks'].apply(grader)




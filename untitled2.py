import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import argparse

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

maxPieChartSize = 8
maxMarks = 40

#Below code will read data from excel

df = pd.read_csv("D:/gurukulam marks program/results.csv")
# df = pd.DataFrame(data=d)


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

#providing input for each record
for i in  range(1,len(df)):
    samp_data=list(df.iloc[i])
    uniquename=str(df.iloc[i][0])+str(df.iloc[i][1])
    marks_snapshot(samp_data,uniquename)
    print('completed for ',str(df.iloc[i][0]),' ',str(df.iloc[i][1]))



#combining the two images and create a pdf/image
    
import sys
from PIL import Image

#images = [Image.open(x) for x in ['Test1.jpg', 'Test2.jpg', 'Test3.jpg']]
images=[Image.open(x) for x in ['D:/gurukulam marks program/marks/AanyaVocal.png','D:/gurukulam marks program/graphs/AanyaVocal.png']]
widths, heights = zip(*(i.size for i in images))

total_width = max(widths)
max_height = sum(heights)

new_im = Image.new('RGB', (total_width, max_height))

y_offset = 0
for im in images:
  new_im.paste(im, (0,y_offset))
  y_offset += im.size[0]

new_im.save('D:/gurukulam marks program/test.jpg')
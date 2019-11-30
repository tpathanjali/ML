import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import argparse
import os
import win32api
from PIL import Image, ImageDraw,ImageFont
import img2pdf 
import cv2

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
df = pd.read_csv("D:/gurukulam marks program/dance/Gurukulam_DanceExam results.csv")
df_graphs=df.drop(['total marks', 'Attitude'],axis=1)
df.columns
numGraphs = df_graphs.shape[0]

for x in range(numGraphs):
    ax = plt.axes([0.1,0.1,0.8,0.8], polar=True)
    row = df_graphs.iloc[x]

    

    ######################   IMPORTANT   #############################
    parameters = row["Aangikam":"Grit"]
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
    plt.suptitle("this is test",y=1.05)
    plt.savefig("D:/gurukulam marks program/dance/graphs/" + str(row["student"]) + ".png", dpi=300, format="png")
    plt.clf()
    #, transparent=True

print("graphs creation complete")
##--graph generation code over--#
##-- this code generates images of individual marks--#	
header_one=['Aangikam','Vaachikam','Aahariam','Satvikam','total marks']
header_two=['Attendance', 'Practice',  'Service', 'Grit','Attitude']
#--unique name is combination of name and category(vocal,violin)
def marks_snapshot(marks,uniquename):
    samp_data=marks
    samp_data_one=[samp_data[2:7]]
    samp_data_two=[samp_data[7:13]]
    
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
    plt.savefig('D:/gurukulam marks program/dance/marks/'+ uniquename +'.png')
    plt.close()
    
def name_creator(name, year):
    img=Image.new('RGB',(1152,60),color='white')
    d=ImageDraw.Draw(img)
    fnt=ImageFont.truetype('C:/Windows/Fonts/Arial.ttf',35)
    title="Name: " + str(name) +"   Year: " + str(year )
    #plt.suptitle(title,y=1.05)
    d.text((50,10),title, fill=(0,0,0),font=fnt)
    uniquename=str(name)
    img.save('D:/gurukulam marks program/dance/names/'+uniquename+'.png')
    
#to resize the images 
def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)
   
def image_combiner(uniquename):
    im1 = cv2.imread('D:/gurukulam marks program/dance/names/'+uniquename+'.png')
    im2 = cv2.imread('D:/gurukulam marks program/dance/graphs/'+uniquename+'.png')
    im3 = cv2.imread('D:/gurukulam marks program/dance/marks/'+uniquename+'.png')
    im_v_resize = vconcat_resize_min([im1, im2, im3])
    cv2.imwrite('D:/gurukulam marks program/dance/backpage/'+uniquename+'.png', im_v_resize)
    print('combined image' )
    
def pdf_creator(name):
    # storing image path 
    img_path = 'D:/gurukulam marks program/dance/backpage/'+name+'.png'     
    # storing pdf path 
    pdf_path = 'D:/gurukulam marks program/dance/backpage/'+name+'.pdf'     
    # opening image 
    image = Image.open(img_path)      
    # converting into chunks using img2pdf 
    pdf_bytes = img2pdf.convert(image.filename)      
    # opening or creating pdf file 
    file = open(pdf_path, "wb")      
    # writing pdf files with chunks 
    file.write(pdf_bytes)      
    # closing image file 
    image.close()      
    # closing pdf file 
    file.close()   
    os.remove(img_path)
    # output 
    print("Successfully made " + uniquename + " pdf file") 

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
    
#providing input for each record
for i in  range(1,len(df)):
    samp_data=list(df.iloc[i])
    uniquename=str(df.iloc[i][0])+str(df.iloc[i][1])
    name=str(df.iloc[i][0])
    year=str(df.iloc[i][1])
    marks_snapshot(samp_data,name)
    name_creator(name, year)
    image_combiner(name)
    pdf_creator(name)
    print('completed for ',str(df.iloc[i][0]),' ',str(df.iloc[i][1]))
print('exported marks for all students as pictures')


df['Grade']=df['total marks'].apply(grader)
final=df[['student','Year','Grade','total marks']]
final['date']='Dec 1 2019'
final['category']='Kuchipudi Dance'
final=final[['student','Year','category','Grade','total marks','date']]

final.to_csv('D:/gurukulam marks program/dance/frontend.csv',index=False)



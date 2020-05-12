# -*- coding: utf-8 -*-
"""
Created on Sun June 16 15:30:27 2019

@author: Eshika Mahajan
"""

#IMPORTING NECESSARY LIBRARIES
import pandas as pd
import csv
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowsert
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import statistics as stats
from PIL import Image   #for storing in tiff
from io import BytesIO  #for storing in tiff


#READING CSV FILE "STUDENTDATA" HAVING SAME LOCATION AS OF THIS PROGRAM
'''KEEP CSV AND THE PYTHON FILE IN THE SAME DIRECTORY'''
df=pd.read_csv("finalstudent.csv")
df_future=pd.read_csv("Book1.csv")

#removing extra columns
df.drop(['employment offer', 'mode of placement',
       'Please provide an estimate of the yearly salary that you have been offered.',
       'Is the employment opportunity off-campus?','If you are not planning for employment immediately, for which examitiore you preparing? (At most you can select three preferences)',
       'Which of the following training programs have been conducted by the institute?'],1,inplace = True)



#CONVERSION FUNCTION CONVERTS CATEGORICAL VARIABLES INTO NUMERIC CODES

def conversion(original,new_name):
  ler=LabelEncoder()
  df[new_name]=ler.fit_transform(df[original])
  return(df[new_name])


#CALLING OUT CONVERSION FUNCTION
conversion("Gender","Gender_converted")
conversion("Sections","Sections_converted")
conversion("GATE qualified","qualifiers_converted")


#BACKING UP DATA
backup=df.copy()



#CREATING DATAFRAMES OF INDIVUAL SECTIONS
def sections(name,cole_name,sect):
  name=df[(df[cole_name]==sect)]
  return(name)

#CREATING THE DUMMY VARIABLES
def get_dummy(original,coln_name):
  df=pd.get_dummies(original[coln_name])
  return(df)

'''
I did not use one hot encoder because it replaces all the column names with 0,1,2..
which would be difficult to rename manually.
In such scenario dummy variables are much preferred than one hot encoder

'''

#CONCATINATING DUMMY VARIABLES TO MAIN DATAFRAME
def to_concat_df(final,initial):
  df = pd.concat([final, initial], axis=1)
  return(df)

#FINDING MEAN AND STANDARD DEVIATION
def avg(df,col,texte):
  #print('\n')
  print(texte)
  print('mean :',np.nanmean(df[col]))
  print('standard deviation:',np.nanstd(df[col]))
  print('\n')
  

#CREATING DUMMY VARIABLES AND CONCATINATING THEM TO THE ORIGINAL DATASET
df=to_concat_df(df,get_dummy(df,"Sections_converted"))


#RENAMING THE DUMMY COLUMNS AND REQUIRED COLUMNS
df.rename(columns={0:"GEN",1:"OBC",2:"PH",3:"SC",4:"ST"},inplace=True)
df.rename(columns={'GATE Marks out of 100':'GATE Marks'},inplace=True)


#SAME PROCEDURE FOR GENDERS TOO
df=to_concat_df(df,get_dummy(df,"Gender_converted"))
df.rename(columns={1:"MALE",0:"FEMALE"},inplace=True)
df.rename(columns={'CGPA':"B.Tech CGPA"},inplace=True)


#SAME PROCEDURE FOR GATE QUALIFIERS TOO
df=to_concat_df(df,get_dummy(df,"qualifiers_converted"))
df.rename(columns={0:"Not qualified",1:"Not Appeared",2:"Qualified"},inplace=True)

#removing unwanted columns
df.drop(["Sections",'GATE qualified','Gender_converted', 'Sections_converted', 'qualifiers_converted'],1,inplace = True)


#GETTING GEN_MALE DATACOLUMN
df.loc[((df.MALE == 1) & (df.GEN == 1)), 'GEN_MALE'] = 1
df.loc[((df.MALE != 1) | (df.GEN != 1)), 'GEN_MALE'] = 0

#GETTING GEN_FEMALE DATACOLUMN
df.loc[((df.FEMALE == 1) & (df.GEN == 1)), 'GEN_FEMALE'] = 1
df.loc[((df.FEMALE != 1) | (df.GEN != 1)), 'GEN_FEMALE'] = 0

#-----------------------------------------------------------------


#GETTING OBC_MALE DATACOLUMN
df.loc[((df.MALE == 1) & (df.OBC == 1)), 'OBC_MALE'] = 1
df.loc[((df.MALE != 1) | (df.OBC != 1)), 'OBC_MALE'] = 0


#GETTING OBC_FEMALE DATACOLUMN
df.loc[((df.FEMALE == 1) & (df.OBC == 1)), 'OBC_FEMALE'] = 1
df.loc[((df.FEMALE != 1) | (df.OBC != 1)), 'OBC_FEMALE'] = 0

#-----------------------------------------------------------------



#GETTING SC_MALE DATACOLUMN
df.loc[((df.MALE == 1) & (df.SC == 1)), 'SC_MALE'] = 1
df.loc[((df.MALE != 1) | (df.SC != 1)), 'SC_MALE'] = 0


#GETTING SC_FEMALE DATACOLUMN
df.loc[((df.FEMALE == 1) & (df.SC == 1)), 'SC_FEMALE'] = 1
df.loc[((df.FEMALE != 1) | (df.SC != 1)), 'SC_FEMALE'] = 0

#-----------------------------------------------------------------


#GETTING ST_MALE DATACOLUMN
df.loc[((df.MALE == 1) & (df.ST == 1)), 'ST_MALE'] = 1
df.loc[((df.MALE != 1) | (df.ST != 1)), 'ST_MALE'] = 0

#GETTING ST_FEMALE DATACOLUMN
df.loc[((df.FEMALE == 1) & (df.ST == 1)), 'ST_FEMALE'] = 1
df.loc[((df.FEMALE != 1) | (df.ST != 1)), 'ST_FEMALE'] = 0

#----------------------------------------------------------------

#GETTING male qualified DATACOLUMN
df.loc[((df.MALE == 1) & (df.Qualified == 1)), 'MALE_QUALIFIED'] = 1
df.loc[((df.MALE != 1) | (df.Qualified != 1)), 'MALE_QUALIFIED'] = 0

#GETTING female qualified DATACOLUMN
df.loc[((df.FEMALE == 1) & (df.Qualified == 1)), 'FEMALE_QUALIFIED'] = 1
df.loc[((df.FEMALE != 1) | (df.Qualified != 1)), 'FEMALE_QUALIFIED'] = 0

#-----------------------------------------------------------------


#GETTING GEN_MALE qualified DATACOLUMN
df.loc[((df.GEN_MALE == 1) & (df.Qualified == 1)), 'GEN_MALE_QUALIFIED'] = 1
df.loc[((df.GEN_MALE != 1) | (df.Qualified != 1)), 'GEN_MALE_QUALIFIED'] = 0

#GETTING GEN_FEMALE qualified DATACOLUMN
df.loc[((df.GEN_FEMALE == 1) & (df.Qualified == 1)), 'GEN_FEMALE_QUALIFIED'] = 1
df.loc[((df.GEN_FEMALE != 1) | (df.Qualified != 1)), 'GEN_FEMALE_QUALIFIED'] = 0

#-----------------------------------------------------------------


#GETTING OBC_MALE QUALIFIEED DATACOLUMN
df.loc[((df.OBC_MALE == 1) & (df.Qualified == 1)), 'OBC_MALE_QUALIFIED'] = 1
df.loc[((df.OBC_MALE != 1) | (df.Qualified != 1)), 'OBC_MALE_QUALIFIED'] = 0

#GETTING OBC_FEMALE QUALIFIED DATACOLUMN
df.loc[((df.OBC_FEMALE == 1) & (df.Qualified == 1)), 'OBC_FEMALE_QUALIFIED'] = 1
df.loc[((df.OBC_FEMALE != 1) | (df.Qualified != 1)), 'OBC_FEMALE_QUALIFIED'] = 0

#-----------------------------------------------------------------


#GETTING SC_MALE QUALIFIED DATACOLUMN
df.loc[((df.SC_MALE == 1) & (df.Qualified == 1)), 'SC_MALE_QUALIFIED'] = 1
df.loc[((df.SC_MALE != 1) | (df.Qualified != 1)), 'SC_MALE_QUALIFIED'] = 0

#GETTING SC_FEMALE QUALIFIED DATACOLUMN
df.loc[((df.SC_FEMALE == 1) & (df.Qualified == 1)), 'SC_FEMALE_QUALIFIED'] = 1
df.loc[((df.SC_FEMALE != 1) | (df.Qualified != 1)), 'SC_FEMALE_QUALIFIED'] = 0

#-----------------------------------------------------------------


#GETTING ST_MALE QUALIFIED DATACOLUMN
df.loc[((df.ST_MALE == 1) & (df.Qualified == 1)), 'ST_MALE_QUALIFIED'] = 1
df.loc[((df.ST_MALE != 1) | (df.Qualified != 1)), 'ST_MALE_QUALIFIED'] = 0

#GETTING ST_FEMALE QUALIFIED DATACOLUMN
df.loc[((df.ST_FEMALE == 1) & (df.Qualified == 1)), 'ST_FEMALE_QUALIFIED'] = 1
df.loc[((df.ST_FEMALE != 1) | (df.Qualified != 1)), 'ST_FEMALE_QUALIFIED'] = 0

#-------------------------------------------------------------------------------------------------------------------------


#GETTING CFTI DATACOLUMN
df['CFTI'] = [1 if Institute in(['IIIT Guwahati','NIT Uttarakhand',
                                 'NIT Sikkim','NIT Agartala',
                                 'NIT Arunachal Pradesh','NIT Srinagar','NIT Meghalaya','NIT Manipur',
                                 'NIT Mizoram','IIIT Manipur','NIT Nagaland']) else 0 for Institute in df['Institute']]



df['NON-CFTI'] = [0 if Institute in(['IIIT Guwahati','NIT Uttarakhand',
                                 'NIT Sikkim','NIT Agartala',
                                 'NIT Arunachal Pradesh','NIT Srinagar','NIT Meghalaya','NIT Manipur',
                                 'NIT Mizoram','IIIT Manipur','NIT Nagaland']) else 1 for Institute in df['Institute']]
#-------------------------------------------------------------------------------------------------------------------------------


#GETTING CFTI_MALE DATACOLUMN
df.loc[((df.MALE == 1) & (df.CFTI == 1)), 'CFTI_MALE'] = 1
df.loc[((df.MALE != 1) | (df.CFTI != 1)), 'CFTI_MALE'] = 0

#GETTING CFTI_FEMALE DATACOLUMN
df.loc[((df.FEMALE == 1) & (df.CFTI == 1)), 'CFTI_FEMALE'] = 1
df.loc[((df.FEMALE != 1) | (df.CFTI != 1)), 'CFTI_FEMALE'] = 0

#-------------------------------------------------------------------------------------


#GETTING NONCFTI_MALE DATACOLUMN
df.loc[((df.MALE == 1) & (df.CFTI == 0)), 'NONCFTI_MALE'] = 1
df.loc[((df.MALE == 0) & (df.FEMALE==1)), 'NONCFTI_MALE'] = 0

#GETTING NONCFTI_FEMALE DATACOLUMN
df.loc[((df.FEMALE == 1) & (df.CFTI == 0)), 'NONCFTI_FEMALE'] = 1
df.loc[((df.FEMALE != 1) & (df.MALE != 0)), 'NONCFTI_FEMALE'] = 0



#---------------------------------------------------------------------------
#HERE CFTI NONQUALIFIED + NON CFTI NON QUALIFIED BOTH ARE GIVEN 1 BUT IF WE WANT ACCURATE RESULTS WE SHOULD ONLY CONSIDER DF_CFTI THEN SEE WHAT HAPPENS!
#GETTING CFTI qualified DATACOLUMN
df.loc[((df.CFTI == 1) & (df.Qualified == 1)), 'CFTI_QUALIFIED'] = 1
df.loc[((df.CFTI != 1) | (df.Qualified != 1)), 'CFTI_QUALIFIED'] = 0

#df.loc[((df.NONCFTI==1) & (df.Qualified == 1)), 'NONCFTI_QUALIFIED'] = 1
#df.loc[((df.NONCFTI != 1) | (df.CFTI!=0)), 'NONCFTI_QUALIFIED'] = 0




#GETTING CFTI_MALE qualified  DATACOLUMN
df.loc[((df.CFTI_MALE == 1) & (df.Qualified == 1)), 'CFTI_MALE_QUALIFIED'] = 1
df.loc[((df.CFTI_MALE != 1) | (df.Qualified != 1)), 'CFTI_MALE_QUALIFIED'] = 0

#GETTING CFTI_FEMALE qualified DATACOLUMN
df.loc[((df.CFTI_FEMALE == 1) & (df.Qualified == 1)), 'CFTI_FEMALE_QUALIFIED'] = 1
df.loc[((df.CFTI_FEMALE != 1) | (df.Qualified != 1)), 'CFTI_FEMALE_QUALIFIED'] = 0


df.to_csv('file1.csv') 


'''

#GETTING NONCFTI_MALE qualified  DATACOLUMN
df.loc[((df.NONCFTI_MALE == 1) & (df.Qualified == 1)), 'NONCFTI_MALE_QUALIFIED'] = 1
df.loc[((df.NONCFTI_MALE != 1) | (df.Qualified != 1)), 'NONCFTI_MALE_QUALIFIED'] = 0

#GETTING NONCFTI_FEMALE qualified DATACOLUMN
df.loc[((df.CFTI_FEMALE == 1) & (df.Qualified == 1)), 'NONCFTI_FEMALE_QUALIFIED'] = 1
df.loc[((df.CFTI_FEMALE != 1) | (df.Qualified != 1)), 'NONCFTI_FEMALE_QUALIFIED'] = 0


'''


#GETTING NENCESSARY DATAFRAMES FROM SELECTION CRITERIA FOR PLOTTING AND SEPERATE RECORD
#CAN ALSO BE DONE USING SECTIONS FUNCTION

df_cfti=df[(df['CFTI']==1)]
df_cfti_male=df[(df['CFTI_MALE']==1)]
df_cfti_female=df[(df['CFTI_FEMALE']==1)]

df_noncfti=df[(df['CFTI']==0)]
df_noncfti_male=df_noncfti[(df_noncfti['MALE']==1)]
df_noncfti_female=df_noncfti[(df_noncfti['FEMALE']==1)]


df_gen=df[(df['GEN']==1)]
df_gen_male=df[(df['GEN_MALE']==1)]
df_gen_female=df[(df['GEN_FEMALE']==1)]



df_obc=df[(df['OBC']==1)]
df_obc_male=df[(df['OBC_MALE']==1)]
df_obc_female=df[(df['OBC_FEMALE']==1)]


df_sc=df[(df['SC']==1)]
df_sc_male=df[(df['SC_MALE']==1)]
df_sc_female=df[(df['SC_FEMALE']==1)]


df_st=df[(df['ST']==1)]
df_st_male=df[(df['ST_MALE']==1)]
df_st_female=df[(df['ST_FEMALE']==1)]


df_qualified=df[(df['Qualified']==1)]
df_qualified_male=df[(df['MALE_QUALIFIED']==1)]
df_qualified_female=df[(df['FEMALE_QUALIFIED']==1)]

df_notappeared=df[(df['Not Appeared']==1)]

df_unqualified=df[(df['Not qualified']==1)]
df_unqualified_male=df_unqualified[df['MALE']==1]
df_unqualified_female=df_unqualified[df['FEMALE']==1]


df_qualified_cfti=df[(df['CFTI_QUALIFIED']==1)]
df_qualified_cfti_male=df[(df['CFTI_MALE_QUALIFIED']==1)]
df_qualified_cfti_female=df[(df['CFTI_FEMALE_QUALIFIED']==1)]

df_qualified_noncfti=df[(df['CFTI']==0)& (df['Qualified']==1)]
df_qualified_noncfti_male=df_noncfti_male[(df_noncfti_male['MALE_QUALIFIED']==1)]
df_qualified_noncfti_female=df_noncfti_female[(df_noncfti_female['FEMALE_QUALIFIED']==1)]


#MULTIPLE ENTRIES OF FUTURE ASPIRATIONS WERE SEPERATED IN THE EXCEL ITSELF THEN
#ALL THE 3 COLUMNS WERE CLUBBED IN 1 TO MAKE ONE COLUMN SO THAT THE FREQUENCY CAN BE CALCULATED

def piechart_combo(df_name):
  list1=df_name['future examination 1'].tolist()
  list2=df_name['future examination 2'].tolist()
  list3=df_name['future examination 3'].tolist()
  finalist=list1+list2+list3
  #print(finalist)
  return(finalist)

#CLUBBING FUTURE ASPIRATIONS OF ALL THE STUDENTS
Data1 = piechart_combo(df)
Data1=[i for i in Data1 if str(i)!='nan']
df_data1=pd.DataFrame(Data1,
    columns=['Combined'])

#CLUBBING FUTURE ASPIRATIONS OF UNQUALIFIED STUDENTS
Data2 = piechart_combo(df_unqualified)
Data2=[i for i in Data2 if str(i)!='nan']
df_data2=pd.DataFrame(Data2,columns=['not qualified'])

#CLUBBING FUTURE ASPIRATIONS OF NOT APPEARED STUDENTS
Data3 = piechart_combo(df_notappeared)
Data3=[i for i in Data3 if str(i)!='nan']
df_data3=pd.DataFrame(Data3,columns=['not appeared'])

#CLUBBING FUTURE ASPIRATIONS OF QUALIFIED STUDENTS
Data4 = piechart_combo(df_qualified)
Data4=[i for i in Data4 if str(i)!='nan']
df_data4=pd.DataFrame(Data4,columns=['qualified'])


#CONCATINATING INDIVIDUAL DATAFRAMES OBTAINED INTO A SINGLE DATASET 'data'
data = pd.concat([df_data1,df_data2,df_data3,df_data4], axis=1)

data.to_csv("newcsv.csv")
not_qual=data['not qualified'].value_counts()
not_appeared=data['not appeared'].value_counts()
qual=data['qualified'].value_counts()
comined=data['Combined'].value_counts()


#GETTING CORRELATION WITHIN DIFFERENT COLUMNS
corr_choice=input('type yes to print the whole correlation')
if corr_choice=='yes':
  print(df.corr().to_string())
  print('\n')

#--------------------------------------------------------------------------------------------------------------- 
print(df[df.columns[1:]].corr()[['B.Tech CGPA','Class XII CGPA','Class X CGPA','GATE Marks','Qualified']])
print(df[df.columns[1:]].corr()[['Class X CGPA','GATE Score']])


df_new=df[['B.Tech CGPA','Class XII CGPA','Class X CGPA'
           ,'GATE Score', 'GATE Marks']]
print('\n new dataframe correlation')

print(df_new.corr())

df_sep=df[['B.Tech CGPA','Class XII CGPA','Class X CGPA','Qualified','CFTI','NON-CFTI']]
df_sep2=df[['B.Tech CGPA','Class XII CGPA','Class X CGPA','Qualified','GATE Score','GATE Marks']]
#-----------------------------------------------------------------------------------------------------------------

#CODE FOR HEATMAP

y=df_sep2.corr()
mask=np.zeros_like(y)
mask=mask+1
mask=np.triu(mask)

plt.figure(figsize=[4,4],dpi=300)
fig=plt.figure(1)
ax=fig.add_subplot(111)
plt.title('CORRELATION HEAT-MAP', fontsize=10)
ax.tick_params(axis='both',which='major',labelsize=4.5)
ax.tick_params(axis='both',which='minor',labelsize=4.5)
sns.heatmap(y,mask=mask,annot=True,annot_kws={'size':7})
plt.xticks(rotation=35)
plt.yticks(rotation=0)
plt.savefig('correlation_map.png')
plt.show()  

'''HEAT MAP HAS TO BE RUN ON SHELL FOR BETTER ALIGNMENT ALONG WITH AXIS ROTATION COMMAND'''


#HEATMAP FOR CFTI DATAFRAME
df_sep_cfti=df_cfti[['B.Tech CGPA','Class XII CGPA','Class X CGPA','Qualified','GATE Score','GATE Marks']]
df_cfti_qualified=df_cfti[(df['Qualified']==1)]
df_cfti_unqualified=df_cfti[(df['Qualified']==0)]

y1=df_sep_cfti.corr()
mask=np.zeros_like(y1)
mask=mask+1
mask=np.triu(mask)

plt.figure(figsize=[4,4],dpi=300)
fig=plt.figure(1)
ax=fig.add_subplot(111)
plt.title('CORRELATION HEAT-MAP CFTI', fontsize=10)
ax.tick_params(axis='both',which='major',labelsize=4.5)
ax.tick_params(axis='both',which='minor',labelsize=4.5)
sns.heatmap(y1,mask=mask,annot=True,annot_kws={'size':5})
plt.xticks(rotation=35)
plt.yticks(rotation=0)
plt.savefig('correlation_map_CFTI.png')
plt.show()  




#HEATMAP FOR NONCFTI DATAFRAME
df_sep_noncfti=df_noncfti[['B.Tech CGPA','Class XII CGPA','Class X CGPA','Qualified','GATE Score','GATE Marks']]
df_noncfti_qualified=df_noncfti[(df['Qualified']==1)]
df_noncfti_unqualified=df_noncfti[(df['Qualified']==0)]


y2=df_sep_noncfti.corr()
mask=np.zeros_like(y2)
mask=mask+1
mask=np.triu(mask)

plt.figure(figsize=[4,4],dpi=300)
fig=plt.figure(1)
ax=fig.add_subplot(111)
plt.title('CORRELATION HEAT-MAP NONCFTI', fontsize=10)
ax.tick_params(axis='both',which='major',labelsize=4.5)
ax.tick_params(axis='both',which='minor',labelsize=4.5)
sns.heatmap(y2,mask=mask,annot=True,annot_kws={'size':5})
plt.xticks(rotation=35)
plt.yticks(rotation=0)
plt.savefig('correlation_map_NONCFTI.png')
plt.show()  



#PLOTTING HISTOGRAMS NO COLOR + COLORED EDGES WITH INCREASED LINEWIDTH

plt.hist(df_unqualified['B.Tech CGPA'], edgecolor='red',linewidth=3.6,label='NON-QUALIFIED \nMEAN:7.37 STD:0.79',fc=(0, 0, 0, 0),density=True)
plt.hist(df_qualified['B.Tech CGPA'], edgecolor='blue',linewidth=3.6,label='QUALIFIED \nMEAN:7.82 STD:0.72',fc=(0, 0, 0, 0),density=True)
plt.title('Normalised B.Tech CGPA ', fontsize=15)
plt.xlabel('CGPA', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(prop={'size':11.25})
plt.grid()
# save figure
# (1) save the image in memory in PNG format
png1 = BytesIO()
fig.savefig(png1,dpi=300, format='png')
# (2) load this image into PIL
png2 = Image.open(png1)
# (3) save as TIFF
png2.save('BtechCgpa_qual.tiff')
png1.close()
plt.show()



plt.hist(df_unqualified['Class X CGPA'], edgecolor='red',linewidth=3.6,label='NON-QUALIFIED \nMEAN:8.18 STD:1.22',fc=(0, 0, 0, 0),density=True)
plt.hist(df_qualified['Class X CGPA'], edgecolor='blue',linewidth=3.6,label='QUALIFIED\nMEAN:8.71 STD:0.95',fc=(0, 0, 0, 0),density=True)
plt.title('Normalised Class X CGPA ', fontsize=14)
plt.xlabel('CGPA', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(prop={'size':11.25})
plt.grid()
png4 = BytesIO()
fig.savefig(png4,dpi=300, format='png')
png3 = Image.open(png4)
png3.save('class10_qual.tiff')
png4.close()
plt.show()


plt.hist(df_unqualified['Class XII CGPA'], edgecolor='red',linewidth=3.6,label='NON-QUALIFIED\nMEAN:7.64 STD:1.06',fc=(0, 0, 0, 0),density=True)
plt.hist(df_qualified['Class XII CGPA'], edgecolor='blue',linewidth=3.6,label='QUALIFIED\nMEAN:8.26 STD:0.83',fc=(0, 0, 0, 0),density=True)
plt.title('Normalised Class XII CGPA', fontsize=14)
plt.xlabel('CGPA', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(prop={'size':11.25})
plt.grid()
png5 = BytesIO()
fig.savefig(png5,dpi=300, format='png')
png6 = Image.open(png5)
png6.save('class12_qual.tiff')
png5.close()
plt.show()

#GRAPH FOR CFTI STUDENTS------------------------------------------------------------------------------------

plt.hist(df_cfti_unqualified['B.Tech CGPA'], edgecolor='red',linewidth=3.6,label='NON-QUALIFIED CFTI \nMEAN:7.86 STD:0.74',fc=(0, 0, 0, 0),density=True)
plt.hist(df_cfti_qualified['B.Tech CGPA'], edgecolor='blue',linewidth=3.6,label='QUALIFIED CFTI\nMEAN:8.03 STD:0.67',fc=(0, 0, 0, 0),density=True)
plt.title('Normalised B.Tech CGPA ', fontsize=15)
plt.xlabel('CGPA', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(prop={'size':11.25})
plt.grid()
plt.savefig('btech_cfti_qual.png')
# save figure
# (1) save the image in memory in PNG format
'''
png1 = BytesIO()
fig.savefig(png1,dpi=300, format='png')
# (2) load this image into PIL
png2 = Image.open(png1)
# (3) save as TIFF
png2.save('3dPlot.tiff')
png1.close()'''
plt.show()



plt.hist(df_cfti_unqualified['Class X CGPA'], edgecolor='red',linewidth=3.6,label='NON-QUALIFIED CFTI \nMEAN:8.60 STD:1.14',fc=(0, 0, 0, 0),density=True)
plt.hist(df_cfti_qualified['Class X CGPA'], edgecolor='blue',linewidth=3.6,label='QUALIFIED CFTI\nMEAN:8.79 STD:1.08',fc=(0, 0, 0, 0),density=True)
plt.title('Normalised Class X CGPA ', fontsize=14)
plt.xlabel('CGPA', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(prop={'size':11.25})
plt.grid()
'''
png4 = BytesIO()
fig.savefig(png4,dpi=300, format='png')
png3 = Image.open(png4)
png3.save('3dPlot2.tiff')
png4.close()'''
plt.show()


plt.hist(df_cfti_unqualified['Class XII CGPA'], edgecolor='red',linewidth=3.6,label='NON-QUALIFIED CFTI\nMEAN:8.44 STD:1.02',fc=(0, 0, 0, 0),density=True)
plt.hist(df_cfti_qualified['Class XII CGPA'], edgecolor='blue',linewidth=3.6,label='QUALIFIED CFTI\nMEAN:8.60 STD:1.04',fc=(0, 0, 0, 0),density=True)
plt.title('Normalised Class XII CGPA', fontsize=14)
plt.xlabel('CGPA', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(prop={'size':11.25})
plt.grid()
'''
png5 = BytesIO()
fig.savefig(png5,dpi=300, format='png')
png6 = Image.open(png5)
png6.save('3dPlot3.tiff')
png5.close();
'''
plt.show()


#GRAPH FOR NON CFTI STUDENTS------------------------------------------------------------------------------------

plt.hist(df_noncfti_unqualified['B.Tech CGPA'], edgecolor='red',linewidth=3.6,label='NON-QUALIFIED NON CFTI\nMEAN:7.34 STD:0.78',fc=(0, 0, 0, 0),density=True)
plt.hist(df_noncfti_qualified['B.Tech CGPA'], edgecolor='blue',linewidth=3.6,label='QUALIFIED NON CFTI\nMEAN:7.80 STD:0.72',fc=(0, 0, 0, 0),density=True)
plt.title('Normalised B.Tech CGPA ', fontsize=15)
plt.xlabel('CGPA', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(prop={'size':11.25})
plt.grid()
# save figure
# (1) save the image in memory in PNG format
'''
png1 = BytesIO()
fig.savefig(png1,dpi=300, format='png')
# (2) load this image into PIL
png2 = Image.open(png1)
# (3) save as TIFF
png2.save('3dPlot.tiff')
png1.close()'''
plt.show()



plt.hist(df_noncfti_unqualified['Class X CGPA'], edgecolor='red',linewidth=3.6,label='NON-QUALIFIED \nMEAN:8.13 STD:1.21',fc=(0, 0, 0, 0),density=True)
plt.hist(df_noncfti_qualified['Class X CGPA'], edgecolor='blue',linewidth=3.6,label='QUALIFIED\nMEAN:8.70 STD:0.94',fc=(0, 0, 0, 0),density=True)
plt.title('Normalised Class X CGPA ', fontsize=14)
plt.xlabel('CGPA', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(prop={'size':11.25})
plt.grid()
'''
png4 = BytesIO()
fig.savefig(png4,dpi=300, format='png')
png3 = Image.open(png4)
png3.save('3dPlot2.tiff')
png4.close()'''
plt.show()


plt.hist(df_noncfti_unqualified['Class XII CGPA'], edgecolor='red',linewidth=3.6,label='NON-QUALIFIED\nMEAN:7.59 STD:1.04',fc=(0, 0, 0, 0),density=True)
plt.hist(df_noncfti_qualified['Class XII CGPA'], edgecolor='blue',linewidth=3.6,label='QUALIFIED\nMEAN:8.23 STD:0.80',fc=(0, 0, 0, 0),density=True)
plt.title('Normalised Class XII CGPA', fontsize=14)
plt.xlabel('CGPA', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(prop={'size':11.25})
plt.grid()
'''
png5 = BytesIO()
fig.savefig(png5,dpi=300, format='png')
png6 = Image.open(png5)
png6.save('3dPlot3.tiff')
png5.close();
'''
plt.show()


#plt.savefig('hello1.tiff', dpi=300, format='TIFF', bbox_inches='tight')

x=input('For going to python documentation for correlation press y')

if x=='y':
  webbrowser.open("https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html")
  


plt.hist(df_gen['Class XII CGPA'], edgecolor='red',linewidth=3.6,label='Class XII CGPA\nMEAN:8.07 STD:1.03',fc=(0, 0, 0, 0),density=True)
plt.hist(df_gen['Class X CGPA'], edgecolor='blue',linewidth=3.6,label='Class X CGPAGEN\nMEAN:8.57 STD:1.13',fc=(0, 0, 0, 0),density=True)
plt.hist(df_gen['B.Tech CGPA'], edgecolor='black',linewidth=3.6,label='B.Tech CGPA \nMEAN:7.66 STD:0.77',fc=(0, 0, 0, 0),density=True)
plt.title('Gen Student Pattern', fontsize=14)
plt.xlabel('CGPA', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(prop={'size':11.25})
plt.grid()
plt.show()



#HISTOGRAMS

plt.hist(df_gen['Class XII CGPA'], edgecolor='red',linewidth=3.6,label='GEN\nMEAN:8.07 STD:1.03',fc=(0, 0, 0, 0),density=True)
plt.hist(df_obc['Class XII CGPA'], edgecolor='blue',linewidth=3.6,label='OBC\nMEAN:7.64 STD:0.91',fc=(0, 0, 0, 0),density=True)
plt.hist(df_sc['Class XII CGPA'], edgecolor='orange',linewidth=3.6,label='SC\nMEAN:7.46 STD:0.92',fc=(0, 0, 0, 0),density=True)
plt.hist(df_st['Class XII CGPA'], edgecolor='green',linewidth=3.6,label='ST\nMEAN:6.93 STD:1.17',fc=(0, 0, 0, 0),density=True)
plt.title('Normalised Class XII CGPA', fontsize=14)
plt.xlabel('CGPA', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(prop={'size':11.25})
plt.grid()
plt.show()




plt.hist(df_gen['Class X CGPA'], edgecolor='red',linewidth=3.6,label='GEN\nMEAN:8.57 STD:1.13',fc=(0, 0, 0, 0),density=True)
plt.hist(df_obc['Class X CGPA'], edgecolor='blue',linewidth=3.6,label='OBC\nMEAN::8.17 STD:1.14',fc=(0, 0, 0, 0),density=True)
plt.hist(df_sc['Class X CGPA'], edgecolor='orange',linewidth=3.6,label='SC\nMEAN:7.90 STD:1.05',fc=(0, 0, 0, 0),density=True)
plt.hist(df_st['Class X CGPA'], edgecolor='green',linewidth=3.6,label='ST\nMEAN:7.60 STD:1.40',fc=(0, 0, 0, 0),density=True)
plt.title('Normalised Class X CGPA', fontsize=14)
plt.xlabel('CGPA', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(prop={'size':11.25})
plt.grid()
plt.show()




plt.hist(df_gen['B.Tech CGPA'], edgecolor='red',linewidth=3.6,label='GEN \nMEAN:7.66 STD:0.77',fc=(0, 0, 0, 0),density=True)
plt.hist(df_obc['B.Tech CGPA'], edgecolor='blue',linewidth=3.6,label='OBC \nMEAN:7.45 STD:0.79',fc=(0, 0, 0, 0),density=True)
plt.hist(df_sc['B.Tech CGPA'], edgecolor='orange',linewidth=3.6,label='SC\nMEAN:7.15 STD:0.62',fc=(0, 0, 0, 0),density=True)
plt.hist(df_st['B.Tech CGPA'], edgecolor='green',linewidth=3.6,label='ST\nMEAN:6.77 STD:0.72',fc=(0, 0, 0, 0),density=True)
plt.title('Normalised Class XII CGPA', fontsize=14)
plt.xlabel('CGPA', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(prop={'size':11.25})
plt.grid()
plt.show()


#IN GENDER_CONVERTED 1 MEANS MALE , 0 MEANS FEMALE.
'''GATE Marks are out of 100'''

'''
EXTRAAAAAAA
df_qual_noncfti_cs=df_qualified_noncfti[(df_qualified_noncfti['Gate Branch']=='CS: Computer Science and Information Technology')]
plt.hist(df_qual_cfti_cs['GATE Marks'], edgecolor='red',linewidth=3.6,label='cfti_cs',fc=(0, 0, 0, 0),density=True)
 
'''

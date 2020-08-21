import pandas as pd
import numpy as np

#################################################
#####################Example#####################
#################################################
#import transformation
#
#titanicCleaned = titanicTrain.copy()
#titanicCleaned = transformation.fillTitanicNa(titanicCleaned,titanicCleaned,True,True)
#
## This is the features data from the training set without the PassengerId
#X = titanicCleaned.drop(['Survived','PassengerId'],axis=1)
#
## This is the response variable from the training set
#y = titanicCleaned['Survived']
#
## This is the test data including the PassengerId
#test = pd.read_csv(os.path.join(Directory.dataPath,'test.csv'))
#
## This is the test data without the PassengerId
#test2 = test.drop(['PassengerId'],axis=1)
#
#<a model>.fit(X,y)
#
#titanicTestCleaned = transformation.fillTitanicNa(test,titanicTrain.copy(),True,True)
#titanicTestCleaned.drop('PassengerId',inplace=True,axis=1)
#transformation.makeColumnsEqual(titanicTestCleaned,X)
#
## lg2 was already trained, we use it here in the function to do the prediction
#transformation.PredictAndSave2(test2,titanicTrain.copy(),X,lg2,os.path.join(Directory.outputPath,'GenderPrediction_LogisticRegression.csv'),test).head()


#################################################
#################Variables################
#################################################


dic_OrigMap = {'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master':'Master','Don':'Don','Rev':'Rev',
    'Mme':'Mme',
    'Ms':'Ms','Major':'Major', 'Lady':'Lady', 'Sir':'Sir', 'Mlle':'Mlle', 
    'Col':'Col', 'Capt':'Capt', 'Countess':'Countess','Jonkheer':'Jonkheer', 'Dr':'Dr', 'Dona':'Dona'}

dic_Map = {'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master':'Master','Don':'Mr','Rev':'Rev',
    'Mme':'Mrs',
    'Ms':'Ms','Major':'Mr', 'Lady':'Mrs', 'Sir':'Sir', 'Mlle':'Miss', 
    'Col':'Col', 'Capt':'Mr', 'Countess':'Mrs','Jonkheer':'Sir', 'Dr':'Dr', 'Dona':'Mrs'}


#################################################
#################Helper Functions################
#################################################

def getTitleFromTitanic(x):
    '''
    A function to extract the title from a passenger name
    x:= Name of passenger
    '''
    for elt in x.split():
        if('.' in elt):
            s = elt.replace('.','')
            return s
    return ''
    
def GetTicketNo(x):
    a = x.split()
    n = len(a)
    if n == 1:
        if(a[0] != 'LINE'):
            return int(a[0])
        else:
            return 0
    else:
        return int(a[n-1])
        
def BucketTicketNo(x):
    if x < 10000:
        return 'l10'
    elif x < 40000:
        return 'l40'
    elif x < 110000:
        return 'l110'
    elif x < 260000:
        return 'l260'
    elif x < 330000:
        return 'l330'
    else:
        return 'g330'
        
def GetFirstCharacter(st):
    '''
    A function to extract the first letter (lower case) from a string. If string is null then output is an empty string
    st: A string
    '''
    if(pd.isnull(st) or len(st)==0):
        return 0
    else:
        return ord(st[0].lower())
        
def TrueFalse(p):
    '''A function to null to 0'''
    if(p):
        return 1
    else:
        return 0
        
def setAge(row,meanAge):
    '''
    This function is intended to be used via the apply() funtion and applies to rows in the format (Age,sex,Pclass).
    This function replaces any null Age with the mean Age of the same sex and Pclass as given in the meanAge argument/dataframe.
    '''
    Age = row[0]
    male = row[1]
    Pclass = row[2]
    TotalPpl = row[3]
    if(pd.isnull(Age) & (np.sum(meanAge['TotalPpl'] == TotalPpl) > 0)):
        return int(meanAge[(meanAge['Pclass'] == Pclass) & (meanAge['male'] == male) & (meanAge['TotalPpl'] == TotalPpl)]['Age'].iloc[0])
    elif(pd.isnull(Age)):
        return int(meanAge[(meanAge['Pclass'] == Pclass) & (meanAge['male'] == male)]['Age'].iloc[0])
    else:
        return int(Age)
        
def setAge0(row,meanAge):
    '''
    This function is intended to be used via the apply() funtion and applies to rows in the format (Age,sex,Pclass).
    This function replaces any null Age with the mean Age of the same sex and Pclass as given in the meanAge argument/dataframe.
    '''
    Age = row[0]
    male = row[1]
    Pclass = row[2]
    if(pd.isnull(Age)):
        return int(meanAge[(meanAge['Pclass'] == Pclass) & (meanAge['male'] == male)]['Age'].iloc[0])
    else:
        return int(Age)
        
def setFare(row,meanFare):
    '''
    This function is intended to be used via the apply() funtion and applies to rows in the format (Fare,sex,Pclass).
    This function replaces any null Fare with the mean Fare of the same sex and Pclass as given in the meanFare argument/dataframe.
    '''
    Fare = row[0]
    
    if pd.isna(Fare):
        return 0
    else:
        return Fare
    
    male = row[1]
    Pclass = row[2]
    TotalPpl = row[3]
    if(pd.isnull(Fare)  & (np.sum(meanFare['TotalPpl'] == TotalPpl) > 0)):
        return int(meanFare[(meanFare['Pclass'] == Pclass) & (meanFare['male'] == male) & (meanFare['TotalPpl'] == TotalPpl)]['Fare'].iloc[0])
    elif(pd.isnull(Fare)):
        return int(meanFare[(meanFare['Pclass'] == Pclass) & (meanFare['male'] == male)]['Fare'].iloc[0])
    else:
        return int(Fare)
        
def setFare0(row,meanFare):
    '''
    This function is intended to be used via the apply() funtion and applies to rows in the format (Fare,sex,Pclass).
    This function replaces any null Fare with the mean Fare of the same sex and Pclass as given in the meanFare argument/dataframe.
    '''
    Fare = row[0]
    
    if pd.isna(Fare):
        return 0
    else:
        return Fare
    
    male = row[1]
    Pclass = row[2]
    if(pd.isnull(Fare)):
        return int(meanFare[(meanFare['Pclass'] == Pclass) & (meanFare['male'] == male)]['Fare'].iloc[0])
    else:
        return int(Fare)
        
#################################################
#################Dataframe Functions#############
#################################################

def addTitle(df):
    df['Title'] = df['Name'].apply(getTitleFromTitanic)
    df['Title'] = df['Title'].map(dic_Map)
    
def addTicketBucket(df):
    df['TicketBucket'] = (df['Ticket'].apply(GetTicketNo)).apply(BucketTicketNo)
    
def addNameLength(df):
    df['NameLengthBucket'] = df['Name'].apply(lambda x: int(len(x)/10)*10)
    
def addCabinExists(df):
    df['CabinExistence'] = pd.isnull(df['Cabin']).apply(lambda x: not x)
    
def addCabinFirstLetter(df):
    df['CabinFirstLetter'] = df['Cabin'].apply(GetFirstCharacter)
    
def addCabinLength(df):
    df['CabinLength'] = df['Cabin'].apply(lambda x: 0 if str(x) == 'nan' else len(x))
    
def addTotalPpl(df):
    df['TotalPpl'] = df['Parch'] + df['SibSp']
    
def addIsChild(df):
    df['IsChild'] = df['Age'].apply(lambda x: 1 if x < 16 else 0)
    
def addIsOAP(df):
    df['IsOAP'] = df['Age'].apply(lambda x: 1 if x >= 60 else 0)
    
def addIsZeroFare(df):
    df['IsZeroFare'] = df['Fare'].apply(lambda x: 1 if x == 0 else 0)
    
def addCabinCount(df):
    df['CabinCount'] = df['Cabin'].apply(lambda x: len(str(x).split()))
    
def makeColumnsEqual(df,df1):
    for col in df1.columns:
        if col not in df.columns:
            print('Added col {}'.format(col))
            df[col] = 0
    for col in df.columns:
        if col not in df1.columns:
            print('Removed col {}'.format(col))
            df.drop(col,inplace=True,axis=1)
           
def addFare(df):
    df['HasFare'] = df['Fare'].apply(lambda x: 0 if pd.isnull(x) else 1) 
    
def addFareBucket(df):
    df['FareBucket'] = df['Fare'].apply(lambda x: 'Unknown' if pd.isnull(x) else '0' if x == 0 else '<50' if x < 50 else '<100' if x < 100 else '>100' if x >= 100 else 'Unknown')
    fareBucket = pd.get_dummies(df['FareBucket'],drop_first=True,prefix='farebucket_')
    df.drop(['FareBucket'],axis=1,inplace=True)
    df = pd.concat([df,fareBucket],axis=1)
    
    return df
    
def addAge(df):
    df['AgeCleaned'] = df['Age'].apply(lambda x: 0 if pd.isnull(x) else x)
    
    
#################################################
#################Overall Functions###############
#################################################

def cleanTitanicData(df):
    '''
    A function to clean the titanic dataframe according to the EDA above
    '''
    
    #Sex is either male or female. Create a series which has entries 0 and 1 specifying whether it's one or the other
    sex = pd.get_dummies(df['Sex'],drop_first=True)

    # PClass get dummies
    pclass = pd.get_dummies(df['Pclass'],drop_first=True,prefix='pclass_')
    
    # Add Age
    addAge(df)
    
    #Similar to the sex series
    #embark = pd.get_dummies(df['Embarked'],drop_first=True)

    #Cabin either exists or doesn't -> 1 or 0
    addCabinExists(df)
    
    # Get the total people
    #addTotalPpl(df)
    
    # Get the ticket number and then bucket them into >200000 and < 200000
    #addTicketBucket(df)
    #TicketBucket = pd.get_dummies(df['TicketBucket'],drop_first=True)
    
    #Get the title from the name
    #addTitle(df)
    #df['Title'] = df['Title'].map(dic_Map)
    #Title =  pd.get_dummies(df['Title'],drop_first=True)
    
    # Get the name lenth and bucket it
    #addNameLength(df)
    
    #addCabinCount(df)
    
    #Get rid of some columns
    df_temp = df.drop(['Sex','Embarked','Name','Ticket','Cabin','Age','Pclass', 'SibSp', 'Parch','Fare'],axis=1,inplace=False)
    
    #Concat the series to the data frame
    df_temp = pd.concat([df_temp,sex,pclass],axis=1)
    
    return df_temp
    
    
def fillTitanicNa(df,trainingData,fillAge=True,fillFare=True):
    '''
    This function takes a titanic data frame, cleans it and fills in missing Age and Fare values. If it is desired
    that the Age not be filled the fillAge should be set to False. Likewise with fillFare.
    '''
    
    # Clean and transform the data
    df_temp = cleanTitanicData(df)
    df_train = cleanTitanicData(trainingData)

    # Get the mean Fare values per grouping
    #t_Fare = pd.DataFrame(df_train.dropna().groupby(['Pclass','male','TotalPpl']).mean()['Fare'])
    #t_Fare = pd.DataFrame(t_Fare.to_records())

    # Get the mean Age values per grouping
    #t_Age = pd.DataFrame(df_train.dropna().groupby(['Pclass','male','TotalPpl']).mean()['Age'])
    #t_Age = pd.DataFrame(t_Age.to_records())
    
    
    #if(fillAge == True):
        #df_temp['Age'] = df_temp[['Age','male','Pclass','TotalPpl']].apply(lambda row: setAge(row,t_Age),axis=1)
        #df_temp['Age'] = df_temp[['Age','male']].apply(lambda row: setAge0(row,t_Age),axis=1)

    #if(fillFare == True):
        #df_temp['Fare'] = df_temp[['Fare','male','Pclass','TotalPpl']].apply(lambda row: setFare(row,t_Fare),axis=1)
        #df_temp['Fare'] = df_temp[['Fare','male']].apply(lambda row: setFare0(row,t_Fare),axis=1)
        
    #if(fillAge == False or fillFare == False):
    #    df_temp.dropna(inplace = True)
        
    #addIsChild(df_temp)
    #addIsOAP(df_temp)
    #addIsZeroFare(df_temp)
        
    return df_temp
    
def formatTitanicPredictions2(df_pred,test,original):
    '''
    A function to format the predictions
    df_pred:= the predictions dataframe without the PassengerIds
    test:= the original test dataframe where the PassengerIds still exist
    '''
    df_FullPred = pd.concat([df_pred,test],axis=1)
    df_FullPred = df_FullPred.dropna()
    df_FullPred['PassengerId'] = original['PassengerId'].astype(int)
    df_FullPred = df_FullPred.set_index('PassengerId')
    return df_FullPred
    
def PredictAndSave2(df,traindata,trainX,lg,path,original):
    test = df
    titanicTestCleaned = fillTitanicNa(test,traindata,True,True)
    makeColumnsEqual(titanicTestCleaned,trainX)
    predictions = lg.predict(titanicTestCleaned)
    df_pred = pd.DataFrame(predictions,columns=['Survived'])
    df_FullPred = formatTitanicPredictions2(df_pred,titanicTestCleaned,original)
    df_FullPred['Survived'].to_csv(path,header=True)
    return df_FullPred
    
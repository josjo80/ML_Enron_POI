#!/usr/bin/python


def outlierCleaner(df):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    import pandas as pd
    import numpy as np
    
    #del df['poi']
    fms = [] #feature, mean, and std
    for column in df:
        m = np.mean(df[column])
        s = np.std(df[column])
        fms.append([column, m, s])
            
    #print fms
    
    outliers = []
    for index, row in df.iterrows():
        for e,i in zip(row, fms):
            if e > (i[1]+3*i[2]):
                outliers.append([index, row[0], e, i[0], i[1]])

    
    #print len(outliers)
    #print outliers
    
    return df, outliers


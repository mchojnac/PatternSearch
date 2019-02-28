# Here is class which is doing pattern search
# it looks for similar pattern as given in a data
# it uses Fourier series expansion

import collections
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class PatternSearch(object):
    def __init__(self,pattern,data,n=10,n_min=-1,n_max=-1, noscale=False):
        # pattern to search as data frame
        self.__pattern=pattern
        # data frame with data to look
        self.__data=data
        # number of fourier series elements to be use during calculation
        self.__n=n
        # min max numbers of setps to look for
        if n_min<0:
            self.__n_min=pattern.shape[0]
        else:
            self.__n_min=n_min
        if n_max<0:
            self.__n_max=pattern.shape[0]
        else:
            self.__n_max=n_max
        # setting switch off scale flags
        self.__noScaleFlags=self.__SetScaleFlags(noscale)
        return



    # internal function to be used during checking input data frame column names , it checks in list matches
    __compare = classmethod(lambda self,x, y: collections.Counter(x) == collections.Counter(y))

    # Setter for n
    def SetNcoefficients(self,n):
        self.__n=n
        return

    # Set if flags of not using a0 , if yes than mean of signal will be neglected
    def __SetScaleFlags(self,flags):
        toreturn_flags = {el:False for el in self.__pattern.columns} # by default we do not switch of scale
        if type(flags) is bool: # for  one bool everything will have the same value
            if flags :
                toreturn_flags = {el:flags for el in self.__pattern.columns}
        elif type(flags) is dict: # puts value from input dictionary
            for i in flags.keys():
                if i in self.__pattern.columns:
                    toreturn_flags[i]=flags[i]
        else:
            print("wrong format of Flags using defaults")
        # set flags in order of columns
        return [toreturn_flags[i] for i in toreturn_flags.keys()]

    # Check if data allows for search
    def __CheckData(self):
        # different columns in data frames
        if not self.__compare(self.__pattern.columns,self.__data.columns):
            return False
        # Checking on numerical data type
        if self.__pattern.select_dtypes(include=np.number).shape[1]!=self.__pattern.shape[1]:
            return False
        if self.__data.select_dtypes(include=np.number).shape[1]!=self.__data.shape[1]:
            return False
        # if min max are correct
        if self.__n_min>self.__n_max:
            return False
        # if min is not too long
        if self.__n_min>self.__data.shape[0]:
            return False
        # if there is enough  we want more conf than data
        if self.__data.shape[0]<=2*self.__n:
            return False
        if self.__pattern.shape[0]<=2*self.__n:
            return False
        return True

    # normalize data so different columns with differnt scale weight the same
    # normalization is done to -1 1 value in pattern
    # those factor are use for data
    def __NormalizedPatternandData(self):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(self.__pattern)
        return scaler.transform(self.__pattern).transpose(), scaler.transform(self.__data).transpose()


    # Get Fourier series for one time series
    def __GetConf1D(self, f_values,noscaleflag):
        compelex_results=np.fft.rfft(f_values)/len(f_values)
        compelex_results*=2
        # a0 constant factor, mean value
        if noscaleflag: # in case of no scale set to zero
            a0=0.0
        else:
            a0=compelex_results[0].real/2.0
        a=compelex_results[1:self.__n].real
        b=compelex_results[1:self.__n].imag*-1.0
        # returns 2n-1 array 0 is constant next n are cos terms last n are sin terms
        return  np.concatenate((np.full(1,a0),a,b))

    # make on array of coefficient for all columns
    def __GetConf(self,df_values):
        return np.concatenate([self.__GetConf1D(df_values[i],self.__noScaleFlags[i]) for i in range(0,df_values.shape[0])])

    #Calculates distance in data to pattern for all possibilities of length
    def __CalculateDistance(self):
        # normalize Data
        patternNormalized,dataNormalized=self.__NormalizedPatternandData()
        # pattern series fourier transfrom
        pattern_con=self.__GetConf(patternNormalized)
        # get size of distance table all possible lengths of found pattern and put -1 as initialization
        Npattern=patternNormalized.shape[1]
        Ndata=dataNormalized.shape[1]
        N1=Ndata-self.__n_min+1
        N2=self.__n_max-self.__n_min+1
        distance=np.full(N1,-1.0)
        distance_length=np.full(N1,-1.0)
        #loop over all possible start points
        for i in range(N1):
            #loop over all possible lengths
            tmp_distance=list()
            for j in range(self.__n_min,self.__n_max+1):
                if i+j>Ndata:
                    break
                tmp_conf=self.__GetConf(dataNormalized[:,i:i+j])
                #distance[i,j-self.__n_min]=np.sqrt(np.sum((tmp_conf-pattern_con)**2))
                tmp_distance.append(np.sqrt(np.sum((tmp_conf-pattern_con)**2)))
            distance[i]=min(tmp_distance)
            distance_length[i]=tmp_distance.index(distance[i])
        return distance,distance_length

    # returns top N matched patterns
    # in case there is not enough patterns  less will be returned
    def __GetTopNScores(self,distance,distance_length,Nscores):
        # fill all -1 (where we do not have data with value max+1)
        # max + 1 value is used as NAN value
        # after finding matching part removes points which are close to avoid find only one part with different lengthss
        max_v=int(distance.max())+1
        distance[distance==-1]=max_v
        # initial values
        min_v=distance.min()
        n_found=0
        results=list()


        while (n_found<Nscores and min_v<max_v): # while we have not found everything and we still have valid data
            # find index of min
            min_index=distance.argmin()
            # pattern start and end
            start_of_found_pattern=int(min_index)
            end_of_found_pattern=int(min_index+distance_length[min_index]+self.__n_min)
            #print(start_of_found_pattern)
            #print(end_of_found_pattern)
            #print(self.__data.shape)
            results.append((min_v,self.__data[start_of_found_pattern:end_of_found_pattern]))
            # cut the neighbour region
            # number of neighbours to be remove
            Ncut=int((distance_length[min_index]+self.__n_min)*0.5)
            Nlow=max(0,min_index-Ncut)
            Nhigh=min(distance.shape[0],min_index+Ncut)
            distance[Nlow:Nhigh,]=max_v
            # get values for next step
            n_found=n_found+1
            min_v=distance.min()
        return results

    def PlotResults(self,results):
        for i in results:
            # because pattern and results can have different length
            # we alignment them than shorter is in center of longer

            N_pattern=self.__pattern.shape[0]
            N_results=i[1].shape[0]
            N_points=max([N_pattern,N_results])
            x_values=np.linspace(0,N_points,N_points,False)
            N_shift_pattern=int((N_points-N_pattern)*0.5)
            N_shift_results=int((N_points-N_results)*0.5)
            # print info
            print("score={}".format(i[0]))
            print("found pattern is from {} to {}".format(i[1].index[0],i[1].index[-1]))
            print("It is {} steps different from pattern".format(N_results-N_pattern))
            # draw column by column
            pattern_means=self.__pattern.mean()
            data_means=i[1].mean()
            for idx_col,col in enumerate(self.__pattern.columns):
                # in case of no scale flag on we have do minus mean to do proper comparison
                if self.__noScaleFlags[idx_col]:
                    mean_pattern=pattern_means[col]
                    mean_data=data_means[col]
                    adjusted_values=" (adjusted values) "
                else:
                    mean_pattern=0.0
                    mean_data=0.0
                    adjusted_values=" (standard values) "
                plt.plot(x_values[N_shift_pattern:N_shift_pattern+N_pattern],self.__pattern[col]-mean_pattern,"b--",label="pattern",linewidth=3.0,alpha=0.5)
                plt.plot(x_values[N_shift_results:N_shift_results+N_results],i[1][col]-mean_data,"r-",label="found",linewidth=1.0)
                plt.title(col+adjusted_values)
                plt.legend()
                plt.show()
        return


    def FindPatterns(self,Nscores=5,plot=False): # number of top scores to
       # bad data get out
       # if data calculations are not done
        if not self.__CheckData():
            print("Problem with inputs lack of results")
            return list()
        # do calculations
        distance,distance_length=  self.__CalculateDistance()
        results=self.__GetTopNScores(distance,distance_length,Nscores)
        if plot:
           self.PlotResults(results)
        return results

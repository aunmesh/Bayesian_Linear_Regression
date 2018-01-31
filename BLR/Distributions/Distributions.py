#!/usr/bin/python

class Distribution(object):

    def __init__(self,Params):
        self.Params = Params
        
    def Probab(self,input):
        raise NotImplementedError
    '''
    def GetParamList(self):
        raise NotImplementedError
    '''
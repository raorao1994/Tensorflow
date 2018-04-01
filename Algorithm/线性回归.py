# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 14:52:30 2018

@author: Administrator
"""
#import tensorflow as tf;

#函数
def f(x):
    return x*x-2*x+1;

#求导
def g(x):
    return 2*x-2;

    
def gradientdescent(x,rate):
    i=0;
    _x=x;
    while i>=0 :
        grad=g(_x);
        _x+=grad*rate;
        y=f(_x);
        print("i="+str(i)+" x="+str(_x)+" y="+str(y));
        if abs(grad)<0.01:
            break;
            
if __name__ =="__main__":
    x=-5.0;
    rate=0.1;
    gradientdescent(x,rate);



        
    
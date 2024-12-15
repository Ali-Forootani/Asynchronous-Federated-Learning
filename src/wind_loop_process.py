#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:30:12 2023

@author: forootani
"""
import torch 




##################################
######### Debugging modules


class WindLoopProcessor:
    def __init__(self, model_fn, wind_loss_func):
        self.model_fn = model_fn
        
        self.wind_loss_func = wind_loss_func
        
    def __call__(self, loader):
        loss_data_total = 0
        
        for batch_idx, (input_data, output_data) in enumerate(loader):
            u_pred = self.model_fn(input_data)
            
            
            
            #loss_data_mse = torch.mean((output_data - u_pred) ** 2)
            
            loss_data = self.wind_loss_func(output_data, u_pred)
            

        return loss_data


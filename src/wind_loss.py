#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:18:29 2024

@author: forootan
"""

import torch

## loss function for Wind data training

def wind_loss_func(u, u_pred):

    loss_data = torch.mean((u - u_pred) ** 2)
   
    

    return loss_data

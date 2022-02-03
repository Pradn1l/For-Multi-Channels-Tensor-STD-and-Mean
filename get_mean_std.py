#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Pradnil Kamble
"""


def get_mean_std(loader):
      # var[X] = E[X**2] - E[X]**2
      channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0
  
      
      torch.set_printoptions(profile="full")
      
      for idx, (inputs,labels) in enumerate(loader):
          
          if idx == 2:
              print("before squeeze idx2: ", inputs.shape)
          
          inputs = inputs.squeeze(0).float()
          
          '''
          if idx == 2:
              print("idx2: ", inputs.shape)
              outfile1.write("shape: " + str(inputs) + "\n")
              outfile1.write(str(inputs))
           '''  
          channels_sum += torch.mean(inputs, dim=[0, 2, 3])
          channels_sqrd_sum += torch.mean(inputs ** 2, dim=[0, 2, 3])
          num_batches += 1
  
      mean = channels_sum / num_batches
      std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5
      
      torch.set_printoptions(profile="default")
  
      return mean, std


if __name__ == "__main__":

    csv_dir_ = "/home/Kamble/"
    img_dir = "/home/Kamble/"
    outfile1= open("/home/Kamble/", 'w')
    
    train_set = DataLoader(img_dir, csv_dir)
    train_loader =DataLoader(dataset = train_set,
        shuffle = True)

    
    mean, std = get_mean_std(train_loader)
    print(mean)
    print(std)
    
   

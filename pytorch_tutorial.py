import torch
import numpy as np

#empty tensor of size 3 
#x = torch.empty(3)
#2D tensor
#can have different data types
#y = torch.empty(2, 3, dtype = torch.int)
#z = torch.tensor([2.5, 0.1])

x = torch.rand(2,2)
y = torch.rand(2,2)
print(x)
print(y)
z = x+y   #element wise addition  
# z = torch.add(x,y)    this does the same thing

#every _ function will do modifying operation
y.add_(x)  #modifies y by adding elements of x to it
# z = x - y   is  equal to  z = torch.sub(x, y)
# z = x * y   is equal to z = torch.mul(x, y)
# z = x / y   is equal to z = torch.div(x, y)

"""
a = torch.rand(5, 3)
print(a)
print(a[:, 0])  #slicing, all the rows, but only column 0
print(a[1, :])  #row 1, all columns
print(a[1, 1])  #just one element
print(a[1, 1].item())  #gets a value instead of tensor, can use if only one element
"""

# y = x.view(16) will create tensor of 16 elements, 1D
# y = x.view(-1, 8)  multiple dimension, 8 elements per row 
# this is how you resizes tensor. Number of row will be automatically determined

a = torch.ones(5)  #tensor with 5 1s 
b = a.numpy()  #to numpy array
#careful, if tensor run on cpu, a and b will share the same memory space and changing one will change the other
a = np.ones(5)
b = torch.from_numpy(a)  # to tensor  #default data type to float64

"""
This will change the value of b
print(b)   
a+=1
print(b)
"""

#putting tensor in GPU 
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device = device)
    #this will put tensor on GPU
    #or
    y = torch.ones(5)
    y = y.to(device)
    
    z = x+y      
    #z.numpy()   WONT WORK, numpy can only handle CPU tensor
    z = z.to("CPU") #have to turn to CPU tensor 
    z.numpy()
    
    
c = torch.ones(5, requires_grad=True)  #default is false
#will tell pytorch it will need to calculate the gradients ???
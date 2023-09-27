import torch
import numpy as np

x = torch.randn(3, requires_grad=True)  #if we want to calculate the gradients of function respect to x, we specify this to true
print(x)
y = x+2  #node is +. input is x value and 2. output is y value. 
#we can calculate gradient(derivative)
print(y)
z= y*y*2
z = z.mean()  
print(z)   

z.backward()  #dz/dx  #if not scalar, we must give a vector ex) z.backward(v), v being a vector 

# x.requires_grad_(False)   -> modifies x so that it does not require gradient 
# y = x.detach()        -> second option, create copy of x to y, not requiring gradient
# with torch.no_grad():
#   y = x + 2             -> create copy of x to y, not requiring gradient 
print(x.grad) 
import time
import cv2
import numpy as np 
x = np.random.rand(1280,1280,100) > 0.5
y = np.random.rand(100) > 0.5
start = time.time()

x = x[..., y]

end = time.time()
print(end - start)

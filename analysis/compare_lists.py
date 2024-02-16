import numpy as np

list_1 = np.loadtxt("data/basin_list.txt")
list_2 = np.loadtxt("data/basin_list2.txt")

count=0
for i in list_1:
    for j in list_2:
        if i==j:
            count +=1 

print(count/len(list_2)*100)
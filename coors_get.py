import os 
import numpy as np

coor_files = []
for (dirpath, dirnames, filenames) in os.walk(r'D:\Python\OCR\text-recognition\result'):
    for file in filenames:
        filename, ext = os.path.splitext(file)
        ext = str.lower(ext)
        if ext == '.txt':
            coor_files.append(os.path.join(dirpath, file))
            
def len_coor_file():
    len_file_coors = []
    for x in coor_files:
        count = 0
        f = open(x, "r")
        for i in f:
            count = count + 1
        f.close() 
        len_file_coors.append(count)
    return len_file_coors

def get_coors(file, index):
    coors_final = []
    coors_round = []
    coors_1 = []
    f = open(file, "r")
    for i in f:
        coors_1.append(i)
    f.close
    coors_round.append(coors_1[index])
    coors_round = coors_round[0].split(",")
    for x in coors_round:
        coors_final.append(int(x))
    coors_round.clear()
    for x in range(0, len(coors_final), 2):
        coors_round.append([coors_final[x], coors_final[x+1]])
    coors_final.clear()
    coors_final = np.array(coors_round)
    return coors_final
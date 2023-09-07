import os
import shutil
import tqdm

with open('/Users/leeching/Desktop/pro/35POVs/F16.pov','r',encoding='utf-8') as file1:
    lines = []
    for line in tqdm.tqdm(file1.readlines()):

        if 'pigment' in line:
            replace = 'pigment { color rgb<1, 1, 1> }\n'
            lines += [replace]
            continue
        if 'finish' in line:
            replace = 'finish {reflection {0.3} ambient 0 diffuse 0.002 specular 0.7 roughness 0.00085}\n'
            #replace = 'finish {reflection {0.3} ambient 0 diffuse 0.005 specular 0.7 roughness 0.00001}\n'
            lines += [replace]
            continue
        lines += [line]
file1.close()

with open('/Users/leeching/Desktop/pro/35Targets/F16.pov','w',encoding='utf-8') as file2:
    for line in tqdm.tqdm(lines):
        file2.write('%s' %line)
file2.close()


import shutil
import os

print(os.path.exists('../tmp/train.txt'))
f = open('../tmp/train.txt', 'r')
lines = f.readlines()
"""
for line in lines:
    #print(line.split('/')[-1][:-1])
    #line = "/".join(line.split('/')[2:])
    line = line[:-1].replace('\\','/')
    if (os.path.exists(line)):
        os.system("copy " + line.replace('/', '\\') + " ../VOC/images/train/".replace('/', '\\'))
print(os.path.exists('../tmp/train.txt'))
f = open('../tmp/train.txt', 'r')
lines = f.readlines()
"""
for line in lines:
    #print(line.split('/')[-1][:-1])
    #line = "/".join(line.split('/')[2:])
    line = line[:-1].replace('\\','/')
    line = line.replace('JPEGImages', 'labels')
    line = line.replace('jpg', 'txt')
    #print(line)
    if (os.path.exists(line)):
        os.system("copy "+ line.replace('/', '\\') + " ../VOC/labels/train/".replace('/', '\\'))

print(os.path.exists('../tmp/2007_test.txt'))
f = open('../tmp/2007_test.txt', 'r')
lines = f.readlines()

for line in lines:
    #print(line.split('/')[-1][:-1])
    #line = "/".join(line.split('/')[2:])
    line = line[:-1].replace('\\','/')
    if (os.path.exists(line)):
        os.system("copy "+ line.replace('/', '\\') + " ../VOC/images/val/".replace('/', '\\'))

print(os.path.exists('../tmp/2007_test.txt'))
f = open('../tmp/2007_test.txt', 'r')
lines = f.readlines()

for line in lines:
    #print(line.split('/')[-1][:-1])
    #line = "/".join(line.split('/')[2:])
    line = line[:-1].replace('\\','/')
    line = line.replace('JPEGImages', 'labels')
    line = line.replace('jpg', 'txt')
    #print(line)
    if (os.path.exists(line)):
        os.system("copy "+ line.replace('/', '\\') + " ../VOC/labels/val/".replace('/', '\\'))

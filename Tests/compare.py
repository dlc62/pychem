from __future__ import print_function
import sys

if len(sys.argv) != 3:
   print('Usage: compare.py <file1> <file2>')
   sys.exit()
else:
   file1 = sys.argv[1]
   file2 = sys.argv[2]

with open(file1,'r') as f1:
   fc1 = f1.readlines()
with open(file2,'r') as f2:
   fc2 = f2.readlines()

small = 1.e-12
for i,(line1,line2) in enumerate(zip(fc1[1:],fc2[1:])):
   diff = [abs(float(x1)-float(x2)) for x1,x2 in zip(line1.split(),line2.split())]
   mad = sum(diff)
   if mad > small:
     print(i)
     print(line1,line2) 


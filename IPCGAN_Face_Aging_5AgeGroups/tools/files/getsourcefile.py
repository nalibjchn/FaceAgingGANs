#source file
import os
import sys

if (len(sys.argv) > 1):
    path = sys.argv[1]
    print(sys.argv)
else:
    path = '../../../DATA/CACD2000_align_withGender'

f = open('../sourcefile.txt', 'a')
f.truncate(0)

for fname in os.listdir(path):
    age = int(fname.split("_")[0])
    f.write("%s %s\n"%(fname,str(age)))
f.close()
print("source code done")
import os
import sys

if (len(sys.argv) > 1):
    path = sys.argv[1]
    print(sys.argv)
else:
    path="../../images/test"
agegrp_1=0
agegrp_2=0
agegrp_3=0
agegrp_4=0
agegrp_5=0
test_age_0=[]
test_age_1=[]
test_age_2=[]
test_age_3=[]
test_age_4=[]

for fname in os.listdir(path):
    age = int(fname.split("_")[0])
    # fname_str = fname.decode('utf-8')
    if(age>=11 and age<=20):
        test_age_0.append(fname + ' ' + str(age))
    if (age >= 21 and age <= 30):
        test_age_1.append(fname + ' ' + str(age))
    if (age >= 31 and age <= 40):
       test_age_2.append(fname + ' ' + str(age))
    if (age >= 41 and age <= 50):
	    test_age_3.append(fname + ' ' + str(age))
    if (age >= 51):
	    test_age_4.append(fname + ' ' + str(age))

with open('../../test_data/test_age_group_0.txt',"a") as f:
    f.truncate(0)
    for fname in test_age_0:
        f.write("%s\n"%fname)
    f.close()

with open('../../test_data/test_age_group_1.txt', "a") as f:
    f.truncate(0)
    for fname in test_age_1:
        f.write("%s\n" %fname)
    f.close()
with open('../../test_data/test_age_group_2.txt', "a") as f:
    f.truncate(0)
    for fname in test_age_2:
        f.write("%s\n" %fname)
    f.close()
with open('../../test_data/test_age_group_3.txt', "a") as f:
    f.truncate(0)
    for fname in test_age_3:
        f.write("%s\n" %fname )
    f.close()
with open('../../test_data/test_age_group_4.txt', "a") as f:
    f.truncate(0)
    for fname in test_age_4:
        f.write("%s\n" %fname )
    f.close()
print("Test files:done")
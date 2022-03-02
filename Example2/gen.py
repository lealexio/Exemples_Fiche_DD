import random

for i in range(1,30):
    for y in range(1,5):
        rateSameObject = True
        if random.randint(1, 3) == 1:
            rateSameObject = False
        while rateSameObject :
            print(str(i)+','+str(y)+','+str(round(random.uniform(1.0, 9.0),1)))
            if random.randint(1, 3) == 1:
                rateSameObject = False

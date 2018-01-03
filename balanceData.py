import random

DIR_PATH = "./../resources/"
TEST_SOL_FILE   = DIR_PATH + "test_with_solutions.csv"
RESULT=DIR_PATH + "test_with_solutions_balanced.csv"
def countClasses():

    df=open(TEST_SOL_FILE,"r")
    line=df.readline()
    count0=0
    count1=1
    i=0
    line=df.readline()
    ids0=[]
    while(line):
        line=line.split(",")
        print(line[0])
        if(int(line[0])==0):
            count0+=1
            ids0.append(i)
        if(int(line[0])==1):
            count1+=1
        line=df.readline()
        i += 1
    df.close()
    return count0,count1, ids0

def under_sample(count0, count1,ids0):
    l=count1+count0

    ids0_to_stay=[]
    i=0
    while(i<count1):
        temp=random.choice(ids0)
        if(temp not in ids0_to_stay):
            ids0_to_stay.append(temp)
            i+=1
    df=open(TEST_SOL_FILE,"r")
    dr=open(RESULT,"w")
    line=df.readline()
    dr.write(line)
    line=df.readline()
    i+=1
    while(line):
        line_elems = line.split(",")
        if (int(line_elems[0]) == 0):
            if(i in ids0_to_stay):
                dr.write(line)
        if (int(line_elems[0]) == 1):
            dr.write(line)
        i+=1
        line=df.readline()

def run():
    count0, count1,ids0= countClasses()
    print(count0+count1)
    under_sample(count0,count1,ids0)

if __name__ == '__main__':
    run()
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np

import json

'''
purpose: return number of retries and the window size
# sample input
0  1    2           3     4   5       6    7            8   9    10
[ ID] Interval           Transfer     Bandwidth       Retr  Cwnd
[  4]   0.00-1.00   sec   613 MBytes  5.14 Gbits/sec    0   3.18 MBytes       
retr: float(tmp[8]) == 0.0
cwnd: float(tmp[9]) == 3120.0
json equiv:
intervals:sum:retransmits
intervals:streams:snd_cwnd
'''
def parse_line(line):
    tmp = line.split()
    if tmp[10] == 'MBytes':
        tmp[9] = float(tmp[9])*1000
    tmpVar1 = float(tmp[8])
    tmpVar2 = float(tmp[9])
    return[float(tmp[8]),float(tmp[9])]

def read_iperf3(location):
    counter = 0
    df = pd.DataFrame(columns=['Retr','Cwnd'])

    print(location)
    # json
    # open
    if ftype == '.json':
        with open(location) as data_file:
            iperf3_data = json.load(data_file)
    # text files
    else:
        #f = open('test_reno_control_1.txt','r')
        f = open(location,'r')
        for line in f:
            counter = counter + 1
            if counter > 9 and counter < 170:
                retData = parse_line(line)
                print("Retr: " + str(retData[0]) + " Cwnd " + str(retData[1]))
                df.loc[int(counter) - 9] = parse_line(line)

        f.close()

    # get data
    if ftype == '.json':
#        import pdb; pdb.set_trace()
        print()
        for inter in iperf3_data["intervals"]:
            retx = inter["sum"]["retransmits"]
            for strm in inter["streams"]:
                cwnd = int(strm["snd_cwnd"] / 1024)
                print("Retr: " + str(retx) + " Cwnd " + str(cwnd))
                df.loc[int(counter)] = [float(retx), float(cwnd)]
                counter += 1

    return(df)

# filetype
ftype = '.txt'
ftype = '.json'
# number of tests
num_tests = 61
num_tests = 2

# types of test
algorithms = ["cubic" , "reno"]
# types of perturbation
perturbations = ['control' , 'delay' , 'loss' , 'corruption']

# now work:
for algorithm in algorithms:
    for perturbation in perturbations:
        # range needs to be from 1 to 61
        for i in range(1,num_tests):
            file_name    = "test_" + algorithm + "_" + perturbation + "_" + str(i) + ftype
            img_name     = "test_" + algorithm + "_" + perturbation + "_" + str(i) + '.png'
            plot_title   = "Test " + algorithm + " " + perturbation + " " + str(i)

            df = read_iperf3(file_name)
            ax = df['Cwnd'].plot(title=plot_title.title(),kind='line',legend=True)
            df['Retr'].plot(ax=ax,legend=True)
            plt.ylabel('Cwnd (KBytes)')
            plt.xlabel('Time (seconds)')
            plt.savefig(img_name)
            plt.close()
import sys
sys.exit(0)

    
for i in range(1,num_tests):
    df = read_iperf3('test_reno_loss_'+str(i) + ftype)
    ax = df['Cwnd'].plot(title='Test_Reno_Loss_'+str(i),kind='line',legend=True)
    df['Retr'].plot(ax=ax,legend=True)
    title = 'test_reno_loss_'+str(i)+'.png'
    plt.ylabel('Cwnd (KBytes)')
    plt.xlabel('Time (seconds)')
    plt.savefig(title)
    plt.close()

for i in range(1,num_tests):
    df = read_iperf3('test_cubic_loss_'+str(i) + ftype)
    ax = df['Cwnd'].plot(title='Test_Cubic_Loss_'+str(i),kind='line',legend=True)
    df['Retr'].plot(ax=ax,legend=True)
    title = 'test_cubic_loss_'+str(i)+'.png'
    plt.ylabel('Cwnd (KBytes)')
    plt.xlabel('Time (seconds)')
    plt.savefig(title)
    plt.close()

for i in range(1,num_tests):
    df = read_iperf3('test_reno_delay_'+str(i) + ftype)
    ax = df['Cwnd'].plot(title='Test_Reno_delay_'+str(i),kind='line',legend=True)
    df['Retr'].plot(ax=ax,legend=True)
    title = 'test_reno_delay_'+str(i)+'.png'
    plt.ylabel('Cwnd (KBytes)')
    plt.xlabel('Time (seconds)')
    plt.savefig(title)
    plt.close()


for i in range(1,num_tests):
    df = read_iperf3('test_cubic_delay_'+str(i) + ftype)
    ax = df['Cwnd'].plot(title='Test_Cubic_Delay_'+str(i),kind='line',legend=True)
    df['Retr'].plot(ax=ax,legend=True)
    title = 'test_cubic_delay_'+str(i)+'.png'
    plt.ylabel('Cwnd (KBytes)')
    plt.xlabel('Time (seconds)')
    plt.savefig(title)
    plt.close()
    
for i in range(1,num_tests):
    df = read_iperf3('test_reno_corrupt_'+str(i) + ftype)
    ax = df['Cwnd'].plot(title='Test_Reno_Corrupt_'+str(i),kind='line',legend=True)
    df['Retr'].plot(ax=ax,legend=True)
    title = 'test_reno_corrupt_'+str(i)+'.png'
    plt.ylabel('Cwnd (KBytes)')
    plt.xlabel('Time (seconds)')
    plt.savefig(title)
    plt.close()

for i in range(1,num_tests):
    df = read_iperf3('test_cubic_corrupt_'+str(i) + ftype)
    ax = df['Cwnd'].plot(title='Test_Cubic_Corrupt_'+str(i),kind='line',legend=True)
    df['Retr'].plot(ax=ax,legend=True)
    title = 'test_cubic_corrupt_'+str(i)+'.png'
    plt.ylabel('Cwnd (KBytes)')
    plt.xlabel('Time (seconds)')
    plt.savefig(title)
    plt.close()

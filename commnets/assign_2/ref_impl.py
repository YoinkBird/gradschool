import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np

import json

import os.path

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
    df = pd.DataFrame(columns=['Retr','Cwnd','Bits','Bytes'])

    debug = 0
    if debug == 1:
        print(location)
    # json
    # open
    if ftype == '.json':
        with open(location) as data_file:
            iperf3_data = json.load(data_file)
    # text files
    else:
        f = open(location,'r')
        for line in f:
            counter = counter + 1
            if counter > 9 and counter < 170:
                retData = parse_line(line)
                if debug == 1:
                    print("Retr: " + str(retData[0]) + " Cwnd " + str(retData[1]))
                df.loc[int(counter) - 9] = parse_line(line)

        f.close()

    # get data
    if ftype == '.json':
#        import pdb; pdb.set_trace()
        for inter in iperf3_data["intervals"]:
            retx = inter["sum"]["retransmits"]
            for strm in inter["streams"]:
                cwnd = int(strm["snd_cwnd"] / 1024)
                # convert to Gbits/sec
                #bits = int(strm["bits_per_second"] / (10^9) )
                #bits = int(strm["bits_per_second"] / (1000000) )
                # convert to KBytes/sec
                bits = int(strm["bits_per_second"] / (8000) )
                # TODO: TODO_201607102325 remove this once timescale in graph is fixed
                # downsample for graph
                bits /= 10
                bytes = int(strm["bytes"] / 1024)
                if debug == 1:
                    print("Retr: " + str(retx) + " Cwnd " + str(cwnd))
                df.loc[int(counter)] = [float(retx), float(cwnd), float(bits), float(bytes)]
                counter += 1

    return(df)

# filetype
ftype = '.txt'
ftype = '.json'
file_dir = "data"
file_prefix = "results"
# number of tests
num_tests = 61
num_tests = 10

# types of test
algorithms = ["cubic" , "reno"]
# types of perturbation
perturbations = ['none' , 'delay' , 'loss' , 'corruption']

# now work:
for algorithm in algorithms:
    for perturbation in perturbations:
        # range needs to be from 1 to 61
        for i in range(1,num_tests):
            file_num       = "%03d" % i
            file_root_name = file_prefix + "_" + algorithm + "_" + perturbation + "_" + file_num
            file_name      = file_dir + "/" + file_root_name + ftype
            img_name       = file_root_name + '.png'
            plot_title     = "Results " + algorithm + " " + perturbation + " " + str(i)

            #if not Path(file_name).is_file():
            if not os.path.exists(file_name):
                print("BAD DATA: " + file_name)
                continue
            df = read_iperf3(file_name)
            if df.empty:
                print("BAD DATA: " + file_name)
                continue
            ax = df['Retr'].plot(title=plot_title.title(),kind='line',legend=True)
            ax2 = ax.twinx()
            ax3 = ax.twinx()
            df['Bits'].plot(ax=ax2,legend=True)
            df['Cwnd'].plot(ax=ax,legend=True)
            # TODO: fix axis and timescale; timescale is TODO_201607102325
            plt.ylabel('Cwnd (KBytes) | KB/10s')
            plt.xlabel('Time (seconds)')
            plt.savefig(img_name)
            plt.show()
            plt.close()
            # tmphack
            import sys
            sys.exit(0)


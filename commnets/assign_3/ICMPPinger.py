from socket import *
import os
import sys
import struct
import time
import select
import binascii  

from collections import namedtuple

ICMP_ECHO_REQUEST = 8
ICMP_H = namedtuple( 'ICMP_Header', 'type code checksum packetID sequence')
ICMP_STRUCT_FORMAT = "bbHHh"
# # https://docs.python.org/2/library/struct.html#format-characters
# Format	C Type	 	Python type 		Standard size 	Notes
# x	pad byte 		no value
# c	char 			string of length	1 	1
# b	signed char 		integer 		1 	(3)
# B	unsigned char 		integer 		1 	(3)
# ?	_Bool 			bool 			1 	(1)
# h	short 			integer 		2 	(3)
# H	unsigned short 		integer 		2 	(3)
# i	int 			integer 		4 	(3)
# I	unsigned int 		integer 		4 	(3)
# l	long 			integer 		4 	(3)
# L	unsigned long 		integer 		4 	(3)
# q	long long 		integer 		8 	(2), (3)
# Q	unsigned long long 	integer 		8 	(2), (3)
# f	float 			float 			4 	(4)
# d	double 			float 			8 	(4)
# s	char[] 			string
# p	char[] 			string
# P	void * 			integer 		  	(5), (3)

# http://www.nthelp.com/icmp.html
def get_icmp_errcode(type):
  # Type	Name					Reference
  # ----	-------------------------		---------
  errorTypes = dict(
    [
      (0,'Echo Reply'),					# [RFC792]
      (1,'Unassigned'),					# [JBP]
      (2,'Unassigned'),					# [JBP]
      (3,'Destination Unreachable'),			# [RFC792]
      (4,'Source Quench'),				  	# [RFC792]
      (5,'Redirect'),					# [RFC792]
      (6,'Alternate Host Address'),				# [JBP]
      (7,'Unassigned'),					# [JBP]
      (8,'Echo'),					  	# [RFC792]
      (9,'Router Advertisement'),			  	# [RFC1256]
      (10,'Router Selection'),				# [RFC1256]
      (11,'Time Exceeded'),				 	# [RFC792]
      (12,'Parameter Problem'),				# [RFC792]
      (13,'Timestamp'),					# [RFC792]
      (14,'Timestamp Reply'),				# [RFC792]
      (15,'Information Request'),			   	# [RFC792]
      (16,'Information Reply'),				# [RFC792]
      (17,'Address Mask Request'),                           # [RFC950]
      (18,'Address Mask Reply'),				# [RFC950]
      (19,'Reserved (for Security)'),			# [Solo]
      (20,'Reserved (for Robustness Experiment)'),           # [ZSu]
      (29,'Reserved (for Robustness Experiment)'),           # [ZSu]
      (30,'Traceroute'),					# [RFC1393]
      (31,'Datagram Conversion Error'),			# [RFC1475]
      (32,'Mobile Host Redirect'),       	                # [David Johnson]
      (33,'IPv6 Where-Are-You'),         	                # [Bill Simpson]
      (34,'IPv6 I-Am-Here'),             	                # [Bill Simpson]
      (35,'Mobile Registration Request'),	                # [Bill Simpson]
      (36,'Mobile Registration Reply'),  	                # [Bill Simpson]
      (37,'Domain Name Request'),        	                # [Simpson]
      (38,'Domain Name Reply'),          	                # [Simpson]
      (39,'SKIP'),                       	                # [Markson]
      (40,'Photuris'),                   	                # [Simpson]
      #(41-'255 Reserved'),				  	# [JBP]
      (41,'Reserved'),				  	# [JBP]
      ]
    )
  return errorTypes[type]

# src: http://codeselfstudy.com/blogs/how-to-calculate-standard-deviation-in-python
# src: http://serverfault.com/questions/333116/what-does-mdev-mean-in-ping8
def stddev(values):
  from math import sqrt
  # src: http://codeselfstudy.com/blogs/how-to-calculate-standard-deviation-in-python
  # src: http://serverfault.com/questions/333116/what-does-mdev-mean-in-ping8
  mean = ( sum(values) / len(values) ) # avg(values)
  deviations = [ x - mean for x in values ]
  deviations_sq = [ d ** 2 for d in deviations ]
  sum_dev_sq = sum(deviations_sq)
  variance = sum_dev_sq / len(values)
  std_dev = sqrt(variance)
  return std_dev

# test for stats calc - simply call this from anywhere in the code
def test_stddev():
  time_list = [ 9.70 , 10.1 , 10.2 , 17.9 ]
  time_list_ns = [ x / 1000 for x in time_list ]
  reference = \
    '''
    # --- 24.155.92.81 ping statistics ---
    # 4 packets transmitted, 4 received, 0% packet loss, time 3003ms
    # rtt min/avg/max/mdev = 9.700/12.009/17.952/3.437 ms
    '''
  print("reference:\n" + reference)
  std_dev = stddev(time_list)
  print("mdev: %0.3f" % std_dev)

def checksum(string): 
  csum = 0
  countTo = (len(string) // 2) * 2
  count = 0
  while count < countTo:
    # python 2/3 compat - 3 doesn't need 'ord'
    try:
      thisVal = ord(string[count + 1])*256 + ord(string[count])
    except TypeError:
      thisVal = string[count + 1] * 256 + string[count]
    csum = csum + thisVal 
    csum = csum & 0xffffffff  
    count = count + 2
  
  if countTo < len(string):
    csum = csum + ord(string[len(string) - 1])
    csum = csum & 0xffffffff 
  
  csum = (csum >> 16) + (csum & 0xffff)
  csum = csum + (csum >> 16)
  answer = ~csum 
  answer = answer & 0xffff 
  answer = answer >> 8 | (answer << 8 & 0xff00)
  return answer
  
def receiveOnePing(mySocket, ID, timeout, destAddr):
  timeLeft = timeout
  
  while 1: 
    startedSelect = time.time()
    whatReady = select.select([mySocket], [], [], timeLeft)
    howLongInSelect = (time.time() - startedSelect)
    if whatReady[0] == []: # Timeout
      return "Request timed out. right away"
  
    timeReceived = time.time() 
    recPacket, addr = mySocket.recvfrom(1024)
        
    # first, get the IP header
    ip_header = struct.unpack('!BBHHHBBH4s4s' , recPacket[0:20])
    # get the IHL, length in bits; last 4 bits of first byte
    ip_ihl = ip_header[0] & 0xF
    # ip header total length - IHL * 32bit = IHL * 4byte
    ip_h_len = ip_ihl * 4
    # verify ICMP protocol
    if(ip_header[6] != getprotobyname("icmp")):
      # src: http://stackoverflow.com/a/37005235
      table = {num:name[8:] for name,num in vars(socket).items() if name.startswith("IPPROTO")}
      return("non icmp packet received: " + table[ip_h_len[6]])



    #Fetch the ICMP header from the IP packet
    # https://en.wikipedia.org/wiki/Internet_Control_Message_Protocol
    # All ICMP packets have an 8-byte header: 'type code checksum packetID sequence'
    icmp_h_end=ip_h_len+8
    icmpHeaderPacket = recPacket[ip_h_len:icmp_h_end]
    icmp_header = ICMP_H._make( struct.unpack(ICMP_STRUCT_FORMAT, icmpHeaderPacket) )

    # TODO handle different messages and whatever 
    #  https://en.wikipedia.org/wiki/Internet_Control_Message_Protocol#Control_messages

    print("-I- Rx packet: header %s " % (str(icmp_header), ) )
    print("-I- Rx packet: code:message %d:%s " % (icmp_header.code,get_icmp_errcode(icmp_header.code), ) )
    # for pinging localhost, ignore echo request
    if icmp_header.type != ICMP_ECHO_REQUEST and icmp_header.packetID == ID:
        bytesInDouble = struct.calcsize("d")
        icmp_data_end = icmp_h_end + bytesInDouble
        timeSent = struct.unpack("d", recPacket[icmp_h_end:icmp_data_end])[0]
        return timeReceived - timeSent
    timeLeft = timeLeft - howLongInSelect
    if timeLeft <= 0:
      return "Request timed out. later"
  
def sendOnePing(mySocket, destAddr, ID):
  # Header is type (8), code (8), checksum (16), id (16), sequence (16)
  
  myChecksum = 0
  # Make a dummy header with a 0 checksum
  # struct -- Interpret strings as packed binary data
  header = struct.pack(ICMP_STRUCT_FORMAT, ICMP_ECHO_REQUEST, 0, myChecksum, ID, 1)
  curtime = time.time()
  data = struct.pack("d", curtime)
  # Calculate the checksum on the data and the dummy header.
  myChecksum = checksum(header + data)
  
  # Get the right checksum, and put in the header
  if sys.platform == 'darwin':
    # Convert 16-bit integers from host to network  byte order
    myChecksum = htons(myChecksum) & 0xffff    
  else:
    myChecksum = htons(myChecksum)
    
  icmp_header = ICMP_H(ICMP_ECHO_REQUEST, 0, myChecksum, ID, 1)
  header = struct.pack(ICMP_STRUCT_FORMAT, *icmp_header)
  packet = header + data
  
  print("-I- Tx packet: header %s | data %d " % (str(icmp_header), curtime) )
  mySocket.sendto(packet, (destAddr, 1)) # AF_INET address must be tuple, not str
  # Both LISTS and TUPLES consist of a number of objects
  # which can be referenced by their position number within the object.
  
def doOnePing(destAddr, timeout): 
  icmp = getprotobyname("icmp")
  # SOCK_RAW is a powerful socket type. For more details:   
#    http://sock-raw.org/papers/sock_raw

  try:
    mySocket = socket(AF_INET, SOCK_RAW, icmp)
  #except socket.error as e_socket:
  except OSError as e_socket:
    print("-E-: no socket established, try running with sudo")
    raise
  
  myID = os.getpid() & 0xFFFF  # Return the current process i
  sendOnePing(mySocket, destAddr, myID)
  delay = receiveOnePing(mySocket, myID, timeout, destAddr)
  
  mySocket.close()
  return delay
  
def ping(host, timeout=1):
  # timeout=1 means: If one second goes by without a reply from the server,
  # the client assumes that either the client's ping or the server's pong is lost
  dest = gethostbyname(host)
  #while 1 :
  print("""
  TODO print like this:
  PING localhost (127.0.0.1) 56(84) bytes of data.
  64 bytes from localhost (127.0.0.1): icmp_seq=1 ttl=64 time=0.063 ms
  """)

  print("Pinging " + host + " == " + dest + " using Python:")
  # Send ping requests to a server separated by approximately one second
  time_list = []
  start_time = time.time()
  for i in range(4):
    delay = doOnePing(dest, timeout)
    # ICMP is in ms
    # src: https://en.wikipedia.org/wiki/Internet_Control_Message_Protocol#Timestamp
    print("reply in %0.3f ms" % (delay * 1000) )
    time_list.append(delay)
    #print(delay * 1000)
    time.sleep(1)# one second
  total_time = time.time() - start_time
  total_time *= 1000 # convert to ms

  # calculate stats
  packets_tx = len(time_list)
  packets_rx = -1 # TODO
  packet_loss = -1 # TODO
  print("%d packets transmitted, %d received, %d packet loss, time %0.0fms" % (packets_tx, packets_rx, packet_loss, total_time) )
  t_min = 1000 * min(time_list)
  t_avg_ns = ( sum(time_list) / len(time_list) ) # avg(time_list)
  t_avg = 1000 * t_avg_ns
  t_max = 1000 * max(time_list)
  std_dev = stddev(time_list)
  t_mdev = 1000 * std_dev #-1 # todo # mdev(time_list)
  print("rtt min/avg/max/mdev = %0.3f/%0.3f/%0.3f/%0.3f ms" % (t_min , t_avg , t_max , t_mdev ,) )

  return delay
  
#TODO: ping("foobar.com.none")
ping("google.com")
ping("localhost")
ping("bbc.co.uk")



print(
'''
        TODO:
            ping -c 5 google.com
            PING google.com (24.155.92.84) 56(84) bytes of data.
            64 bytes from google-24-155-92-84.grandecom.net (24.155.92.84): icmp_seq=1 ttl=60 time=10.3 ms
            64 bytes from google-24-155-92-84.grandecom.net (24.155.92.84): icmp_seq=2 ttl=60 time=10.1 ms
            64 bytes from google-24-155-92-84.grandecom.net (24.155.92.84): icmp_seq=3 ttl=60 time=10.5 ms
            64 bytes from google-24-155-92-84.grandecom.net (24.155.92.84): icmp_seq=4 ttl=60 time=10.3 ms
            64 bytes from google-24-155-92-84.grandecom.net (24.155.92.84): icmp_seq=5 ttl=60 time=11.1 ms

            --- google.com ping statistics ---
            5 packets transmitted, 5 received, 0% packet loss, time 4006ms
            rtt min/avg/max/mdev = 10.171/10.528/11.132/0.341 ms
'''
)

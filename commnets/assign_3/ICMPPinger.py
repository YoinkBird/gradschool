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
      return "Request timed out."
  
    timeReceived = time.time() 
    recPacket, addr = mySocket.recvfrom(1024)
         
         #Fill in start
        
          #Fetch the ICMP header from the IP packet
        
         #Fill in end
    timeLeft = timeLeft - howLongInSelect
    if timeLeft <= 0:
      return "Request timed out."
  
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
  print("Pinging " + host + " == " + dest + " using Python:")
  print("")
  # Send ping requests to a server separated by approximately one second
  #while 1 :
  print("""
  TODO print like this:
  PING localhost (127.0.0.1) 56(84) bytes of data.
  64 bytes from localhost (127.0.0.1): icmp_seq=1 ttl=64 time=0.063 ms
  """)

  for i in range(4):
    delay = doOnePing(dest, timeout)
    # ICMP is in ms
    # src: https://en.wikipedia.org/wiki/Internet_Control_Message_Protocol#Timestamp
    print("reply in %0.3f ms" % (delay * 1000) )
    #print(delay * 1000)
    time.sleep(1)# one second
  return delay
  
#TODO: ping("foobar.com.none")
ping("google.com")
ping("localhost")

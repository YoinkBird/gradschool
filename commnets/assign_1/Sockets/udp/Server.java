package udp;
import java.io.*; 
import java.net.*; 

/**
 * Example of a Server using UDP. The server
 * accepts a client and serves the client. 
 * The service being provided to 
 * clients is a simple capitalization of string.
 * Each client is serviced only once (one string conversion)
 * Run Server as:
 * java udp.Server <server_port>
 * where server_port is the port at which the server is to be run
 *  
 * @author rameshyerraballi
 * 
 */
class Server { 
  public static void main(String args[]) throws Exception 
    { 
      int ServerPort = java.lang.Integer.parseInt(args[0]);
      //      DatagramSocket serverSocket = new DatagramSocket(9876); 
      DatagramSocket serverSocket = new DatagramSocket(ServerPort); 
      byte[] receiveData = new byte[1024]; 
      byte[] sendData  = new byte[1024]; 
  
      while(true) 
        { 
  
          DatagramPacket receivePacket = 
             new DatagramPacket(receiveData, receiveData.length); 
  
          serverSocket.receive(receivePacket); 
  
          String sentence = new String(receivePacket.getData()); 
          System.out.println("Received form UDP Client: " + sentence);
          InetAddress ClientIPAddress = receivePacket.getAddress(); 
  
          int ClientPort = receivePacket.getPort(); 
  
                      String capitalizedSentence = sentence.toUpperCase(); 

          sendData = capitalizedSentence.getBytes(); 
  
          DatagramPacket sendPacket = 
             new DatagramPacket(sendData, sendData.length, ClientIPAddress, 
                               ClientPort); 
  
          serverSocket.send(sendPacket); 
        } 
    } 
} 


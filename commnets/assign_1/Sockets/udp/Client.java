package udp;

import java.io.*; 
import java.net.*; 
/**
 * Example of a UDP Client that sends a string to
 * a server, for the server to convert to upper-case. Reads 
 * the string returned by the server and displays it on the
 * screen
 * Run Client as
 * java udp.Client <server_host> <server_port>
 * where server_host is the host ip of the server
 * and server_port is the port at which the server is running
 * @author rameshyerraballi
 *
 */
class Client { 
	public static void main(String args[]) throws Exception 
    { 
  
		BufferedReader inFromUser = 
			new BufferedReader(new InputStreamReader(System.in)); 
  
		DatagramSocket clientSocket = new DatagramSocket(); 
		
		InetAddress ServerIPAddress = InetAddress.getByName(args[0]);
		int ServerPort = java.lang.Integer.parseInt(args[1]);
		//      InetAddress ServerIPAddress = InetAddress.getByName("127.0.0.1"); 

		byte[] sendData = new byte[1024]; 
		byte[] receiveData = new byte[1024]; 
		
		String sentence = inFromUser.readLine(); 
      
		sendData = sentence.getBytes(); 
		DatagramPacket sendPacket = 
			new DatagramPacket(sendData, sendData.length, ServerIPAddress, ServerPort);  
		//         new DatagramPacket(sendData, sendData.length, ServerIPAddress, 9876); 
		clientSocket.send(sendPacket); 
  
		DatagramPacket receivePacket = 
			new DatagramPacket(receiveData, receiveData.length); 
  
		clientSocket.receive(receivePacket); 
  
		String modifiedSentence = 
			new String(receivePacket.getData()); 
  
		System.out.println("FROM SERVER:" + modifiedSentence); 

		clientSocket.close(); 
  
    } 
} 


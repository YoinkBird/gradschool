package tcp;

import java.io.*; 
import java.net.*;
import java.util.*;

/**
 *
 * Example of a Server using TCP. The server
 * accepts a client and serves the client. 
 * The service being provided to 
 * clients is a simple capitalization of string.
 * Each client is serviced only once (one string conversion)
 * Run Server as:
 * java tcp.Server <server_port>
 * where server_port is the port at which the server is to be run
 *
 * @author rameshyerraballi
 *
 */
public class Server {

	/**
	 * @param args args[0] is the port number at which the server must be run
	 */
	public static void main(String[] args) throws Exception{
		// TODO Auto-generated method stub
	     String clientSentence; 
	     String capitalizedSentence; 

	      //ServerSocket welcomeSocket = new ServerSocket(80); 
	     ServerSocket welcomeSocket = new ServerSocket(Integer.parseInt(args[0])); 
	  
	     while(true) { 
	  
	    	 Socket connectionSocket = welcomeSocket.accept(); 
	         BufferedReader inFromClient = 
	        	 new BufferedReader(new InputStreamReader(connectionSocket.getInputStream())); 

	         DataOutputStream  outToClient = 
	        	 new DataOutputStream(connectionSocket.getOutputStream()); 

	         clientSentence = inFromClient.readLine(); 
	         System.out.println("Received from Client: " + clientSentence);
	         capitalizedSentence = clientSentence.toUpperCase() + '\n';

	         outToClient.writeBytes(capitalizedSentence); 
	     } 
	}

}

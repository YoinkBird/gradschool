package tcp;

import java.io.*; 
import java.net.*;

/**
 *
 * Example of a TCP Client that sends a string to
 * a server, for the server to convert to upper-case. Reads 
 * the string returned by the server and displays it on the
 * screen
 * Run Client as
 * java tcp.Client <server_host> <server_port>
 * where server_host is the host ip of the server
 * and server_port is the port at which the server is running
 * @author rameshyerraballi
 *
 */
public class Client {

	/**
	 * @param args args[0] is the server's host ip and args[1] is its port number
	 * 
	 */
	public static void main(String[] args) throws Exception{
		// TODO Auto-generated method stub
		String sentence; 
	    String modifiedSentence; 
	    Socket clientSocket = null;
	    BufferedReader inFromUser = 
	          new BufferedReader(new InputStreamReader(System.in)); 

	    String ServerHostname = args[0];
	    int ServerPort = java.lang.Integer.parseInt(args[1]);
	    try {
	    	clientSocket = new Socket(ServerHostname, ServerPort);
	    } catch ( Exception e) {
		
	    } // end of try-catch
	      
	    //	        Socket clientSocket = new Socket("data.uta.edu", 6789); 

	    DataOutputStream outToServer = 
	    	new DataOutputStream(clientSocket.getOutputStream()); 
	    
	    BufferedReader inFromServer = 
	    	new BufferedReader(new InputStreamReader(clientSocket.getInputStream())); 
	    
	    sentence = inFromUser.readLine(); 

	    outToServer.writeBytes(sentence + '\n'); 

	    modifiedSentence = inFromServer.readLine(); 

	    System.out.println("FROM SERVER: " + modifiedSentence); 

	    clientSocket.close(); 
		
		}

}

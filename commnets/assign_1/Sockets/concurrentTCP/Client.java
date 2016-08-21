package concurrentTCP;

import java.io.*; 
import java.net.*; 
/**
 * Example of a TCP Client that continuosly sends strings to
 * a server, for the server to convert to upper-case. Reads 
 * the string returned by the server and displays it on the
 * screen
 * Run Client as
 * java concurrentTCP.Client <server_host> <server_port>
 * where server_host is the host ip of the server
 * and server_port is the port at which the server is running
 * 
 * @author rameshyerraballi
 *  
 */

class Client { 

	public static void main(String args[]) throws Exception 
	{ 
		String sentence; 
		String modifiedSentence; 
		if (args.length != 2) {
			System.out.println("Run Program as\n \t java  concurrentTCP.Client <Server_hostIP> <Server_Port>");
			System.exit(-1);
		}
		BufferedReader inFromUser = 
			new BufferedReader(new InputStreamReader(System.in)); 

		Socket clientSocket = new Socket(args[0], java.lang.Integer.parseInt(args[1])); 

		DataOutputStream outToServer = 
			new DataOutputStream(clientSocket.getOutputStream()); 

		BufferedReader inFromServer = 
			new BufferedReader(new InputStreamReader(clientSocket.getInputStream())); 

		while ((sentence = inFromUser.readLine()) != null) { 
			outToServer.writeBytes(sentence + '\n'); 
   
			modifiedSentence = inFromServer.readLine(); 
			
			System.out.println("FROM SERVER: " + modifiedSentence); 
		}
		
	} 
}
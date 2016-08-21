package concurrentTCP;
import java.io.*; 
import java.net.*; 

/**
 * 
 * Example of a Concurrent Server using TCP. The server
 * accepts a client and serves the client in a separate 
 * thread, allowing it to continue to accept other clients
 * and serve them likewise. The service being provided to 
 * clients is a simple capitalization of string.
 * Run Server as:
 * java concurrentTCP.Server <server_port>
 * where server_port is the port at which the server is to be run 
 * 
 * @author rameshyerraballi
 * 
 */
class Server { 
	/**
	 * @param args args[0] is the port number at which the server must be run
	 */
	public static void main(String args[]) throws Exception 
	{ 

		if (args.length != 1) {
			System.out.println("Run Program as\n \t java ConTCPServer <server_port>");
			System.exit(-1);
		}
		ServerSocket welcomeSocket =
			new ServerSocket(java.lang.Integer.parseInt(args[0])); 

		while(true) { 
			Socket connectionSocket = welcomeSocket.accept(); 
			Servant newServant = new Servant(connectionSocket);
			// try
			// BetterServant newServant = new BetterServant(connectionSocket);
		} 
	} 
} 

class Servant extends Thread
{
	private String clientSentence; 
	private String capitalizedSentence; 
	private Socket SocketToClient;

	public Servant (Socket sock)
	{
		SocketToClient = sock;
		start();
	}

	public void run()
	{
		try {
			BufferedReader inFromClient = 
				new BufferedReader(new  
						InputStreamReader(SocketToClient.getInputStream())); 
	
			DataOutputStream  outToClient = 
				new DataOutputStream(SocketToClient.getOutputStream());
			
			while ((clientSentence = inFromClient.readLine()) != null) {
				System.out.println("From Client on IP: " + SocketToClient.getInetAddress() 
						+ " @port: " + SocketToClient.getPort() + " :\n\t" + clientSentence);
				capitalizedSentence = clientSentence.toUpperCase() + '\n'; 
	    
				outToClient.writeBytes(capitalizedSentence); 
			}
		}
		catch (IOException e) {
			System.out.println("Socket problems");
		}
	}
}
   
class BetterServant implements Runnable 
{
	private String clientSentence; 
	private String capitalizedSentence; 
	private Socket SocketToClient;
	Thread myThread;

	public BetterServant (Socket sock)
	{
		SocketToClient = sock;
		myThread = new Thread(this);
		myThread.start();
	}
	public void run()
	{
		try {
			BufferedReader inFromClient = 
				new BufferedReader(new InputStreamReader(SocketToClient.getInputStream())); 
	
			DataOutputStream  outToClient = 
				new DataOutputStream(SocketToClient.getOutputStream());
			while ((clientSentence = inFromClient.readLine()) != null) {
				capitalizedSentence = clientSentence.toUpperCase() + '\n'; 	    
				outToClient.writeBytes(capitalizedSentence); 
			}
		}
		catch (IOException e) {
			System.out.println("Socket problems");
		}
	}
}


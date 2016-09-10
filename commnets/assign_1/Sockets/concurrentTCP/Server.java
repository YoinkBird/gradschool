package concurrentTCP;
import java.io.*;
import java.net.*;
import java.util.Hashtable;
import java.util.*;

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
  private String className;
  private Hashtable peerHash;
  private InetAddress ServerIPAddress;
  private int ServerPort;
  private Hashtable<String, String> protocolStrings;
  private ServerSocket tcpSocket;
  private DatagramSocket udpSocket;
  private DataOutputStream outToServer;

  public Server(String[] args){
    this.className = new Throwable().getStackTrace()[0].getClassName();
    // parse args
    if (args.length < 1) {
      System.out.println("[" + this.className + "][-E-]: Usage: java Server <port number>");

      System.exit(1);
    }

    //    only works in 'main'
//    this.ServerIPAddress = InetAddress.getByName(args[0]);
//    InetAddress ServerIPAddress = InetAddress.getByName(args[0]);
    this.ServerPort = java.lang.Integer.parseInt(args[0]);

    // init
    this.peerHash = new Hashtable();
  }
  /**
   * @param args args[0] is the port number at which the server must be run
   */
  public static void main(String args[]) throws Exception 
  { 
    // <init_ritual>
    Server ServerInst = new Server(args);
    // </init_ritual>

    // set up sockets
    ServerInst.initSockets();

    while(true) { 
      Socket connectionSocket = ServerInst.tcpSocket.accept(); 
      Servant newServant = new Servant(connectionSocket);
      // try
      // BetterServant newServant = new BetterServant(connectionSocket);
    } 
  }

  public void initSockets() throws Exception{
    String methodName = new Throwable().getStackTrace()[0].getMethodName();
    String logPreAmble = "[" + className + "][" + methodName + "]";
    System.out.println( logPreAmble + "[-I-]: [INIT(server)|" 
        + "]");
    // TCP
    try {
      this.tcpSocket = new ServerSocket(this.ServerPort);
    } catch ( Exception e) {
    System.out.println( logPreAmble + "[-I-]: [INIT(server)|tcp|" + /*this.ServerHostname +*/ ":" + this.ServerPort + "]");
    // UDP
    this.udpSocket = new DatagramSocket();
    int udpPort = this.udpSocket.getLocalPort();
    System.out.println( logPreAmble + "[-I-]: [INIT(server)|udp|"
        + udpPort
        + "]");
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


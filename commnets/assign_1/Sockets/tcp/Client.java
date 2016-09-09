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
    // class name for debug messages
    String className = new Throwable().getStackTrace()[0].getClassName();
    // TODO Auto-generated method stub
    String sentence;
    String modifiedSentence;

    // TCP
    Socket clientSocket = null;
    // UDP
    //DatagramSocket clientSocket = new DatagramSocket();
    // user input
    BufferedReader inFromUser =
      new BufferedReader(new InputStreamReader(System.in));

    // parse args
    if (args.length != 3) {
      System.out.println("[" + className + "][-E-]: Usage: java Client <host name> <port number> <screen_name>");

      System.exit(1);
    }

    // honestly not quite sure what the difference is
    String ServerHostname = args[0];
    InetAddress ServerIPAddress = InetAddress.getByName(args[0]);
    int ServerPort = java.lang.Integer.parseInt(args[1]);
    
    // UDP - for datagram
    byte[] sendData = new byte[1024];
    byte[] receiveData = new byte[1024];

    String userName = args[2];
    System.out.println("[" + className + "][-I-]: will transmit to " + ServerHostname + ":" + ServerPort);
    try {
      clientSocket = new Socket(ServerHostname, ServerPort);
    } catch ( Exception e) {

    } // end of try-catch
    int myPort = clientSocket.getLocalPort();

    //	        Socket clientSocket = new Socket("data.uta.edu", 6789);

    DataOutputStream outToServer =
      new DataOutputStream(clientSocket.getOutputStream());

    BufferedReader inFromServer =
      new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));

    // parties: [Tx|tcp|client,server]
    // HELO¤<screen_name>¤<IP>¤<Port> \n
    sentence = "HELO " + userName + " " + ServerHostname + " " + myPort + "\n";
    System.out.println("[" + className + "][-I-]: [Tx(server)|tcp|" + ServerHostname + ":" + ServerPort + "|" + sentence + "]");

    outToServer.writeBytes(sentence + '\n');

    modifiedSentence = inFromServer.readLine();

    System.out.println("[" + className + "][-I-]: [Rx(server)|tcp|" + ServerHostname + ":" + ServerPort + "|" + modifiedSentence + "]");

    for(;;){
    }
    /* TODO: udp
    // parties: [Tx|udp|client,clients]
    // MESG¤<screen_name>:¤<message>\n
    sentence = inFromUser.readLine();

    sentence = "MESG " + userName + " " + sentence;
    sentence += "\n";
    System.out.println("[" + className + "][-I-]: [Tx(server)|" + ServerHostname + ":" + ServerPort + "|" + sentence + "]");

    outToServer.writeBytes(sentence + '\n');

    modifiedSentence = inFromServer.readLine();

    System.out.println("[" + className + "][-I-]: [Rx(server)|" + ServerHostname + ":" + ServerPort + "|" + modifiedSentence + "]");
    */

    /*
    // TODO: read from UDP
    // parties: [Tx|tcp|client,server]
    // EXIT\n
    sentence = "EXIT \n";
    System.out.println("[" + className + "][-I-]: [Tx(server)|" + ServerHostname + ":" + ServerPort + "|" + sentence + "]");

    outToServer.writeBytes(sentence + '\n');

    modifiedSentence = inFromServer.readLine();

    System.out.println("[" + className + "][-I-]: [Rx(server)|" + ServerHostname + ":" + ServerPort + "|" + modifiedSentence + "]");

    // done
    clientSocket.close();
    */

  }

}

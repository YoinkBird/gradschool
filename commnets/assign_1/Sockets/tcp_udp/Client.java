package tcp_udp;

import java.io.*;
import java.net.*;
import java.util.Hashtable;
import java.util.*;

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
  private ArrayList<String[]> peerList;// = new ArrayList<String[]>();

  /**
   * @param args args[0] is the server's host ip and args[1] is its port number
   *
   */
  public static void main(String[] args) throws Exception{
    // class name for debug messages
    String className = new Throwable().getStackTrace()[0].getClassName();
    Client thisClient = new Client();
    // TODO Auto-generated method stub
    String sentence;
    String modifiedSentence;

    // TCP
    Socket tcpSocket = null;
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
    // screen_name
    String userName = args[2];
    
    // UDP - for datagram
    byte[] sendData = new byte[1024];
    byte[] receiveData = new byte[1024];

    System.out.println("[" + className + "][-I-]: will transmit to " + ServerHostname + ":" + ServerPort);
    try {
      tcpSocket = new Socket(ServerHostname, ServerPort);
    } catch ( Exception e) {

    } // end of try-catch
    int myPort = tcpSocket.getLocalPort();
    // UDP
    DatagramSocket udpSocket = new DatagramSocket();
    int udpPort = udpSocket.getLocalPort();

    //	        Socket tcpSocket = new Socket("data.uta.edu", 6789);

    DataOutputStream outToServer =
      new DataOutputStream(tcpSocket.getOutputStream());

    BufferedReader inFromServer =
      new BufferedReader(new InputStreamReader(tcpSocket.getInputStream()));

    // store the deterministic components of the protocol strings, e.g. username
    // when calling, add the dynamic components
    Hashtable<String, String> protocolStrings = new Hashtable<>();
    // HELO¤<screen_name>¤<IP>¤<Port>\n
    protocolStrings.put("HELO", "HELO " + userName + " " + ServerHostname + " " + udpPort);
    // RJCT¤<screen_name>\n
    protocolStrings.put("RJCT", "RJCT " + userName);
    // MESG¤<screen_name>:¤<message>\n
    protocolStrings.put("MESG", "MESG " + userName);
    // ACPT¤<SNn>¤<IPn>¤<PORTn>
    protocolStrings.put("ACPT", "ACPT");
    // EXIT\n
    protocolStrings.put("EXIT", "EXIT");

    // parties: [Tx|tcp|client,server]
    // HELO¤<screen_name>¤<IP>¤<Port> \n
    sentence = protocolStrings.get("HELO");
    System.out.println("[" + className + "][-I-]: [Tx(server)|" + ServerHostname + ":" + ServerPort + "|" + sentence + "]");

    outToServer.writeBytes(sentence + '\n');

    modifiedSentence = inFromServer.readLine();

    System.out.println("[" + className + "][-I-]: [Rx(server)|" + ServerHostname + ":" + ServerPort + "|" + modifiedSentence + "]");

    /*
    The server sends this message in response to the Greeting, to let the Chat Client know that the screen name is already in use.
    // parties: [Rx|tcp|server,client]
    RJCT¤<screen_name>\n
    */
    if ( modifiedSentence.equals(protocolStrings.get("RJCT")) ){
      System.out.println("bad username, exiting");
      System.exit(2);
    }

    /*
    The server sends this message in response to the Greeting, to acknowledge the validity of the screen name
     and to inform the Chatter Client of the Identities of the ALL Chatters (including yourself).
    Each identity is separated by a “:”.
    // parties: [Rx|tcp|server,client]
    ACPT¤<SNn>¤<IPn>¤<PORTn>\n
    */
    if (! modifiedSentence.startsWith(protocolStrings.get("ACPT")) ){
      System.out.println("bad ACPT response, exiting");
      System.exit(2);
    }
    // parse reply
    thisClient.parseAccept(modifiedSentence);
    System.out.println("-D-: printing out user array");
    // print out table/retrieve element/whatever
    {
      for (String[] peerDataArr : thisClient.getPeerList()){
        for (String data : peerDataArr){
          System.out.print(data + "|");
        }
        System.out.println();
      }
    }
    // UDP section
    {
      // process response
      // TODO: remove hard-code

      /*
      This message is sent to the UDP ports of all members in the membership list, when a chat user types in a message (in the JTextField).
      */
      // parties: [Tx|udp|client,clients]
      // MESG¤<screen_name>:¤<message>\n
      // TODO: remove hard-code
      //sentence = inFromUser.readLine();
      sentence = "where is sauce";

      sentence = protocolStrings.get("MESG") + " " + sentence;
      sentence += "\n";
      sendData = sentence.getBytes();
      // print out table/retrieve element/whatever
      {
        for (String[] peerDataArr : thisClient.getPeerList()){
          String todoSN = peerDataArr[0];
          InetAddress todoIP = InetAddress.getByName(peerDataArr[1]);
          int todoPort = java.lang.Integer.parseInt(peerDataArr[2]);

          System.out.println("[" + className + "][-I-]: [Tx(peer)|udp|" + todoSN + "|" + todoIP + ":" + todoPort + "|" + sentence + "]");

          DatagramPacket sendPacket =
            new DatagramPacket(sendData, sendData.length, todoIP, todoPort);
          System.out.println("[" + className + "][-I-]: UDP packet created");
          udpSocket.send(sendPacket);

          System.out.println("[" + className + "][-I-]: UDP packet sent");
        }
      }


//      udpSocket.close();

    // TODO: for now, loop until receive new message
    boolean udpReceived = false;
//    while(! udpReceived){
    for(;;){
      /*
         This is a message received on the UDP Socket.
         It is another Chatter’s chat message.
         Parse it and display the message in the JtextArea as shown in the GUI
      // parties: [Rx|udp|client,clients]
      MESG¤<screen_name>:¤<message>\n
      */
      DatagramPacket receivePacket =
        new DatagramPacket(receiveData, receiveData.length);

      System.out.println("[" + className + "][-I-]: waiting for reply on " + udpSocket.getLocalPort());
      try{
      udpSocket.receive(receivePacket);
      }
      catch (IOException localIOException) {}
      //udpSocket.close();
      System.out.println("[" + className + "][-I-]: received reply " + udpSocket.getLocalPort());

      String response =
        new String(receivePacket.getData());

      //System.out.println("[" + className + "][-I-]: [Rx(peer)|udp|" + todoIP + ":" + todoPort + "|" + response + "]");
      System.out.println("[" + className + "][-I-]: [Rx(peer)|udp|" + ServerHostname + ":" + ServerPort + "|" + response + "]");
    }


  }

  // parse ACPT reply
  public ArrayList<String[]> parseAccept(String response) {
    ArrayList<String[]> peerArr = new ArrayList<String[]>();
    //System.out.println("-D-: parsing ACPT reply");
    // reply format: ACPT¤<SNn>¤<IPn>¤<PORTn>:<SNn+1>¤<IPn+1>¤<PORTn+1>:
    // extract,remove keyword
    // then split on ':'
    // then split on ' '
    String type = response.substring(0,4);
    String sequence = response.substring(5,response.length()-1);
    //System.out.println("-D-:" + type + "|" + sequence);
    //System.out.println("-D-: ACPT List");
    String[] replyArr = sequence.split("[:]");
    for (String iter: replyArr) {
      //System.out.println(iter);
      String[] peerInfo = iter.split("[\\s]");
      peerArr.add(peerInfo);
    }
    this.peerList = peerArr;
    return peerArr;
  }

  public ArrayList<String[]> getPeerList(){
    // TODO: remove self
    return this.peerList;
  }

}

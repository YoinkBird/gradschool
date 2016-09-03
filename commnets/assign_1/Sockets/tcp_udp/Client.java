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
  private String className;
  private String userName;
  private ArrayList<String[]> peerList;// = new ArrayList<String[]>();
  private String ServerHostname;
  private InetAddress ServerIPAddress;
  private int ServerPort;

  public Client(String[] args){
    this.className = new Throwable().getStackTrace()[0].getClassName();
    // parse args
    if (args.length != 3) {
      System.out.println("[" + this.className + "][-E-]: Usage: java Client <host name> <port number> <screen_name>");

      System.exit(1);
    }

    // honestly not quite sure what the difference is
    this.ServerHostname = args[0];
    //    only works in 'main'
//    this.ServerIPAddress = InetAddress.getByName(args[0]);
//    InetAddress ServerIPAddress = InetAddress.getByName(args[0]);
    this.ServerPort = java.lang.Integer.parseInt(args[1]);
    // screen_name
    this.userName = args[2];
  }

  /**
   * @param args args[0] is the server's host ip and args[1] is its port number
   *
   */
  public static void main(String[] args) throws Exception{
    // <init ritual>
    Client thisClient = new Client(args);
    // cannot do this in the constructor for some reason
    thisClient.ServerIPAddress = InetAddress.getByName(args[0]);
    // </init ritual>
    // TODO Auto-generated method stub

    thisClient.connectToServer();
  }

  private void connectToServer() throws Exception{
    String className = this.className;
    String sentence;
    String modifiedSentence;
    // TCP
    Socket tcpSocket = null;
    // user input
    BufferedReader inFromUser =
      new BufferedReader(new InputStreamReader(System.in));

    
    System.out.println("[" + className + "][-I-]: will transmit to " + this.ServerHostname + ":" + this.ServerPort);
    try {
      tcpSocket = new Socket(this.ServerHostname, this.ServerPort);
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

    // src: http://stackoverflow.com/a/31550047
    // store the deterministic components of the protocol strings, e.g. username
    // when calling, add the dynamic components
    Hashtable<String, String> protocolStrings = new Hashtable<>();
    // HELO¤<screen_name>¤<IP>¤<Port>\n
    protocolStrings.put("HELO", "HELO " + this.userName + " " + this.ServerHostname + " " + udpPort);
    // RJCT¤<screen_name>\n
    protocolStrings.put("RJCT", "RJCT " + this.userName);
    // MESG¤<screen_name>:¤<message>\n
    protocolStrings.put("MESG", "MESG " + this.userName);
    // ACPT¤<SNn>¤<IPn>¤<PORTn>
    protocolStrings.put("ACPT", "ACPT");
    // EXIT\n
    protocolStrings.put("EXIT", "EXIT");

    // parties: [Tx|tcp|client,server]
    // HELO¤<screen_name>¤<IP>¤<Port> \n
    sentence = protocolStrings.get("HELO");
    System.out.println("[" + className + "][-I-]: [Tx(server)|" + this.ServerHostname + ":" + this.ServerPort + "|" + sentence + "]");

    outToServer.writeBytes(sentence + '\n');

    modifiedSentence = inFromServer.readLine();

    System.out.println("[" + className + "][-I-]: [Rx(server)|" + this.ServerHostname + ":" + this.ServerPort + "|" + modifiedSentence + "]");

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
    this.parseAccept(modifiedSentence);
    System.out.println("-D-: printing out user array");
    // print out table/retrieve element/whatever
    {
      for (String[] peerDataArr : this.getPeerList()){
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
      // add timestamp for easierdebug
      // http://stackoverflow.com/a/6953926
      //String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(Calendar.getInstance().getTime());
      //sentence = "[" + timeStamp + "]";
      sentence = "where is sauce";

      sentence = protocolStrings.get("MESG") + " " + sentence;
      sentence += "\n";
      byte[] sendData = new byte[1024];
      sendData = sentence.getBytes();

      // print out table/retrieve element/whatever
      {
        for (String[] peerDataArr : this.getPeerList()){
          String todoSN = peerDataArr[0];
          InetAddress todoIP = InetAddress.getByName(peerDataArr[1]);
          int todoPort = java.lang.Integer.parseInt(peerDataArr[2]);

          System.out.println("[" + className + "][-I-]: [Tx(peer)|udp|" + todoSN + "|" + todoIP + ":" + todoPort + "|" + sentence + "]");

          DatagramPacket sendPacket =
            new DatagramPacket(sendData, sendData.length, todoIP, todoPort);
          //         new DatagramPacket(sendData, sendData.length, todoIP, 9876);
          System.out.println("[" + className + "][-I-]: UDP packet created");
          udpSocket.send(sendPacket);

          System.out.println("[" + className + "][-I-]: UDP packet sent");
        }
      }
    }

    // TODO: for now, loop until receive new message
    boolean udpReceived = false;
    while(! udpReceived){
      /*
         This is a message received on the UDP Socket.
         It is another Chatter’s chat message.
         Parse it and display the message in the JtextArea as shown in the GUI
      // parties: [Rx|udp|client,clients]
      MESG¤<screen_name>:¤<message>\n
      */
      byte[] receiveData = new byte[1024];

      DatagramPacket receivePacket =
        new DatagramPacket(receiveData, receiveData.length);

      System.out.println("[" + className + "][-I-]: waiting for reply on " + udpSocket.getLocalPort());
      try{
      udpSocket.receive(receivePacket);
      }
      catch (IOException localIOException) {}
      System.out.println("[" + className + "][-I-]: received reply " + udpSocket.getLocalPort());

      String response =
        new String(receivePacket.getData());

      //System.out.println("[" + className + "][-I-]: [Rx(peer)|udp|" + todoIP + ":" + todoPort + "|" + response + "]");
      System.out.println("[" + className + "][-I-]: [Rx(peer)|udp|" + this.ServerHostname + ":" + this.ServerPort + "|" + response + "]");
      String[] respArr = this.parseIncoming(response);
      if(respArr[0].equals("MESG")){
        udpReceived = true;
      }
    }



    /*
    When the Chat Client wants to terminate (or exit) the chat it sends this to the Membership Server over TCP.
    The exit should take effect ONLY when the client clicks on the EXIT button provided in the GUI.
    The Client must read a response (see below) back from the server over the UDP Socket and then terminate.
    // parties: [Tx|tcp|client,server]
    EXIT\n
    */
    sentence = "EXIT\n";
    System.out.println("[" + className + "][-I-]: [Tx(server)|" + this.ServerHostname + ":" + this.ServerPort + "|" + sentence + "]");

    outToServer.writeBytes(sentence + '\n');

    System.out.println("[" + className + "][-I-]: [Rx(server)|" + this.ServerHostname + ":" + this.ServerPort + "|" + modifiedSentence + "]");


    // TODO: for now, loop until receive new message
    udpReceived = false;
    while(! udpReceived){
      /*
         This is a message received on the UDP Socket.
         It is from the Membership Server notifying the exit of a member from the chatroom.
         Parse it and display an appropriate message in the JtextArea (Elvis has left the Building); Remove from local list.
         // parties: [Rx|udp|server,client]
         EXIT¤<screen_name>\n
         */
      byte[] receiveData = new byte[1024];
      DatagramPacket receivePacket =
        new DatagramPacket(receiveData, receiveData.length);

      System.out.println("[" + className + "][-I-]: waiting for reply on " + udpSocket.getLocalPort());
      try{
      udpSocket.receive(receivePacket);
      }
      catch (IOException localIOException) {}
      System.out.println("[" + className + "][-I-]: received reply " + udpSocket.getLocalPort());

      String response =
        new String(receivePacket.getData());


      System.out.println("[" + className + "][-I-]: [Rx(peer)|udp|" + this.ServerHostname + ":" + this.ServerPort + "|" + response + "]");
      String[] respArr = this.parseIncoming(response);
      if(respArr[0].equals("EXIT")){
        if(respArr[1].startsWith(this.userName)){
          udpReceived = true;
        }
      }
    }
    // done
    tcpSocket.close();
    udpSocket.close();

  }

  // parse incomming messages
  public String[] parseIncoming(String response) {
    String[] replyArr = new String[2];
    // reply format: <keyword>¤<content>
    // extract,remove keyword
    String type = response.substring(0,4);
    replyArr[0] = type;
    String content = response.substring(5,response.length());
    if(content != null){
      replyArr[1] = content;
    }
    return replyArr;
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
    String sequence = response.substring(5,response.length());
    String[] replyArr1 = this.parseIncoming(response);
    type = replyArr1[0];
    sequence = replyArr1[1];
    //System.out.println("-D-:" + type + "|" + sequence);
    //System.out.println("-D-: ACPT List");
    String[] replyArr = sequence.split("[:]");
    for (String iter: replyArr) {
      //System.out.println(iter);
      String[] peerInfo = iter.split("[\\s]");
      // TODO: remove this hack.
      // sending msg to self removes need to manually start a client and type a reply
      // skip self
      /*
      if(peerInfo[0].equals(this.userName)){
        continue;
      }
      */
      peerArr.add(peerInfo);
      System.out.print("[");
      for ( String val : peerInfo ){
        System.out.print(val);
        System.out.print("|");
      }
      System.out.print("]");
      System.out.println();
    }
    this.peerList = peerArr;
    return peerArr;
  }

  public ArrayList<String[]> getPeerList(){
    // TODO: remove self
    return this.peerList;
  }

}

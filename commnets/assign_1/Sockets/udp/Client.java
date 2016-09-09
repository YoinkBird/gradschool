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
    // class name for debug messages
    String className = new Throwable().getStackTrace()[0].getClassName();

    BufferedReader inFromUser =
      new BufferedReader(new InputStreamReader(System.in));

    DatagramSocket udpSocket = new DatagramSocket();

    // parse args
    if (args.length != 3) {
      System.out.println("[" + className + "][-E-]: Usage: java Client <host name> <port number> <screen_name>");
      System.exit(1);
    }
    //      InetAddress ServerIPAddress = InetAddress.getByName("127.0.0.1");
    InetAddress ServerIPAddress = InetAddress.getByName(args[0]);
    int ServerPort = java.lang.Integer.parseInt(args[1]);
    String userName = args[2];

    System.out.println("[" + className + "][-I-]: will transmit to " + ServerIPAddress + ":" + ServerPort);
    byte[] sendData = new byte[1024];
    byte[] receiveData = new byte[1024];

    String sentence = inFromUser.readLine();

    /* TODO <newsection> */
    // parties: [Tx|udp|client,clients]
    // MESG¤<screen_name>:¤<message>\n
    sentence = "MESG " + userName + " " + sentence;
    sentence += "\n";

    /* TODO </newsection> */

    System.out.println("[" + className + "][-I-]: [Tx(peer)|udp|" + ServerIPAddress + ":" + ServerPort + "|" + sentence + "]");
    sendData = sentence.getBytes();

    DatagramPacket sendPacket =
      new DatagramPacket(sendData, sendData.length, ServerIPAddress, ServerPort);
    //         new DatagramPacket(sendData, sendData.length, ServerIPAddress, 9876);
    System.out.println("[" + className + "][-I-]: UDP packet created");
    udpSocket.send(sendPacket);

    System.out.println("[" + className + "][-I-]: UDP packet sent");

    DatagramPacket receivePacket =
      new DatagramPacket(receiveData, receiveData.length);

    udpSocket.receive(receivePacket);
    System.out.println("[" + className + "][-I-]: UDP waiting for reply");

    String modifiedSentence =
      new String(receivePacket.getData());

    System.out.println("[" + className + "][-I-]: [Rx(peer)|udp|" + ServerIPAddress + ":" + ServerPort + "|" + modifiedSentence + "]");

    udpSocket.close();

  }
}


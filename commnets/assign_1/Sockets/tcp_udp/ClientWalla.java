package tcp_udp;

import tcp_udp.Client.*;
/**
 * Created by weelee on 9/2/16.
 */
public class ClientWalla {
  public static void main(String[] args) throws Exception{
    String className = new Throwable().getStackTrace()[0].getClassName();
    // copy-paste starting here:
    //import tcp_udp.Client.*;
    Client thisClient = new Client(args);
    System.out.println("[" + className + "][-I-]: userName: " + thisClient.getUserName());
    thisClient.connectToServer();
    thisClient.communicateWithServer();
  }
}

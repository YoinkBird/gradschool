package tcp_udp;

import java.util.*;
import tcp_udp.Client.*;
/**
 * Created by weelee on 9/2/16.
 */
public class ClientWalla {
  public static void main(String[] args) throws Exception{
    String className = new Throwable().getStackTrace()[0].getClassName();
    // get message
    String message = "where is sauce";
    if(args.length > 3){
      message = args[3];
    }

    // copy-paste starting here:
    //import tcp_udp.Client.*;
    Client thisClient = new Client(args);
    System.out.println("[" + className + "][-I-]: userName: " + thisClient.getUserName());
    thisClient.connectToServer();
    // TODO: testing hard-code
    thisClient.sendToPeer(message);
    // TODO: for testing, let an exit happen by having another client issue an exit cmd
    boolean exit = false;
    if(message.contains("EXIT:")){
      exit = true;
    }
    for(;;){
      try{
        String response = "EXIT " + thisClient.getUserName();
        if(! exit){
          // get response
          response = thisClient.getUdp();
        }
        String[] respArr = thisClient.parseIncoming(response);
        // figure it out
        if(respArr[0].equals("MESG")){
          // TODO: don't assume defined
          String msg = respArr[1];
          System.out.println(msg);
          // force quit for testing purposes
          if(respArr[1].contains("EXIT:" + thisClient.getUserName())){
            exit = true;
            System.out.println("received EXIT, setting exit flag to:" + exit);
          }

        }
        else if(respArr[0].equals("JOIN")){
          String msg = new String();
          // parse reply
          // TODO: fix the condition in which own username is left out
          ArrayList<String[]> joinArr = thisClient.parseAccept(response);
          for (String[] peerDataArr : joinArr){
            msg += peerDataArr[0] + " ";
          }
          msg += " has joined the chatroom";
          System.out.println("-I-: " + msg);
          System.out.println("-D-: printing out user array");
          thisClient.printPeerList();
        }
        //  EXIT
        else if(respArr[0].equals("EXIT") || exit){
          // only exit if the function confirms it; still depends on username and so forth
          if( thisClient.disconnectFromServerFinalise(response) ){
            System.exit( 0 );
          }else{
            //TODO: remove the newline
            String[] peerInfo = respArr[1].split("[\\s]");
            String goneUser = peerInfo[0];
            String msg = goneUser + " has left the chatroom";
            // TODO: remove user
            thisClient.removePeer(goneUser);
            System.out.println("-D-: removed users, printing out user array");
            thisClient.printPeerList();
            System.out.println("-I-: " + msg);

          }
        }
      } catch ( Exception e1) {
          System.out.println(e1.getMessage());
      }
    }
    /*
    thisClient.receiveFromPeer();
    thisClient.disconnectFromServer();
    */
  }
}

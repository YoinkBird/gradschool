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
    // TODO: testing hard-code
    thisClient.sendToPeer("where is sauce");
    for(;;){
      try{
        // get response
        String response = thisClient.getUdp();
        String[] respArr = thisClient.parseIncoming(response);
        // figure it out
        if(respArr[0].equals("MESG")){
          // TODO: don't assume defined
          String msg = respArr[1];
          //this.WriteData(msg);
          System.out.println(msg);
        }
        else if(respArr[0].equals("JOIN")){
          // parse reply
          thisClient.parseAccept(response);
          System.out.println("-D-: printing out user array");
          thisClient.printPeerList();
        }
        //  EXIT
        else if(respArr[0].equals("EXIT")){
          // only exit if the function confirms it; still depends on username and so forth
          if( thisClient.disconnectFromServerFinalise(response) ){
            System.exit( 0 );
          }
        }
      } catch ( Exception e1) {
      }
    }
    /*
    thisClient.receiveFromPeer();
    thisClient.disconnectFromServer();
    */
  }
}

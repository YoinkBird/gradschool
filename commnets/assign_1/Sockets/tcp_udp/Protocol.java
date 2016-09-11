package tcp_udp;

import java.util.*;

public class Protocol {
  private String className;
  public Hashtable<String, String> protocolStrings;
  public Hashtable userHash;

  public Protocol(){
    String className = new Throwable().getStackTrace()[0].getClassName();
    // store the basic components of the protocol strings
    // when calling, add the dynamic components
    Hashtable<String, String> protocolStrings = new Hashtable<>();
    // HELO¤<screen_name>¤<IP>¤<Port>\n
    protocolStrings.put("HELO", "HELO");// + this.userName + " " + this.ServerHostname + " " + udpPort);
    // RJCT¤<screen_name>\n
    protocolStrings.put("RJCT", "RJCT");// + this.userName);
    // MESG¤<screen_name>:¤<message>\n
    protocolStrings.put("MESG", "MESG");// + this.userName);
    // ACPT¤<SNn>¤<IPn>¤<PORTn>
    protocolStrings.put("ACPT", "ACPT");
    // EXIT\n
    protocolStrings.put("EXIT", "EXIT");
    // TODO: set up a method
    this.protocolStrings = protocolStrings;

    // init
    this.userHash = new Hashtable();
  }

  public static void main(String[] args) throws Exception{
  }

  // parse incoming messages
  public String[] parseIncoming(String response) {
    String[] replyArr = new String[2];
    if(response.length() < 4){
      replyArr = null;
    }
    else{
      // remove newlines
      // response = response.replace("\n", "").replace("\r", "");
      // reply format: <keyword>¤<content>
      // extract,remove keyword
      String type = response.substring(0,4);
      replyArr[0] = type;
      if(response.length() > 4){
        if(response.substring(4,5).equals(" ")) {
          String content = response.substring(5, response.length());
          if (content != null) {
            replyArr[1] = content;
          }
        }
      }
    }
    return replyArr;
  }
  // parse HELO reply
  public Hashtable<String, String[]> parseHelo(String input) {
    // reference
    Hashtable heloHashTmp = new Hashtable();
    // leave possibility to parse entire message
    String sequence = input;
    String[] heloArr = sequence.split("[\\s]");
    heloHashTmp.put("user",heloArr[0]);
    heloHashTmp.put("host",heloArr[1]);
    heloHashTmp.put("port",heloArr[2]);
    return heloHashTmp;
  }
  // parse ACPT reply
  public ArrayList<String[]> parseAccept(String response) {
    ArrayList<String[]> userArr = new ArrayList<String[]>();
    // reference
    Hashtable userHashTmp = new Hashtable();
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
      String[] userInfo = iter.split("[\\s]");
      userArr.add(userInfo);
      userHashTmp.put(userInfo[0],userInfo);
      System.out.print("[");
      for ( String val : userInfo ){
        System.out.print(val);
        System.out.print("|");
      }
      System.out.print("]");
      System.out.println();
    }
    this.userHash.putAll(userHashTmp);
    return userArr;
  }

  // reply format: ACPT¤<SNn>¤<IPn>¤<PORTn>:<SNn+1>¤<IPn+1>¤<PORTn+1>:
  public ArrayList<String[]> getUserList(){
    // TODO: remove self
    ArrayList<String[]> values = new ArrayList<String[]>(this.userHash.values());
    return values;
  }

  // create 'JOIN' string to message other clients about new user
  public String createJoinMsg(Hashtable userInfo){
    String separator = " ";
    String joinMsg = "JOIN" + separator
      + userInfo.get("user") + separator
      + userInfo.get("host") + separator
      + userInfo.get("port") + separator;
    return joinMsg;
  }
  public String createJoinMsg(String userName){
    Hashtable userInfo = (Hashtable) this.userHash.get(userName);
    String separator = " ";
    String joinMsg = "JOIN" + separator
      + userInfo.get("user") + separator
      + userInfo.get("host") + separator
      + userInfo.get("port") + separator;
    return joinMsg;
  }

  public String createAcptMsg(){
    // print out table/retrieve element/whatever
    {
      Enumeration userEnum = this.userHash.keys();
      String separator = " ";
      String acptTxt = "ACPT" + separator;
      while ( userEnum.hasMoreElements() ){
        String user = (String) userEnum.nextElement();
        Hashtable userInfo = (Hashtable) this.userHash.get(user);

        acptTxt += 
            userInfo.get("user") + separator +
            userInfo.get("host") + separator +
            userInfo.get("port") + ":";

      }
      return acptTxt;
    }
  }

  public String createUserList(){
    String userList = "";
    // print out table/retrieve element/whatever
    {
      Enumeration userEnum = this.userHash.keys();
      String separator = " ";
      String recordSeparator = "\n";
      while ( userEnum.hasMoreElements() ){
        String user = (String) userEnum.nextElement();
        Hashtable userInfo = (Hashtable) this.userHash.get(user);

        userList +=
            userInfo.get("user") + separator +
            userInfo.get("host") + separator +
            userInfo.get("port") + recordSeparator;
      }
    }
    return userList;
  }
  public void printUserList(){
    // print out table/retrieve element/whatever
    {
      Enumeration userEnum = this.userHash.keys();
      System.out.println("-I-: user list");
      System.out.println( this.createUserList() );
    }
  }
  public void removeUser(String removeUserName){
    this.userHash.remove(removeUserName);
  }
}

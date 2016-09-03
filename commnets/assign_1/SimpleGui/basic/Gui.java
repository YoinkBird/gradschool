package basic;

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.text.DefaultCaret;

import tcp_udp.*;
/**
 * 
 * @author rameshyerraballi
 * This program demonstrates three basic GUI components in Java, 
 * 		a Button (JButton) to click on, 
 * 		a Text Field (JTextField) to enter text into (usually a single line)
 * and  a Text Area (JTextArea) where you can see multiple lines displayed
 *        (Text Areas may be editable like in an Editor or just readable like in a Browser)
 * For components like Button and Text Fields with which we can interact we can
 * register callbacks, aka Event/Action Listeners that get invoked when the event
 * of interest occurs on them.
 */
public class Gui extends JFrame{
  // Gui extends JFrame which gives us the basic infrastructure of a window
  // with the ability to add other components to  
  private JTextField enter;
  private JTextArea display;
  private JButton qbutton;

  public Client myClient;

  public Gui(Client initClient)
  {
    super( initClient.getUserName() );	// Call JFrame's constructor to set the title of the Window
    this.myClient = initClient;
    Container c = getContentPane(); // To add components to a JFrame we need a handle to its Content Pane

    // Create a new TextField
    enter = new JTextField("Enter Here:");
    // Set its editability to be true; One can toggle this property to temporarily disable  entering of text in it;
    //  Enabled by default though
    enter.setEnabled( true );
    /*
     *   MESG
     */
    // There are several ways to register a Callback on a Gui component, here is the simplest
    //  We are creating a new anonymous object that implements the ActionListener Interface
    //   The ActionListener interface requires that you implement the ActionPerformed method
    //   and hence the code. The ActionPerformed method will be called when any event (like 
    //   entering text in a JTextField and typing enter) occurs on enter. In our implementation
    //   below we call the WriteData method and pass it what the user typed in
    enter.addActionListener(
        new ActionListener() {
          public void actionPerformed( ActionEvent e )
          {
            try{
              initClient.sendToPeer();
            } catch ( Exception e1) {
            }
            WriteData( e.getActionCommand() ); // e.getActionCommand() returns the text typed in
          }
        }
        );
    c.add( enter, BorderLayout.NORTH ); // Place this component at the top (i.e., NORTH)

    display = new JTextArea();	// Create a JTextArea
    // http://stackoverflow.com/a/2483824
    DefaultCaret caret = (DefaultCaret)display.getCaret();
    caret.setUpdatePolicy(DefaultCaret.ALWAYS_UPDATE);
    //
    c.add( new JScrollPane( display ), 
        BorderLayout.CENTER ); // Attach a Scrollpane to it before adding so we can scroll
    /*
     *   EXIT
     */
    // Also add it to the CENTER 
    qbutton = new JButton("QUIT"); // Create a JButton
    qbutton.setEnabled( true );	   // Enable Button so user can click on it
    // Register a Callback; Same as with TextField except we are writing the code in place rather
    // than call another method
    qbutton.addActionListener(
        new ActionListener() {
          public void actionPerformed( ActionEvent e )
          {
            System.out.println("Quit Pressed");
            try{
              // send initial request to exit
              initClient.disconnectFromServerInit();
            } catch ( Exception e1) {
            }
            System.exit(1);
          }
        }
        );
    c.add( qbutton, BorderLayout.SOUTH );
    setSize( 640, 480 ); // Set the size of the JFrame
    this.setVisible(true);	  // Display it
  }
  /**
   * This routine keeps the code running indefinitely
   */
  public void runGui() 
  {
    do {
      try{
        this.myClient.receiveFromPeer();
        if( this.myClient.disconnectFromServerVerify() ){
          System.exit( 0 );
        }
      } catch ( Exception e1) {
      }
    }while (true);
  }
  /**
   * Writes data read from the JTextField (entered by user) into
   * the JTextArea
   * @param s
   */
  private void WriteData( String s )
  {
    display.append( "\nTyped>>>" + s ); // write to the TextArea
    enter.setText(""); //Clear the TextField
  }

  /**
   * @param args
   */
  public static void main(String[] args) throws Exception{
    if (args.length != 1) {
      System.out.println("Must run with one command Line argument as: java Gui <Title>");
//      System.exit(-1);
    }
    Client myClient = new Client(args);
    /*
     *   HELO,ACPT
     */
    myClient.connectToServer();
    Gui app = new Gui(myClient);
    // The following listener responds to the close event on the window
    // invoked when the user presses on the X at top right of the window in Windows
    // or the red button  at the top left in MacOSX
    app.addWindowListener(
        new WindowAdapter() {
          public void windowClosing( WindowEvent e )
          {
            System.out.println("-I-: pressed [x]");
            try{
              app.myClient.connectToServer();
              System.exit( 0 );
            } catch ( Exception e1) {
            }
          }
        }
        );
    app.runGui();

  }

}

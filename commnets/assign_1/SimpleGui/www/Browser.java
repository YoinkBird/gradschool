package www;

/**
 * Browser.java
 * This program uses a JEditorPane to display the
 * contents of a file from a Web server.
 * Class JEditorPane is capable of rendering both text and HTML
 * formatted text.
 */

import java.awt.*;
import java.awt.event.*;
import java.net.*;
import java.io.*;
import javax.swing.*;
import javax.swing.event.*;

public class Browser extends JFrame {
private JTextField enter;
private JEditorPane contents;

public Browser()
{
   super( "My Web Browser" ); // Call JFrame's constructor to set the title of the Window

   Container c = getContentPane(); // To add components to a JFrame we need a handle to its Content Pane

   enter = new JTextField( "Enter file URL here" ); // Create a new TextField where we want to user to enter the URL
    // There are several ways to register a Callback on a GUI component, here is the simplest
	//  We are creating a new anonymous object that implements the ActionListener Interface
	//   The ActionListener interface requires that you implement the ActionPerformed method
	//   and hence the code. The ActionPerformed method will be called when any event (like 
	//   entering text in a JTextField and typing enter) occurs on enter. In our implementation
	//   below we call the getThePage method and pass it what the user typed in
   	//      The alternative to this technique will be to create an instance of an Class
   	//      that implements the ActionListener interface. The important consideration with this 
   	//      alternate technique is to know that this newly created object may need more than
    //      just the String that the user typed in to be able to do something useful. In which
    //      we may have to pass that information in the constructor as well. (I will demonstrate
    //      this in class by modifying this code). 
   	// Note, use of anonymous inner classes is bad OO because, in your design documents, you cannot 
   	// refer to a class that is anonymous (no name)!! What do you call a class that has no name?
    //      
   enter.addActionListener(
      new ActionListener() {
         public void actionPerformed( ActionEvent e )
         {
            try {
				getThePage( new URL(e.getActionCommand()) );
			} catch (MalformedURLException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			} // e.getActionCommand() returns the URL typed in
            									// It would probably make sense to check if it is a well formed URL
            									// before sending it to the routine
         }
      }
   );
   c.add( enter, BorderLayout.NORTH ); // Place the "enter" component at the top (i.e., NORTH)

   contents = new JEditorPane(); // Create a JEditorPane which allows displaying html content
   contents.setEditable( false );
   // Adding a HyperlinkListener is conceptually similar to adding an ActionListener.
   // the difference is that a ActionListener reacts to what the user "typed" but
   // the HyperlinkListener reacts to what the user "clicked" on.
   contents.addHyperlinkListener(
      new HyperlinkListener() {
         public void hyperlinkUpdate( HyperlinkEvent e )
         {
            if ( e.getEventType() ==
                 HyperlinkEvent.EventType.ACTIVATED )    //There are many different events on a Hyperlink one can program for
            											//  Here we are interested in a Activation event.
               getThePage( e.getURL() );		// e.getURL will get the URL clicked on as URL object; We want it as a String
         }												// We could hve used the URL itself as well!
      }
   );
   c.add( new JScrollPane( contents ),
          BorderLayout.CENTER ); // Attach a Scrollpane to the JEditorPane before adding so we can scroll
                                 // Also add it to the CENTER

   setSize( 800, 600 );			// Set the size of the JFrame
   this.setVisible(true);	
}

private void getThePage( URL location )
{
   setCursor( Cursor.getPredefinedCursor(
                 Cursor.WAIT_CURSOR ) );		//Change the Cursor while the page loads

   try {
      contents.setPage( location );			// Update what gets displayed in the Editor Pane
      enter.setText( location.toString() );			// Update the URL string in the TextField
   }
   catch ( IOException io ) {
      JOptionPane.showMessageDialog( this,
         "Error retrieving specified URL",
         "Bad URL",
         JOptionPane.ERROR_MESSAGE );
   }

   setCursor( Cursor.getPredefinedCursor(
                 Cursor.DEFAULT_CURSOR ) ); //Change the Cursor back once page finished loading
}

public static void main( String args[] )
{
   Browser app = new Browser();
   
	// The following listener responds to the close event on the window
	// invoked when the user presses on the X at top right of the window in Windows
	// or the red button  at the top left in MacOSX
   app.addWindowListener(
      new WindowAdapter() {
         public void windowClosing( WindowEvent e )
         {
            System.exit( 0 );
         }
      }
   );
}
}


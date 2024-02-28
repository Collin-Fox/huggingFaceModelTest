package Collin.Fox.LLMServer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;

public class ApplicationInterface {
    private String data;
    private final int PORT_TO_PYTHON = 5050;

    public void sendToApplication(String data) throws IOException{
        System.out.println("Test");
        ServerSocket pythonSocket = new ServerSocket(PORT_TO_PYTHON);
        Socket socket = pythonSocket.accept();
        System.out.println("Connected to Python Application");

        InputStreamReader in = new InputStreamReader(socket.getInputStream());
        BufferedReader bf = new BufferedReader(in);

        PrintWriter printWriter = new PrintWriter(socket.getOutputStream());
        printWriter.println(data);
        printWriter.flush();

    }

}

package Collin.Fox.LLMServer;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.IOException;

@SpringBootApplication
public class LlmServerApplication {

	public static void main(String[] args) throws IOException {
		LLMRequest l = new LLMRequest("This is a new test prompt");
		///ApplicationInterface applicationInterface = new ApplicationInterface();
		//applicationInterface.sendToApplication("Connected to collins server");
		System.out.println(l.generateJson());
		SpringApplication.run(LlmServerApplication.class, args);
	}

}

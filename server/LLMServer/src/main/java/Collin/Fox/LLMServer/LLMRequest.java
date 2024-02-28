package Collin.Fox.LLMServer;

import org.springframework.stereotype.Controller;

import java.io.FileWriter;
import java.io.IOException;


public class LLMRequest {

    private String prompt;

    public LLMRequest(){}
    public LLMRequest(String prompt) {
        this.prompt = prompt;
    }

    public String generateJson(){
        return "{\n \"prompt\": \"" + this.prompt + "\"\n}";
    }
}

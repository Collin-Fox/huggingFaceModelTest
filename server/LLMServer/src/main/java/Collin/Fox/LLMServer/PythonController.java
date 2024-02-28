package Collin.Fox.LLMServer;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;

import java.io.IOException;

@Controller
@RequestMapping("/sendPrompt")
public class PythonController {

    @GetMapping("/target/{prompt}")
    public String sendToLLM(@PathVariable("prompt") String prompt) throws IOException {
        LLMRequest promptToPython = new LLMRequest(prompt);
        String data = promptToPython.generateJson();
        ApplicationInterface applicationInterface = new ApplicationInterface();
        applicationInterface.sendToApplication(prompt);

        return "SUCCESS";
    }
}

import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.commands import Command

class MusicMasterChat(Command):
    def __init__(self):
        super().__init__()
        self.name = "music"
        self.description = "Interact with a Music Expert AI to find new music to listen to."
        self.history = []
        load_dotenv()
        API_KEY = os.getenv('OPEN_AI_KEY')
        self.llm = ChatOpenAI(openai_api_key=API_KEY)  # Initialize once and reuse

    def calculate_tokens(self, text):
        # More accurate token calculation mimicking OpenAI's approach
        return len(text) + text.count(' ')

    def interact_with_ai(self, user_input, character_name):
        # Generate a more conversational and focused prompt
        prompt_text = "You're a Music Expert AI. Engage the user in a natural conversation about their favorite music genres. Use your insights to recommend music artists they might like based on their favorite genres."
        prompt = ChatPromptTemplate.from_messages(self.history + [("system", prompt_text)])
        
        output_parser = StrOutputParser()
        chain = prompt | self.llm | output_parser

        response = chain.invoke({"input": user_input})

        # Token usage logging and adjustment for more accurate counting
        tokens_used = self.calculate_tokens(prompt_text + user_input + response)
        logging.info(f"OpenAI API call made. Tokens used: {tokens_used}")
        return response, tokens_used

    def execute(self, *args, **kwargs):
        character_name = kwargs.get("character_name", "Movie Expert")
        print(f"Welcome to the Music Master Chat! Let's talk about your favorite music! Type 'done' to exit anytime.")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "done":
                print("Thank you for using the Music Master Chat. Goodbye!")
                break

            self.history.append(("user", user_input))
            
            try:
                response, tokens_used = self.interact_with_ai(user_input, character_name)
                print(f"Music Master: {response}")
                print(f"(This interaction used {tokens_used} tokens.)")
                self.history.append(("system", response))
            except Exception as e:
                print("Sorry, there was an error processing your request. Please try again.")
                logging.error(f"Error during interaction: {e}")
from langchain_core.tools import tool
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document


def create_a_guess_heatmap(llm):
    @tool("guess_heatmap", description="Shows a heatmap of top character guesses based on history and facts")
    def guess_heatmap_tool(history_and_context: str) -> str:
        """
        Analyzes the dialogue and context to guess the top 3 most likely characters.
        """
        prompt = f"""
        Do not guess just after 2 questions!

        Based on the following history and known facts, estimate the top 3 characters the user might be thinking of.
        Provide the name and a short confidence reason.

        HISTORY AND CONTEXT:
        {history_and_context}

        Respond in this format:
        1. Character A - 80% confidence: Reason
        2. Character B - 60% confidence: Reason
        3. Character C - 40% confidence: Reason
        """
        return llm.invoke(prompt).content
    return guess_heatmap_tool

def create_emotional_bot_tool(llm, user_input :str):
    @tool("emotion_response", description="React emotionally to the user's last message")
    def emotional_bot_tool() -> str:
        """
        Detects user's emotional tone and responds playfully or dramatically.
        """
        prompt = f"""
        
        Do not guess just after 2 questions!

        Read the user's input and craft a short, emotionally reactive response in character (like a pirate or anime narrator).
        Respond in a playful, dramatic, or witty tone depending on mood.

        User said:
        "{user_input}"
        """
        return llm.invoke(prompt).content
    return emotional_bot_tool


def create_trivia_mode(llm, context):
    @tool("one_piece_trivia", description="Ask a canon One Piece trivia question")
    def one_piece_trivia_tool() -> str:
        """
        Asks a trivia question to the user to keep things fun or challenge them.
        """
        prompt = f"""
        Ask the user a One Piece trivia question based on manga/anime facts. 
        Do not repeat previous trivia.

        Use the context data, for the only characters present in it: 
        {context}
        Example:
        'Trivia Time! 🔍 What is the name of Zoro’s cursed sword?'
        """
        return llm.invoke(prompt).content
    return one_piece_trivia_tool




def create_summarize_character_tool(llm):
    @tool
    def summarize_character(info: str) -> str:
        """
        Creatively summarizes a character based on raw fact data (info).
        """
        prompt = f"""
        Turn the following character details into an engaging, pirate-themed summary.

        Character Info:
        {info}
        """
        return llm.invoke(prompt).content

    return summarize_character


def create_generate_question_tool(llm):

    @tool("generate_question", description= " Generates strategic questions")
    def generate_question_tool(history_and_context: str) -> str:
        """
        This Function Generates strategic questions
        """
        
        prompt = f"""
        You are guessing a One Piece character. Use the following character context and history to generate a smart, strategic yes/no or multiple-choice question.

        CHAT HISTORY and CONTEXT:
        {history_and_context}
    
        Do not guess the answer just after 2 questions!
        Don't combine options into one paragraph.  Options should by line by line
        Don’t guess the character unless you are very confident.  
        Ask a **single question** that helps narrow down the character. Do NOT refer to unrelated characters.

        Now generate the question:
        """
        return llm.invoke(prompt).content
    return generate_question_tool


def create_generate_question_tool_QAPair(llm, clues):

    @tool("generate_question", description= " Generates strategic questions")
    def generate_question_tool(history_and_context: str) -> str:
        """
        This Function Generates strategic questions
        """
        
        prompt = f"""
        You are guessing a One Piece character. Use the following character context and clues to generate a smart, strategic yes/no or multiple-choice question.

        CHAT HISTORY and CONTEXT:
        {history_and_context}
        
        Only ask about the characters in the context.

        Here are the clues collected so far:
        {clues}

        Based on this structured clue data, ask the next best question to narrow down the character. 
        Only refer to characters that match the clues. Do not guess yet.

       
        Do not guess the answer just after 2 questions!
        Don't combine options into one paragraph.  Options should by line by line
        Don’t guess the character unless you are very confident.  
        Ask a **single question** that helps narrow down the character. Do NOT refer to unrelated characters.

        Now generate the question:
        """
        return llm.invoke(prompt).content
    return generate_question_tool
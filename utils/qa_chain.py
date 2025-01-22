#qa_chain.py
from openai import OpenAI
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

class QAChain:
    def __init__(self):
        """Initialize QA Chain with OpenAI."""
        self.model = "gpt-3.5-turbo"
        # self.model = "GPT-4o mini"
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_answer(self, question: str, context: List[Dict]) -> str:
        """Generate answer using OpenAI's chat completion."""
        # Format context for better readability
        formatted_context = "\n\n".join([
            f"Context {i+1}:\n{c['text']}"
            for i, c in enumerate(context)
        ])
        
        messages = [
            {
                "role": "system", 
                "content": """ 
                            
                            You are a professional assistant providing accurate, concise, and contextually relevant answers. 
                            Respond to the user's question based on the given context. If the answer is not available in the provided context, 
                            avoid wrong answers. 

                            if the context is present give the good answer with proper headings or
                            suppose the context not presenet
                            Instead of wrong answer follow the process:
                            1. Intimate the context not present,if it possible from the context Give some relevent data with proper heading and Suggest 2-3 alternative or related questions that only with in context of Database, only if the context not available and don't say any unwanted answer
                            2. Always maintain a professional and helpful tone.

                            Act like a real chatbot to give the resposnse, user may ask question or keywords only but give proper resposne properly
        """
            },
            # Format your responses in a clear, professional manner using markdown for better readability.
            {
                "role": "user", 
                "content": f"Context:\n{formatted_context}\n\nQuestion: {question}\n\nProvide a detailed answer based on the context above."
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content
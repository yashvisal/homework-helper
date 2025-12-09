"""
Tutor Agent - Q&A over uploaded documents (textbooks, notes, etc.)

Simple RAG-based conversational agent for studying and learning from documents.
"""
import logging
from typing import List, Optional, Dict, Any

from openai import OpenAI

from src.vectorstore import VectorStore
from src.tools import create_retrieval_tool

logger = logging.getLogger(__name__)


TUTOR_SYSTEM_PROMPT = """You are a helpful, patient tutor helping a student understand their course materials.

Your role:
- Answer questions based on the uploaded documents (textbooks, notes, lecture slides, etc.)
- Explain concepts clearly and at the student's level
- Provide examples and analogies to aid understanding
- If asked about something not in the documents, say so and offer to help find other resources
- Be encouraging and supportive

When referencing information from documents, cite the source:
- Use format: [Source: <filename>, p.<page>] for document citations

Always:
- Break down complex topics into digestible parts
- Check for understanding
- Offer to elaborate or clarify"""


class TutorAgent:
    """
    Tutor Agent for Q&A over uploaded documents.
    
    Uses RAG to retrieve relevant context from uploaded materials
    and provides helpful, educational responses.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        model: str = "gpt-4o",
        session_id: Optional[str] = None,
    ):
        """
        Initialize the Tutor Agent.
        
        Args:
            vector_store: VectorStore for document retrieval
            model: OpenAI model to use
            session_id: Session ID for filtering documents
        """
        self.vector_store = vector_store
        self.model = model
        self.session_id = session_id
        self.client = OpenAI()
        
        # Set up retrieval tool
        self.retrieval_tool = create_retrieval_tool(
            vector_store, 
            k=8,  # More context for tutoring
            session_id=session_id
        )
        
        print(f"[TUTOR] Initialized with model={model}, session_id={session_id}")
    
    def chat(
        self,
        user_message: str,
        conversation_history: Optional[List[dict]] = None,
    ) -> str:
        """
        Process a user question and return a helpful response.
        
        Args:
            user_message: The user's question
            conversation_history: Previous messages for context
            
        Returns:
            Tutor's response
        """
        history = conversation_history or []
        
        # Retrieve relevant context from documents
        context = ""
        if self.vector_store and self.vector_store.count > 0:
            try:
                print(f"[TUTOR] Retrieving context for: {user_message[:50]}...")
                context = self.retrieval_tool.func(user_message)
                print(f"[TUTOR] Retrieved {len(context)} chars of context")
            except Exception as e:
                print(f"[TUTOR] Retrieval error: {e}")
                context = ""
        
        # Build messages
        messages = [{"role": "system", "content": TUTOR_SYSTEM_PROMPT}]
        
        # Add conversation history
        for msg in history[-10:]:  # Keep last 10 messages for context
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Build user message with retrieved context
        if context and "No relevant documents found" not in context:
            augmented_message = f"""Student's Question: {user_message}

Relevant material from their documents:
{context}

Please answer based on the above material. Cite sources using [Source: filename, p.X] format."""
        else:
            augmented_message = f"""Student's Question: {user_message}

Note: No relevant documents were found. Please let the student know and offer general guidance if possible."""
        
        messages.append({"role": "user", "content": augmented_message})
        
        # Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_completion_tokens=4096
            )
            return response.choices[0].message.content or "I'm sorry, I couldn't generate a response."
        except Exception as e:
            print(f"[TUTOR] LLM error: {e}")
            return f"I encountered an error: {str(e)}"
    
    def summarize_document(self, source_name: Optional[str] = None) -> str:
        """
        Summarize an uploaded document or all documents.
        
        Args:
            source_name: Specific document to summarize, or None for all
            
        Returns:
            Summary of the document(s)
        """
        if not self.vector_store or self.vector_store.count == 0:
            return "No documents have been uploaded yet."
        
        # Get content from documents
        try:
            query = "main topics concepts key points summary"
            context = self.retrieval_tool.func(query)
            
            if not context or "No relevant documents found" in context:
                return "Could not retrieve content from documents."
            
            prompt = f"""Based on the following excerpts from the student's materials, provide a helpful summary of the main topics and concepts covered:

{context}

Provide a clear, organized summary that would help a student understand what's in their materials."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful tutor summarizing study materials."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_completion_tokens=2048
            )
            return response.choices[0].message.content or "Could not generate summary."
        except Exception as e:
            return f"Error generating summary: {str(e)}"


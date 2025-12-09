"""
Report Writing Agent - Research and draft papers with citations.

Uses LangGraph for multi-step workflow:
1. Research - Gather information from documents and web
2. Outline - Create structured outline
3. Draft - Write full report with citations
4. Revise - Handle user feedback and revisions
"""

import os
import re
import logging
from typing import TypedDict, List, Optional, Annotated, Dict, Any
from dataclasses import dataclass, field

from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import Tool

from src.vectorstore import VectorStore
from src.tools import create_retrieval_tool, create_web_search_tool

# Lightweight logger for stdout (Streamlit terminal)
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# State definition for the agent
class ReportState(TypedDict):
    """State passed between nodes in the report writing workflow."""
    topic: str
    messages: List[dict]
    research_notes: str
    outline: str
    draft: str
    citations: List[str]
    current_step: str
    user_feedback: Optional[str]


# Prompt templates
RESEARCH_PROMPT = """You are a meticulous research assistant. Your task is to gather high-quality, academic-grade information about the following topic:

{topic}

Use the available tools to:
1. Search uploaded documents for relevant information
2. Search the web for additional sources (aim for 3+ credible web sources with URLs)

Requirements:
- Produce comprehensive research notes suitable for an academic paper
- For document sources, cite as [Source: <filename>, p.<page>]
- For web sources, cite as [Source: <title>, <url>] (not just 'Web')
- If web search yields nothing, explicitly state "No web sources found" and do NOT invent citations or URLs
- Do NOT say "in a real report" or apologize; always produce polished content
- If sources are limited, still produce the best possible evidence-backed notes without fabrication

Available context from conversation:
{context}
"""

OUTLINE_PROMPT = """Based on the following research notes, create a detailed, academically-rigorous outline for a research paper.

Topic: {topic}

Research Notes:
{research_notes}

Create an outline with:
- Clear thesis statement
- Introduction section
- 3-5 main body sections with subsections
- Conclusion section
- References section placeholder

Format the outline with proper numbering and hierarchy."""

DRAFT_PROMPT = """Write a complete academic report based on the following outline and research.

Topic: {topic}

Outline:
{outline}

Research Notes:
{research_notes}

Requirements:
- Write in academic style with formal tone
- Include inline citations:
  - Documents: [Source: <filename>, p.<page>]
  - Web: [Source: <title>, <url>]
- Expand each outline section into **2–4 well-developed paragraphs** with evidence and explanation
- Maintain logical flow between sections
- Include a Works Cited section at the end
- Do NOT mention limitations about sources; always provide polished, academic-quality prose
- Avoid meta comments such as "in a real report"
- Target total length: **at least 1,000–1,500 words** for the full draft

Write the complete report now."""

REVISE_PROMPT = """Revise the following report based on user feedback.

Current Draft:
{draft}

User Feedback:
{feedback}

Make the requested changes while maintaining:
- Academic tone and style
- Proper citations
- Logical structure

Return the revised report."""


class ReportAgent:
    """
    Multi-step report writing agent using LangGraph.
    
    Capabilities:
    - Document retrieval via RAG
    - Web search for additional sources
    - Structured outline generation
    - Full report drafting with citations
    - Revision handling
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        model: str = "gpt-5.1",
        enable_web_search: bool = True,
        session_id: Optional[str] = None,
    ):
        """
        Initialize the Report Agent.
        
        Args:
            vector_store: VectorStore for document retrieval
            model: OpenAI model to use
            enable_web_search: Whether to enable web search tool
            session_id: Session ID for filtering documents
        """
        self.vector_store = vector_store
        self.model = model
        self.session_id = session_id
        self.client = OpenAI()
        
        # Set up tools
        self.tools = []
        
        # Retrieval tool (with session filtering)
        retrieval_tool = create_retrieval_tool(vector_store, k=5, session_id=session_id)
        self.tools.append(retrieval_tool)
        
        # Web search tool (optional)
        if enable_web_search:
            web_tool = create_web_search_tool(max_results=5)
            if web_tool:
                self.tools.append(web_tool)
            else:
                print("[WARN] Web search tool not created (TAVILY_API_KEY missing?)")
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create graph with state schema
        workflow = StateGraph(ReportState)
        
        # Add nodes
        workflow.add_node("research", self._research_node)
        workflow.add_node("outline", self._outline_node)
        workflow.add_node("draft", self._draft_node)
        workflow.add_node("revise", self._revise_node)
        
        # Define edges
        workflow.set_entry_point("research")
        workflow.add_edge("research", "outline")
        workflow.add_edge("outline", "draft")
        workflow.add_edge("draft", END)
        workflow.add_edge("revise", END)
        
        return workflow.compile()
    
    def _call_llm(self, system_prompt: str, user_message: str, max_tokens: int = 16384) -> str:
        """Make a call to the LLM."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_completion_tokens=max_tokens
            )
            content = response.choices[0].message.content
            if not content:
                print(f"[WARN] LLM returned empty content. Finish reason: {response.choices[0].finish_reason}")
                return ""
            return content
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            return ""
    
    def _call_llm_with_tools(self, system_prompt: str, user_message: str) -> str:
        """Make a call to the LLM with tool use."""
        # Convert tools to OpenAI format
        tool_schemas = []
        for tool in self.tools:
            tool_schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Initial call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tool_schemas if tool_schemas else None,
            temperature=0.7,
            max_tokens=4096
        )
        
        assistant_message = response.choices[0].message
        
        # Handle tool calls
        all_results = []
        while assistant_message.tool_calls:
            messages.append(assistant_message)
            
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                
                # Parse arguments
                import json
                args = json.loads(tool_call.function.arguments)
                query = args.get("query", "")
                
                # Find and execute tool
                result = "Tool not found"
                for tool in self.tools:
                    if tool.name == tool_name:
                        result = tool.func(query)
                        break
                
                all_results.append(f"[{tool_name}]: {result}")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            
            # Get next response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_schemas,
                temperature=0.7,
                max_tokens=4096
            )
            assistant_message = response.choices[0].message
        
        return assistant_message.content
    
    def _research_node(self, state: ReportState) -> ReportState:
        """Research node - gather information from documents and web."""
        topic = state["topic"]
        context = "\n".join([
            f"{m['role']}: {m['content']}" 
            for m in state.get("messages", [])[-5:]  # Last 5 messages for context
        ])
        
        prompt = RESEARCH_PROMPT.format(topic=topic, context=context)
        
        # Use tools for research
        research_notes = self._call_llm_with_tools(
            system_prompt="You are a thorough research assistant. Use tools to gather comprehensive information.",
            user_message=prompt
        )
        
        state["research_notes"] = research_notes
        state["current_step"] = "research_complete"
        return state
    
    def _outline_node(self, state: ReportState) -> ReportState:
        """Outline node - create structured report outline."""
        prompt = OUTLINE_PROMPT.format(
            topic=state["topic"],
            research_notes=state["research_notes"]
        )
        
        outline = self._call_llm(
            system_prompt="You are an expert at creating clear, logical outlines for academic papers.",
            user_message=prompt
        )
        
        state["outline"] = outline
        state["current_step"] = "outline_complete"
        return state
    
    def _draft_node(self, state: ReportState) -> ReportState:
        """Draft node - write the full report."""
        prompt = DRAFT_PROMPT.format(
            topic=state["topic"],
            outline=state["outline"],
            research_notes=state["research_notes"]
        )
        
        draft = self._call_llm(
            system_prompt="You are an expert academic writer. Write clear, well-structured reports with proper citations.",
            user_message=prompt
        )
        
        state["draft"] = draft
        state["current_step"] = "draft_complete"
        
        # Extract citations (simple extraction)
        citations = []
        import re
        citation_pattern = r'\[([^\]]+)\]'
        found = re.findall(citation_pattern, draft)
        citations = list(set(found))
        state["citations"] = citations
        
        return state
    
    def _revise_node(self, state: ReportState) -> ReportState:
        """Revise node - handle user revision requests."""
        prompt = REVISE_PROMPT.format(
            draft=state["draft"],
            feedback=state["user_feedback"]
        )
        
        revised = self._call_llm(
            system_prompt="You are an expert editor. Revise the report based on feedback while maintaining quality.",
            user_message=prompt
        )
        
        state["draft"] = revised
        state["current_step"] = "revision_complete"
        return state
    
    def generate_report(
        self,
        topic: str,
        messages: Optional[List[dict]] = None
    ) -> dict:
        """
        Generate a complete report on the given topic.
        
        Args:
            topic: The research topic or question
            messages: Optional conversation history for context
            
        Returns:
            Dict with research_notes, outline, draft, and citations
        """
        initial_state: ReportState = {
            "topic": topic,
            "messages": messages or [],
            "research_notes": "",
            "outline": "",
            "draft": "",
            "citations": [],
            "current_step": "start",
            "user_feedback": None
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        return {
            "research_notes": final_state["research_notes"],
            "outline": final_state["outline"],
            "draft": final_state["draft"],
            "citations": final_state["citations"]
        }
    
    def revise_report(
        self,
        current_draft: str,
        feedback: str,
        topic: str = ""
    ) -> str:
        """
        Revise an existing report based on feedback.
        
        Args:
            current_draft: The current report draft
            feedback: User's revision feedback
            topic: Original topic (optional)
            
        Returns:
            Revised report
        """
        state: ReportState = {
            "topic": topic,
            "messages": [],
            "research_notes": "",
            "outline": "",
            "draft": current_draft,
            "citations": [],
            "current_step": "awaiting_revision",
            "user_feedback": feedback
        }
        
        revised_state = self._revise_node(state)
        return revised_state["draft"]
    
    def chat(
        self,
        user_message: str,
        conversation_history: Optional[List[dict]] = None,
        topic: str = ""
    ) -> str:
        """
        Handle a chat message in the context of report writing.
        
        Args:
            user_message: User's message
            conversation_history: Previous messages in the conversation
            topic: The report topic for context
            
        Returns:
            Agent's response
        """
        history = conversation_history or []
        
        # Build system prompt with topic context
        system_prompt = f"""You are a helpful report writing assistant. You can:
1. Help research topics using uploaded documents and web search
2. Create outlines for academic papers
3. Draft reports with proper citations
4. Revise and improve existing drafts

{"Current report topic: " + topic if topic else ""}

IMPORTANT: Maintain context from the conversation history. Reference previous work (outlines, research, drafts) when the user asks follow-up questions.

Be helpful, professional, and thorough."""
        
        # Build messages with full conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in history[-10:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        # Determine if this needs tool use
        needs_research = any(word in user_message.lower() for word in [
            "research", "find", "search", "look up", "sources", "write report",
            "draft", "paper", "essay", "outline", "create", "write"
        ])
        
        if needs_research:
            response = self._call_llm_with_tools_and_history(messages)
        else:
            response = self._call_llm_with_history(messages)
        
        return response
    
    def _call_llm_with_history(self, messages: List[dict]) -> str:
        """Make a call to the LLM with full message history."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_completion_tokens=4096
        )
        return response.choices[0].message.content
    
    def _call_llm_with_tools_and_history(self, messages: List[dict]) -> str:
        """Make a call to the LLM with tools and full history."""
        # Convert tools to OpenAI format
        tool_schemas = []
        for tool in self.tools:
            tool_schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
        
        # Initial call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tool_schemas if tool_schemas else None,
            temperature=0.7,
            max_completion_tokens=4096
        )
        
        # Tool usage logging
        if tool_schemas:
            print(f"[TOOLS] Enabled tools: {[t['function']['name'] for t in tool_schemas]}")

        assistant_message = response.choices[0].message
        
        # Handle tool calls
        while assistant_message.tool_calls:
            messages.append(assistant_message)
            
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                
                import json
                args = json.loads(tool_call.function.arguments)
                query = args.get("query", "")
                
                result = "Tool not found"
                for tool in self.tools:
                    if tool.name == tool_name:
                        print(f"[TOOL_CALL] {tool_name} query='{query[:120]}' len={len(query)}")
                        result = tool.func(query)
                        break
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_schemas,
                temperature=0.7,
                max_tokens=4096
            )
            assistant_message = response.choices[0].message
        
        return assistant_message.content

    # --------- New helper methods for stepwise workflow ----------
    def generate_research_only(
        self,
        topic: str,
        messages: Optional[List[dict]] = None,
        has_documents: bool = True,
    ) -> Dict[str, Any]:
        """Generate research notes only (no outline yet)."""
        context = ""
        if messages:
            context = "\n".join([f"{m['role']}: {m['content']}" for m in messages[-6:]])

        # Explicitly run retrieval and web search; if docs exist, use both; otherwise only web
        retrieved_docs = ""
        web_results = ""
        doc_available = False
        web_available = False

        print(f"[DEBUG] Tools available: {[t.name for t in self.tools]}")
        print(f"[DEBUG] has_documents={has_documents}")
        
        for tool in self.tools:
            if tool.name == "retrieve_documents" and has_documents:
                try:
                    print(f"[TOOL] Calling retrieve_documents...")
                    retrieved_docs = tool.func(f"{topic} academic sources key arguments evidence")
                    print(f"[TOOL] retrieve_documents returned {len(retrieved_docs)} chars")
                    if retrieved_docs and "No relevant documents found" not in retrieved_docs:
                        doc_available = True
                except Exception as e:
                    print(f"[TOOL] retrieve_documents error: {e}")
                    retrieved_docs = ""
            if tool.name == "web_search":
                try:
                    print(f"[TOOL] Calling web_search...")
                    web_results = tool.func(f"{topic} academic sources environmental economic safety impacts")
                    print(f"[TOOL] web_search returned {len(web_results)} chars")
                    if web_results and "No results found" not in web_results:
                        web_available = True
                except Exception as e:
                    print(f"[TOOL] web_search error: {e}")
                    web_results = ""

        if not doc_available:
            retrieved_docs = "No documents found. If none, rely on web sources only."
        if not web_available:
            web_results = "No web sources found. DO NOT invent or fabricate web sources or URLs. If none, omit web citations."

        # Build synthesis prompt with both sources
        synthesis_prompt = f"""
You are a meticulous research assistant. Combine the retrieved document excerpts and web search results to produce academic-grade research notes.

Topic: {topic}

Conversation context:
{context}

Retrieved documents:
{retrieved_docs or 'No documents found.'}

Web search results:
{web_results or 'No web results found.'}

Requirements:
- Use document citations as [Source: <filename>, p.<page>]
- Use web citations as [Source: <title>, <url>]
- Include at least 3 credible web citations with URLs even if documents exist
- If no documents are available, rely solely on web sources; if no web sources, say so and do NOT fabricate
- Produce comprehensive, well-organized notes suitable for drafting an academic paper
"""

        research_notes = self._call_llm(
            system_prompt="You synthesize sources into high-quality academic research notes with explicit citations.",
            user_message=synthesis_prompt,
        )

        citations = self._extract_citations(research_notes)
        print(f"[STAGE] Research generated len={len(research_notes)} citations={len(citations)}")

        return {
            "research_notes": research_notes,
            "citations": citations,
        }

    def generate_outline_only(
        self, topic: str, research_notes: str
    ) -> Dict[str, Any]:
        """Generate outline from existing research notes."""
        print(f"[OUTLINE] Starting outline generation. Topic: {topic[:50]}... Research notes len: {len(research_notes)}")
        
        if not research_notes or len(research_notes) < 50:
            print(f"[OUTLINE] Warning: research_notes is too short or empty!")
        
        outline_prompt = OUTLINE_PROMPT.format(
            topic=topic,
            research_notes=research_notes,
        )
        print(f"[OUTLINE] Calling LLM with prompt len: {len(outline_prompt)}")
        
        outline = self._call_llm(
            system_prompt="You are an expert at creating clear, logical outlines for academic papers.",
            user_message=outline_prompt,
        )

        citations = self._extract_citations(outline)
        print(f"[STAGE] Outline generated len={len(outline)} citations={len(citations)}")

        return {
            "outline": outline,
            "citations": citations,
        }

    def generate_draft_from_outline(
        self, topic: str, outline: str, research_notes: str, messages: Optional[List[dict]] = None
    ) -> str:
        """Generate a full draft from outline and research notes."""
        draft_prompt = DRAFT_PROMPT.format(
            topic=topic,
            outline=outline,
            research_notes=research_notes,
        )
        draft = self._call_llm(
            system_prompt="You are an expert academic writer. Produce a polished, well-cited draft.",
            user_message=draft_prompt,
        )
        logger.info("Generated draft (len=%d)", len(draft))
        return draft

    def regenerate_outline_with_feedback(
        self, topic: str, research_notes: str, feedback: str
    ) -> str:
        """Regenerate outline with feedback."""
        prompt = (
            "Revise and improve the outline based on feedback.\n\n"
            f"Topic: {topic}\n\n"
            f"Existing research notes:\n{research_notes}\n\n"
            f"Feedback:\n{feedback}\n\n"
            "Return a clean, numbered outline."
        )
        outline = self._call_llm(
            system_prompt="You are an expert at producing clear academic outlines.",
            user_message=prompt,
        )
        return outline

    def _extract_citations(self, text: str) -> List[str]:
        """Extract bracketed citations like [Source #] or [Author, Year]."""
        found = re.findall(r"\[([^\]]+)\]", text or "")
        # Deduplicate while preserving order
        seen = set()
        citations: List[str] = []
        for c in found:
            if c not in seen:
                seen.add(c)
                citations.append(c)
        return citations


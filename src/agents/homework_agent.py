"""
Homework Solver Agent - Step-by-step guided problem solving.

Uses GPT's native multimodal capabilities to:
- Understand images of homework problems directly in chat
- Retrieve relevant textbook/notes sections via RAG
- Generate chain-of-thought step-by-step solutions
- Analyze student work and provide feedback
"""

import os
import re
from typing import TypedDict, List, Optional, Union

from openai import OpenAI
from langgraph.graph import StateGraph, END

from src.vectorstore import VectorStore
from src.tools import create_retrieval_tool, encode_image_base64, create_wolfram_tool


# State definition
class HomeworkState(TypedDict):
    """State for the homework solving workflow."""
    problem_text: str
    problem_image: Optional[bytes]
    student_work_image: Optional[bytes]
    messages: List[dict]
    reference_context: str
    solution_steps: List[str]
    final_answer: str
    feedback: str
    current_step: str


# Prompt templates
SOLUTION_PROMPT = """You are an expert tutor helping a student solve homework problems. 

Problem:
{problem}

Relevant Reference Material:
{reference_context}

Provide a complete step-by-step solution following these guidelines:
1. First, identify what type of problem this is and what concepts are needed
2. Break down the solution into clear, numbered steps
3. Explain the reasoning behind each step
4. Show all work and calculations
5. Highlight key formulas or concepts used
6. Provide the final answer clearly

Use the reference material to support your explanations where relevant.
Be educational - help the student understand, not just get the answer."""

CHAT_SYSTEM_PROMPT = """You are a helpful homework tutor. You can:
1. Solve homework problems step-by-step with clear explanations
2. Look at images of homework and understand them directly
3. Look up relevant information from uploaded textbooks/notes
4. Analyze student's work and provide feedback
5. Explain concepts in a clear, educational way

When solving problems:
- Always show your work step-by-step
- Explain the reasoning behind each step
- Reference relevant concepts from the textbook when available
- Be encouraging and educational

When the student shares an image of their work, provide constructive feedback.
When they share an image of a problem, solve it step by step."""


class HomeworkAgent:
    """
    Step-by-step homework solver with native multimodal capabilities.
    
    Features:
    - Multimodal: Understands images directly via GPT's vision
    - RAG: References textbook/notes for context
    - Chain-of-thought: Step-by-step reasoning
    - Feedback: Analyzes student work
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        model: str = "gpt-5.1"
    ):
        """
        Initialize the Homework Agent.
        
        Args:
            vector_store: VectorStore for textbook/notes retrieval
            model: OpenAI model to use (multimodal model like gpt-5.1)
        """
        self.vector_store = vector_store
        self.model = model
        self.client = OpenAI()
        
        # Set up tools
        self.retrieval_tool = create_retrieval_tool(vector_store, k=5)
        self.wolfram_tool = create_wolfram_tool()
        
        if self.wolfram_tool:
            print("[HOMEWORK] Wolfram Alpha tool enabled")
        else:
            print("[HOMEWORK] Wolfram Alpha tool not available (set WOLFRAM_ALPHA_APPID)")
        
        # Build workflow graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for homework solving."""
        workflow = StateGraph(HomeworkState)
        
        # Add nodes
        workflow.add_node("understand", self._understand_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("solve", self._solve_node)
        workflow.add_node("feedback", self._feedback_node)
        
        # Entry point
        workflow.set_entry_point("understand")
        
        # Define edges
        workflow.add_edge("understand", "retrieve")
        workflow.add_edge("retrieve", "solve")
        
        # Conditional edge for feedback
        workflow.add_conditional_edges(
            "solve",
            lambda state: "feedback" if state.get("student_work_image") else "end",
            {
                "feedback": "feedback",
                "end": END
            }
        )
        workflow.add_edge("feedback", END)
        
        return workflow.compile()
    
    def _understand_node(self, state: HomeworkState) -> HomeworkState:
        """Understand the problem - extract from image if provided."""
        if state.get("problem_image") and not state.get("problem_text"):
            # Use multimodal chat to understand the image
            base64_image = encode_image_base64(state["problem_image"])
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please examine this homework problem image and extract/describe the problem clearly. Include all text, equations, diagrams, and any relevant details."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2048
            )
            state["problem_text"] = response.choices[0].message.content
        
        state["current_step"] = "understood"
        return state
    
    def _retrieve_node(self, state: HomeworkState) -> HomeworkState:
        """Retrieve relevant reference material."""
        problem = state["problem_text"]
        
        # Search for relevant context
        context = self.retrieval_tool.func(problem)
        state["reference_context"] = context
        state["current_step"] = "retrieved"
        
        return state
    
    def _solve_node(self, state: HomeworkState) -> HomeworkState:
        """Generate step-by-step solution."""
        prompt = SOLUTION_PROMPT.format(
            problem=state["problem_text"],
            reference_context=state["reference_context"]
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert tutor who explains problems clearly with step-by-step solutions."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4096
        )
        
        solution = response.choices[0].message.content
        
        # Parse steps
        steps = re.split(r'\n(?=\d+\.|\*\*Step)', solution)
        steps = [s.strip() for s in steps if s.strip()]
        
        state["solution_steps"] = steps
        state["final_answer"] = solution
        state["current_step"] = "solved"
        
        return state
    
    def _feedback_node(self, state: HomeworkState) -> HomeworkState:
        """Analyze student work and provide feedback using multimodal chat."""
        if not state.get("student_work_image"):
            state["feedback"] = ""
            return state
        
        base64_image = encode_image_base64(state["student_work_image"])
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a supportive tutor providing constructive feedback on student work."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Analyze this student's work for the following problem:

{state['problem_text']}

The correct approach is:
{state['final_answer']}

Please provide:
1. What the student did correctly
2. Any errors or misconceptions identified
3. Specific corrections with explanations
4. Suggestions for improvement
5. Encouragement and positive feedback"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2048
        )
        
        state["feedback"] = response.choices[0].message.content
        state["current_step"] = "feedback_complete"
        
        return state
    
    def solve_problem(
        self,
        problem_text: Optional[str] = None,
        problem_image: Optional[Union[str, bytes]] = None,
        student_work_image: Optional[Union[str, bytes]] = None
    ) -> dict:
        """
        Solve a homework problem with step-by-step explanation.
        
        Args:
            problem_text: Text description of the problem
            problem_image: Image of the problem (path or bytes)
            student_work_image: Optional image of student's work for feedback
            
        Returns:
            Dict with problem, solution_steps, final_answer, feedback, and reference_context
        """
        # Load images if paths provided
        if isinstance(problem_image, str):
            with open(problem_image, "rb") as f:
                problem_image = f.read()
        
        if isinstance(student_work_image, str):
            with open(student_work_image, "rb") as f:
                student_work_image = f.read()
        
        # Initialize state
        initial_state: HomeworkState = {
            "problem_text": problem_text or "",
            "problem_image": problem_image,
            "student_work_image": student_work_image,
            "messages": [],
            "reference_context": "",
            "solution_steps": [],
            "final_answer": "",
            "feedback": "",
            "current_step": "start"
        }
        
        # Run workflow
        final_state = self.graph.invoke(initial_state)
        
        return {
            "problem": final_state["problem_text"],
            "solution_steps": final_state["solution_steps"],
            "final_answer": final_state["final_answer"],
            "feedback": final_state["feedback"],
            "reference_context": final_state["reference_context"]
        }
    
    def chat(
        self,
        user_message: str,
        conversation_history: Optional[List[dict]] = None,
        image: Optional[Union[str, bytes]] = None
    ) -> str:
        """
        Handle a chat message, optionally with an image.
        
        Images are passed directly to the multimodal model - no separate
        vision processing needed.
        
        Args:
            user_message: User's message
            conversation_history: Previous messages
            image: Optional image to include in the message
            
        Returns:
            Agent's response
        """
        history = conversation_history or []
        
        # Build messages
        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        
        # Add history
        for msg in history[-10:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Check if we need RAG
        needs_retrieval = any(word in user_message.lower() for word in [
            "solve", "help", "problem", "question", "homework", "explain",
            "textbook", "notes", "formula", "equation", "calculate"
        ])
        
        # Get reference context if needed
        reference_context = ""
        if needs_retrieval:
            reference_context = self.retrieval_tool.func(user_message)
            if reference_context and reference_context != "No relevant documents found.":
                user_message = f"{user_message}\n\nRelevant reference material:\n{reference_context}"
        
        # Build user message - include image if provided
        if image:
            if isinstance(image, str):
                with open(image, "rb") as f:
                    image = f.read()
            
            base64_image = encode_image_base64(image)
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": user_message})
        
        # Call multimodal LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.5,
            max_tokens=4096
        )
        
        return response.choices[0].message.content
    
    def extract_questions_from_text(self, document_text: str) -> List[dict]:
        """
        Extract individual questions from a homework document.
        
        Args:
            document_text: Full text of the homework document
            
        Returns:
            List of dicts with 'number' and 'question' keys
        """
        prompt = f"""Analyze this homework document and extract each individual question/problem.

Document:
{document_text[:8000]}  # Limit to avoid token issues

Return a numbered list of questions in this exact format:
Q1: [full question text]
Q2: [full question text]
...

Extract ALL questions including sub-parts (a, b, c, etc.). 
Include any necessary context or data provided with each question.
If there are equations, include them exactly as written."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at parsing homework documents and extracting individual questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_completion_tokens=4096
            )
            
            content = response.choices[0].message.content or ""
            
            # Parse the response into questions
            questions = []
            lines = content.split('\n')
            current_q = None
            current_text = []
            
            for line in lines:
                # Check if this is a new question
                match = re.match(r'^Q(\d+):\s*(.*)$', line, re.IGNORECASE)
                if match:
                    # Save previous question
                    if current_q is not None:
                        questions.append({
                            'number': current_q,
                            'question': '\n'.join(current_text).strip()
                        })
                    current_q = int(match.group(1))
                    current_text = [match.group(2)]
                elif current_q is not None:
                    current_text.append(line)
            
            # Don't forget the last question
            if current_q is not None:
                questions.append({
                    'number': current_q,
                    'question': '\n'.join(current_text).strip()
                })
            
            print(f"[HOMEWORK] Extracted {len(questions)} questions from document")
            return questions
            
        except Exception as e:
            print(f"[HOMEWORK] Error extracting questions: {e}")
            return []
    
    def solve_question_with_wolfram(self, question: str) -> dict:
        """
        Solve a single question, using Wolfram Alpha for computations.
        
        Args:
            question: The question text
            
        Returns:
            Dict with 'question', 'wolfram_result', 'solution', 'explanation'
        """
        result = {
            'question': question,
            'wolfram_result': None,
            'solution': '',
            'explanation': ''
        }
        
        # Try Wolfram Alpha for mathematical computations
        if self.wolfram_tool:
            # Extract mathematical expressions to compute
            math_prompt = f"""Analyze this question and extract any mathematical expressions that need to be computed:

{question}

If there are equations to solve, integrals, derivatives, or calculations, return them in a format suitable for Wolfram Alpha.
Return ONLY the mathematical expression, nothing else. If there's nothing to compute, return "NONE"."""
            
            try:
                math_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Extract mathematical expressions for computation."},
                        {"role": "user", "content": math_prompt}
                    ],
                    temperature=0,
                    max_completion_tokens=256
                )
                
                math_expr = math_response.choices[0].message.content.strip()
                
                if math_expr and math_expr.upper() != "NONE":
                    print(f"[WOLFRAM] Computing: {math_expr[:50]}...")
                    wolfram_result = self.wolfram_tool.func(math_expr)
                    result['wolfram_result'] = wolfram_result
                    print(f"[WOLFRAM] Result: {wolfram_result[:100]}...")
            except Exception as e:
                print(f"[WOLFRAM] Error: {e}")
        
        # Get reference context
        reference_context = ""
        try:
            reference_context = self.retrieval_tool.func(question)
        except:
            pass
        
        # Generate full solution with explanation
        solution_prompt = f"""Solve this homework problem with a complete step-by-step explanation:

Problem:
{question}

{"Wolfram Alpha Result: " + result['wolfram_result'] if result['wolfram_result'] else ""}

{"Reference Material: " + reference_context if reference_context and "No relevant" not in reference_context else ""}

Provide:
1. Problem understanding - what is being asked
2. Step-by-step solution with clear reasoning
3. Final answer clearly marked
4. Brief explanation of key concepts used

Be educational and thorough."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert tutor providing detailed homework solutions."},
                    {"role": "user", "content": solution_prompt}
                ],
                temperature=0.3,
                max_completion_tokens=4096
            )
            
            result['solution'] = response.choices[0].message.content or ""
            result['explanation'] = "Solution generated with step-by-step reasoning."
            
        except Exception as e:
            result['solution'] = f"Error generating solution: {str(e)}"
        
        return result
    
    def process_homework_document(self, document_text: str) -> List[dict]:
        """
        Process a full homework document: extract questions and solve each one.
        
        Args:
            document_text: Full text of the homework document
            
        Returns:
            List of solution dicts, one per question
        """
        print("[HOMEWORK] Processing homework document...")
        
        # Extract questions
        questions = self.extract_questions_from_text(document_text)
        
        if not questions:
            return [{
                'question': document_text[:500],
                'wolfram_result': None,
                'solution': "Could not extract individual questions. Please try again with clearer formatting.",
                'explanation': ''
            }]
        
        # Solve each question
        solutions = []
        for i, q in enumerate(questions):
            print(f"[HOMEWORK] Solving question {q['number']} ({i+1}/{len(questions)})...")
            solution = self.solve_question_with_wolfram(q['question'])
            solution['number'] = q['number']
            solutions.append(solution)
        
        print(f"[HOMEWORK] Completed {len(solutions)} solutions")
        return solutions

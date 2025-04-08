from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from dotenv import load_dotenv
import os
import json

from textwrap import dedent
from agno.agent import Agent, RunResponse
from agno.models.groq import Groq
from agno.tools.googlesearch import GoogleSearchTools
from agno.models.huggingface import HuggingFace
from models import CaseStudyInput, CaseStudyOutput

app=FastAPI()

load_dotenv()


websearch_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    tools = [GoogleSearchTools()],
    description="You are a web serch agent which searches for the information related to client provided via input for the case study",
    instructions=[
        "Serach through the web for the information of the client",
        "Summarize the information of the client in a single paragraph, detailing about what they do, which industry they belong to and since when they started",
        "Provide the paragraph in a proper structured format which should engaging while reading it",
        "Make sure that all the information provided is factually correct",
    ],

    show_tool_calls=True,
    debug_mode=True,
    markdown=True
)



enhancer_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    description="Expert in elaborating case study details with factual accuracy and engaging narrative",
    instructions=dedent("""\
    You must EXCLUSIVELY work with the provided input data. Never generate generic examples.

    Your output should be structured in Markdown format with the following sections for each input:
    
    Processing Steps:
    1. For each client requirement bullet point:
       - Create a detailed paragraph explaining what it entails
       - Add 2-3 technical/business specifics
       - State the business value
    
    2. For each challenge:
       - Explain why it was problematic
       - Describe the technical complexity
       - State what would happen if not solved
    
    3. For each solution component:
       - Explain implementation details
       - State selection rationale
       - Describe integration points
    
    4. For each result:
       - Add context about measurement methodology
       - Explain business impact
       - Provide relevant comparisons
    
    Strict Requirements:
    - ONLY use information from the provided input
    - Never invent new facts or figures
    - Maintain original numbering/grouping
    - Output in clean markdown format
    - Please make the content look less AI generated and keep 
"""),
    markdown=True
)

@app.post("/generate_case_study")
async def generate_case_study (input_data: CaseStudyInput) -> CaseStudyOutput:
    try:
        client_info = input_data.client_name
        if not client_info:
            raise HTTPException(status_code=400, detail="Client name is required for Web Search")

        # Use the websearch_agent to get client information
        web_search_response: RunResponse = await websearch_agent.run(client_info)
        client_background = web_search_response.content if web_search_response and web_search_response.content else "No specific client information found on the web."

        formatted_input = input_data.model_dump_json(indent=4) # Use Pydantic's method for JSON serialization
        enhancer_response: RunResponse = await enhancer_agent.run(formatted_input) # Use await for async agent run

        final_response=f"""
        ## About {client_info}

        {client_background}

        ## Client Requirements

        {enhancer_response.content.get("client_requirements", "") if enhancer_response and enhancer_response.content else ""}

        ## Challenges Faced

        {enhancer_response.content.get("challenges_faced", "") if enhancer_response and enhancer_response.content else ""}

        ## Solution Provided

        {enhancer_response.content.get("solution_provided", "") if enhancer_response and enhancer_response.content else ""}

        ## Results Achieved

        {enhancer_response.content.get("results_achieved", "") if enhancer_response and enhancer_response.content else ""}
        """

        return CaseStudyOutput(generated_case_study={"result": final_response})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

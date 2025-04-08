from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class CaseStudyInput(BaseModel):
    client_name: Optional [str] = None
    client_requirements: List[str]=[]
    challenges_faced: List[str]=[]
    solution_provided: List[str]=[]
    results_achieved: List[str]=[]

class CaseStudyOutput(BaseModel):
    generated_case_study: Dict[str, Any]
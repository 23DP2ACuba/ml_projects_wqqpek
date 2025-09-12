from pydantic import BaseModel, Field

class Task(BaseModel):
    id: str = Field(..., description="The unique identifier for the field")
    name: str = Field(..., description="name ot the task")
    is_completed: bool = Field(..., description="Indicates if is completed")

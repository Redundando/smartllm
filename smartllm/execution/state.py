from enum import Enum

class LLMRequestState(Enum):
    COMPLETED = "completed"
    FAILED = "failed"
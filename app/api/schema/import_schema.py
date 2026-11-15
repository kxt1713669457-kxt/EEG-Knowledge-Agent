from pydantic import BaseModel, Field


class UploadResponseSchema(BaseModel):
    code: int = 200
    message: str | None = None
    task_ids: list[str]
    documents: list[dict] = Field(default_factory=list)

class StatusResponseSchema(BaseModel):
    code:int=200
    task_id:str
    status:str
    done_list:list[str]
    running_list:list[str]
    document_metadata: dict = Field(default_factory=dict)
    error: str = ""
    error_code: str = ""
    failed_stage: str = ""
    parse_metadata: dict = Field(default_factory=dict)
    index_metadata: dict = Field(default_factory=dict)

from app.agent.tools.schemas import ToolPayload, ToolError
from app.workspace import media


def inspect_uploaded_image(args):
    try:
        document = media.inspect_image(args.image_id, args.question)
        return ToolPayload(status="ok", data={"documents":[document]})
    except ValueError as exc:
        return ToolPayload(status="failed",error=ToolError(code="image_unavailable",message=str(exc)))

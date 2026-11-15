"""可公开给导入页面的解析错误，不携带密钥或第三方原始响应。"""


class MinerUServiceError(RuntimeError):
    def __init__(self, code, message, *, operation, http_status=None, recovery_action="retry"):
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.operation = operation
        self.http_status = http_status
        self.recovery_action = recovery_action

    def public_details(self):
        return {"error_code": self.code, "error_message": self.public_message,
                "request_phase": self.operation, "http_status": self.http_status,
                "recovery_action": self.recovery_action}


def check_mineru_response(response, *, operation):
    """鉴权失败立即停止；不把原响应或请求头放进错误消息。"""
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    status = response.status_code
    code = payload.get("code")
    message = str(payload.get("msg") or payload.get("message") or "").lower()
    expired = code == "A0211" or ("token" in message and "expired" in message)
    if status == 401 or code in ("A0202", "A0211"):
        raise MinerUServiceError(
            "mineru_auth_expired" if expired else "mineru_auth_invalid",
            ("MinerU Token 已过期" if expired else "MinerU Token 无效或鉴权失败")
            + "，PDF 尚未完成解析。请更新服务端 .env 中的 MINERU_API_TOKEN，重启导入服务后重试；原件已保留。",
            operation=operation, http_status=status, recovery_action="update_mineru_token")
    if status == 403:
        raise MinerUServiceError("mineru_forbidden", "MinerU 拒绝访问，请检查账号及解析任务权限；原件已保留。",
                                operation=operation, http_status=status, recovery_action="check_mineru_permission")
    if status != 200:
        raise MinerUServiceError("mineru_http_error", f"MinerU 解析服务请求失败（HTTP {status}），原件已保留，请稍后重试。",
                                operation=operation, http_status=status)
    if code != 0:
        raise MinerUServiceError("mineru_api_error", "MinerU 未接受本次解析请求，原件已保留，请检查解析服务后重试。",
                                operation=operation, http_status=status)
    return payload

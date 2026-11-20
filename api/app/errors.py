class ApiError(Exception):
    def __init__(self, status_code, code, message, detail=None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.detail = detail or {}

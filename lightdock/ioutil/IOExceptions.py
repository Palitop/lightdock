"""New exceptions related to IO operations."""


class UnsupportedFileTypeError(Exception):
    def __init__(self, file_type: str):
        self.file_type = file_type
        self.message = f"Unsupported file type: {self.file_type}"
        super().__init__(self.message)


class EmptyPathError(Exception):
    def __init__(self):
        self.message = "Empty path"
        super().__init__(self.message)


class FolderPathError(Exception):
    def __init__(self):
        self.message = "Provider path is a folder, please check the path"
        super().__init__(self.message)

"""New exceptions related to IO operations."""


class UnsupportedFileTypeError(Exception):
    def __init__(self, file_type: str):
        self.file_type = file_type
        self.message = f"Unsupported file type: {self.file_type}"
        super().__init__(self.message)


class EmptyFileNameError(Exception):
    def __init__(self):
        self.message = "Please provide a non empty file name"
        super().__init__(self.message)

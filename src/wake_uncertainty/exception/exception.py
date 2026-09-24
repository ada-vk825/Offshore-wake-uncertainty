import sys


class CustomException(Exception):
    def __init__(self, error_message:str, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message
        _, _, exc_tb = error_detail.exc_info()  # gives error type, error message and traceback object
        self.line_number = exc_tb.tb_lineno  # gives the line number where the exception occurred
        self.file_name = exc_tb.tb_frame.f_code.co_filename  # gives the file name where the exception occurred

    def __str__(self):
        return f"Error occurred in script: {self.file_name} at line number: {self.line_number} with error message: {self.error_message}"

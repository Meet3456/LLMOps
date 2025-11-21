import sys
import traceback
from datetime import datetime
from typing import Optional, cast


class DocumentPortalException(Exception):
    def __init__(self, error_message, error_details: Optional[object] = None):
        # Normalize message
        norm_msg = str(error_message)

        # Resolve exc_info (supports sys module, exception object, or current context)
        exc_type = exc_value = exc_tb = None
        if error_details is None:
            exc_type, exc_value, exc_tb = sys.exc_info()
        else:
            if hasattr(error_details, "exc_info"):  # e.g., sys
                exc_info_obj = cast(sys, error_details)
                exc_type, exc_value, exc_tb = exc_info_obj.exc_info()
            elif isinstance(error_details, BaseException):
                exc_type, exc_value, exc_tb = (
                    type(error_details),
                    error_details,
                    error_details.__traceback__,
                )
            else:
                exc_type, exc_value, exc_tb = sys.exc_info()

        # Walk to the deepest traceback frame
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        self.file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        self.lineno = last_tb.tb_lineno if last_tb else -1
        self.error_message = norm_msg

        # Add readable timestamp - Example: 21_Nov_2025_05:15_PM
        self.timestamp = datetime.now().strftime("%d_%b_%Y_%I:%M_%p")

        # Full pretty traceback
        if exc_type and exc_tb:
            self.traceback_str = "".join(
                traceback.format_exception(exc_type, exc_value, exc_tb)
            )
        else:
            self.traceback_str = ""

        super().__init__(self.__str__())

    def __str__(self):
        base = (
            f"[{self.timestamp}] "
            f"Error in [{self.file_name}] at line [{self.lineno}] | "
            f"Message: {self.error_message}"
        )
        if self.traceback_str:
            return f"{base}\nTraceback:\n{self.traceback_str}"
        return base

    def __repr__(self):
        return (
            f"DocumentPortalException(file={self.file_name!r}, "
            f"line={self.lineno}, message={self.error_message!r}, "
            f"timestamp={self.timestamp!r})"
        )

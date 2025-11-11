"""
Custom exceptions for document processing
"""


class DocumentProcessingError(Exception):
    """Base exception for document processing errors"""
    pass


class PDFCorruptedError(DocumentProcessingError):
    """Raised when PDF file is corrupted or unreadable"""
    pass


class PDFPasswordProtectedError(DocumentProcessingError):
    """Raised when PDF is password protected"""
    pass


class PDFEmptyError(DocumentProcessingError):
    """Raised when PDF has no extractable content"""
    pass


class PDFNoTablesError(DocumentProcessingError):
    """Raised when PDF has no tables to extract"""
    pass


class TableExtractionError(DocumentProcessingError):
    """Raised when table extraction fails"""
    pass


class InvalidPDFFormatError(DocumentProcessingError):
    """Raised when file is not a valid PDF"""
    pass

"""
Celery tasks for background processing

This module contains Celery tasks for asynchronous document processing.
"""
import logging
from typing import Dict, Any
from celery import Task
from app.celery_app import celery_app
from app.services.document_processor import DocumentProcessor
from app.db.session import SessionLocal
from app.models.document import Document

logger = logging.getLogger(__name__)


class CallbackTask(Task):
    """Base task with callbacks for state updates"""

    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds"""
        logger.info(f"Task {task_id} completed successfully")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails"""
        logger.error(f"Task {task_id} failed: {exc}")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried"""
        logger.warning(f"Task {task_id} retrying due to: {exc}")


@celery_app.task(
    bind=True,
    base=CallbackTask,
    name='app.tasks.process_document_task',
    max_retries=3,
    default_retry_delay=60  # Retry after 1 minute
)
def process_document_task(
    self,
    file_path: str,
    document_id: int,
    fund_id: int
) -> Dict[str, Any]:
    """
    Process a PDF document in the background

    This task:
    1. Extracts tables and text from PDF
    2. Classifies tables (capital calls, distributions, adjustments)
    3. Parses and stores data in database
    4. Creates text chunks for vector storage
    5. Updates document status

    Args:
        file_path: Path to the PDF file
        document_id: Database document ID
        fund_id: Fund ID

    Returns:
        Processing result with statistics

    Raises:
        Exception: If processing fails after retries
    """
    try:
        logger.info(f"Starting document processing task for document_id={document_id}")

        # Update document status to processing
        db = SessionLocal()
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.parsing_status = 'processing'
                db.commit()
        finally:
            db.close()

        # Process document
        processor = DocumentProcessor()

        # Use asyncio.run to call async function from sync context
        import asyncio
        result = asyncio.run(processor.process_document(
            file_path=file_path,
            document_id=document_id,
            fund_id=fund_id
        ))

        # Update document status based on result
        db = SessionLocal()
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                if result['status'] == 'completed':
                    document.parsing_status = 'completed'
                elif result['status'] == 'completed_with_warnings':
                    document.parsing_status = 'completed_with_warnings'
                    # Store warnings if needed
                elif result['status'] == 'failed':
                    document.parsing_status = 'failed'
                    document.error_message = result.get('error', 'Unknown error')

                db.commit()
        finally:
            db.close()

        logger.info(f"Document processing completed for document_id={document_id}: {result['status']}")
        return result

    except Exception as exc:
        logger.error(f"Error processing document {document_id}: {exc}", exc_info=True)

        # Update document status to failed
        db = SessionLocal()
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.parsing_status = 'failed'
                document.error_message = str(exc)
                db.commit()
        finally:
            db.close()

        # Retry task if under max_retries
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying task for document_id={document_id} (attempt {self.request.retries + 1}/{self.max_retries})")
            raise self.retry(exc=exc)
        else:
            logger.error(f"Task failed after {self.max_retries} retries for document_id={document_id}")
            raise


@celery_app.task(name='app.tasks.health_check')
def health_check() -> Dict[str, str]:
    """
    Simple health check task

    Returns:
        Status message
    """
    return {'status': 'healthy', 'message': 'Celery worker is running'}

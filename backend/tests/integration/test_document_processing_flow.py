"""
Integration tests for end-to-end document processing

Tests the complete flow from PDF upload to data extraction and storage.
Aligned with Evaluation Criteria: Parsing Accuracy & Functionality
"""
import pytest
import os
from pathlib import Path
from app.services.document_processor import DocumentProcessor
from app.services.table_parser import TableParser


class TestRealPDFProcessing:
    """Integration tests with real PDF files"""

    def setup_method(self):
        """Setup test instance"""
        self.processor = DocumentProcessor()
        self.parser = TableParser()

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY"),
        reason="Requires API key for embeddings"
    )
    async def test_process_sample_pdf(self, sample_pdf_path, postgres_test_fund):
        """Test processing the actual sample PDF file"""
        if not os.path.exists(sample_pdf_path):
            pytest.skip(f"Sample PDF not found at {sample_pdf_path}")

        try:
            result = await self.processor.process_document(
                file_path=sample_pdf_path,
                document_id=1,
                fund_id=postgres_test_fund.id
            )

            # Verify successful processing
            assert result['status'] == 'completed', f"Processing should complete successfully. Got: {result.get('error', 'unknown error')}"
            assert 'statistics' in result, "Should return statistics"
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in str(e) or "rate" in error_msg or "quota" in error_msg:
                pytest.skip(f"API rate limited or quota exceeded: {e}")
            # Print result for debugging
            if 'result' in locals():
                print(f"Result: {result}")
            raise

        stats = result['statistics']

        # Verify pages were processed
        assert stats['pages_processed'] > 0, "Should process at least one page"

        # Verify tables were found
        assert stats['tables_found'] > 0, "Should find tables in sample PDF"
        assert stats['tables_classified'] > 0, "Should classify at least one table"

        # Verify records were extracted
        total_records = (
            stats.get('capital_calls', 0) +
            stats.get('distributions', 0) +
            stats.get('adjustments', 0)
        )
        assert total_records > 0, "Should extract at least one transaction record"

        # Verify text chunks were created
        assert stats['text_chunks'] > 0, "Should create text chunks"
        assert 'text_chunks' in result, "Should include text chunks in result"

        print(f"\n✓ PDF Processing Results:")
        print(f"  Pages: {stats['pages_processed']}")
        print(f"  Tables found: {stats['tables_found']}")
        print(f"  Tables classified: {stats['tables_classified']}")
        print(f"  Capital calls: {stats.get('capital_calls', 0)}")
        print(f"  Distributions: {stats.get('distributions', 0)}")
        print(f"  Adjustments: {stats.get('adjustments', 0)}")
        print(f"  Text chunks: {stats['text_chunks']}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY"),
        reason="Requires API key for embeddings"
    )
    async def test_verify_expected_data_from_sample_pdf(self, sample_pdf_path, postgres_test_fund):
        """Test that expected data is extracted from sample PDF"""
        if not os.path.exists(sample_pdf_path):
            pytest.skip(f"Sample PDF not found at {sample_pdf_path}")

        try:
            result = await self.processor.process_document(
                file_path=sample_pdf_path,
                document_id=1,
                fund_id=postgres_test_fund.id
            )

            assert result['status'] == 'completed', f"Processing should complete. Got: {result.get('error', 'unknown error')}"
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in str(e) or "rate" in error_msg or "quota" in error_msg:
                pytest.skip(f"API rate limited or quota exceeded: {e}")
            raise

        stats = result['statistics']

        # Based on create_sample_pdf.py, we expect:
        # - 4 capital calls totaling $11.5M
        # - 4 distributions totaling $4.3M
        # - 3 adjustments

        assert stats.get('capital_calls', 0) == 4, "Should extract 4 capital calls"
        assert stats.get('distributions', 0) == 4, "Should extract 4 distributions"
        assert stats.get('adjustments', 0) == 3, "Should extract 3 adjustments"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_text_chunks_contain_meaningful_content(self, sample_pdf_path, postgres_test_fund):
        """Test that text chunks contain meaningful content"""
        if not os.path.exists(sample_pdf_path):
            pytest.skip(f"Sample PDF not found at {sample_pdf_path}")

        result = await self.processor.process_document(
            file_path=sample_pdf_path,
            document_id=1,
            fund_id=postgres_test_fund.id
        )

        assert result['status'] in ['completed', 'completed_with_warnings']
        chunks = result.get('text_chunks', [])

        assert len(chunks) > 0, "Should create text chunks"

        # Verify chunks have proper structure
        for chunk in chunks:
            assert 'text' in chunk, "Chunk should have text field"
            assert 'metadata' in chunk, "Chunk should have metadata field"
            assert len(chunk['text']) > 0, "Chunk text should not be empty"
            assert 'document_id' in chunk['metadata'], "Metadata should have document_id"
            assert 'fund_id' in chunk['metadata'], "Metadata should have fund_id"

        # Check if any chunk contains expected keywords
        all_text = ' '.join(chunk['text'].lower() for chunk in chunks)
        expected_keywords = ['fund', 'capital', 'distribution', 'dpi', 'irr']

        found_keywords = [kw for kw in expected_keywords if kw in all_text]
        assert len(found_keywords) >= 3, f"Should find at least 3 expected keywords, found: {found_keywords}"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_special_sections_extracted(self, sample_pdf_path, postgres_test_fund):
        """Test that special sections like definitions are extracted"""
        if not os.path.exists(sample_pdf_path):
            pytest.skip(f"Sample PDF not found at {sample_pdf_path}")

        result = await self.processor.process_document(
            file_path=sample_pdf_path,
            document_id=1,
            fund_id=postgres_test_fund.id
        )

        assert result['status'] in ['completed', 'completed_with_warnings']
        chunks = result.get('text_chunks', [])

        # Look for special section chunks
        section_chunks = [c for c in chunks if c['metadata'].get('chunk_type') == 'section']

        # Sample PDF should have definitions section
        section_names = [c['metadata'].get('section') for c in section_chunks]

        # Should extract at least one special section
        assert len(section_chunks) > 0, "Should extract special sections"

        print(f"\n✓ Special sections extracted: {section_names}")


class TestParsingAccuracy:
    """Test parsing accuracy with known data"""

    def setup_method(self):
        """Setup test instance"""
        self.parser = TableParser()

    @pytest.mark.integration
    def test_capital_call_total_matches_expected(self, sample_capital_call_table):
        """Test that capital call totals match expected values"""
        records = self.parser.parse_capital_call_table(sample_capital_call_table)

        total = sum(r['amount'] for r in records)
        expected_total = 11_500_000.0  # From sample data

        assert total == expected_total, f"Capital call total should be ${expected_total:,.2f}"

    @pytest.mark.integration
    def test_distribution_total_matches_expected(self, sample_distribution_table):
        """Test that distribution totals match expected values"""
        records = self.parser.parse_distribution_table(sample_distribution_table)

        total = sum(r['amount'] for r in records)
        expected_total = 4_300_000.0  # From sample data

        assert total == expected_total, f"Distribution total should be ${expected_total:,.2f}"

    @pytest.mark.integration
    def test_adjustment_total_matches_expected(self, sample_adjustment_table):
        """Test that adjustment totals match expected values"""
        records = self.parser.parse_adjustment_table(sample_adjustment_table)

        total = sum(r['amount'] for r in records)
        expected_total = -450_000.0  # -500k + 100k - 50k

        assert total == expected_total, f"Adjustment total should be ${expected_total:,.2f}"

    @pytest.mark.integration
    def test_net_pic_calculation(self, sample_capital_call_table, sample_adjustment_table):
        """Test Net PIC calculation accuracy"""
        capital_calls = self.parser.parse_capital_call_table(sample_capital_call_table)
        adjustments = self.parser.parse_adjustment_table(sample_adjustment_table)

        total_calls = sum(r['amount'] for r in capital_calls)
        total_adjustments = sum(r['amount'] for r in adjustments)

        net_pic = total_calls + total_adjustments
        expected_net_pic = 11_050_000.0  # $11.5M - $450k

        assert net_pic == expected_net_pic, f"Net PIC should be ${expected_net_pic:,.2f}"

    @pytest.mark.integration
    def test_dpi_calculation(self, sample_capital_call_table, sample_distribution_table, sample_adjustment_table):
        """Test DPI calculation accuracy"""
        capital_calls = self.parser.parse_capital_call_table(sample_capital_call_table)
        distributions = self.parser.parse_distribution_table(sample_distribution_table)
        adjustments = self.parser.parse_adjustment_table(sample_adjustment_table)

        total_calls = sum(r['amount'] for r in capital_calls)
        total_distributions = sum(r['amount'] for r in distributions)
        total_adjustments = sum(r['amount'] for r in adjustments)

        net_pic = total_calls + total_adjustments
        dpi = total_distributions / net_pic if net_pic > 0 else 0

        expected_dpi = 0.39  # 4.3M / 11.05M ≈ 0.389

        assert abs(dpi - expected_dpi) < 0.01, f"DPI should be approximately {expected_dpi}"

    @pytest.mark.integration
    def test_recallable_distribution_count(self, sample_distribution_table):
        """Test counting recallable distributions"""
        records = self.parser.parse_distribution_table(sample_distribution_table)

        recallable_count = sum(1 for r in records if r['is_recallable'])
        recallable_amount = sum(r['amount'] for r in records if r['is_recallable'])

        assert recallable_count == 1, "Should have 1 recallable distribution"
        assert recallable_amount == 2_000_000.0, "Recallable amount should be $2M"


class TestErrorHandling:
    """Test error handling in document processing"""

    def setup_method(self):
        """Setup test instance"""
        self.processor = DocumentProcessor()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_handle_corrupted_pdf(self):
        """Test handling of corrupted PDF file"""
        # Create a fake corrupted file
        corrupted_path = "/tmp/corrupted.pdf"
        with open(corrupted_path, 'w') as f:
            f.write("This is not a valid PDF file")

        try:
            result = await self.processor.process_document(
                file_path=corrupted_path,
                document_id=1,
                fund_id=1
            )

            # Should fail gracefully, not crash
            assert result['status'] == 'failed', "Should mark as failed"
            assert 'error' in result, "Should include error message"

        finally:
            # Cleanup
            if os.path.exists(corrupted_path):
                os.remove(corrupted_path)

    @pytest.mark.integration
    def test_handle_malformed_table_data(self):
        """Test handling of malformed table data"""
        parser = TableParser()

        malformed_tables = [
            [],  # Empty table
            [[]],  # Table with empty row
            [['header']],  # Table with only header
            [['header'], []],  # Table with empty data row
            [['date', 'amount'], ['invalid-date', 'not-a-number']],  # Invalid data
        ]

        for table in malformed_tables:
            # Should not crash, just return empty list
            result = parser.parse_capital_call_table(table)
            assert isinstance(result, list), "Should return list even for malformed data"


class TestEndToEndMetrics:
    """End-to-end tests verifying complete metrics calculation flow"""

    @pytest.mark.integration
    def test_complete_metrics_flow(self, sample_capital_call_table, sample_distribution_table, sample_adjustment_table):
        """Test complete metrics calculation from parsed data"""
        parser = TableParser()

        # Parse all tables
        capital_calls = parser.parse_capital_call_table(sample_capital_call_table)
        distributions = parser.parse_distribution_table(sample_distribution_table)
        adjustments = parser.parse_adjustment_table(sample_adjustment_table)

        # Calculate metrics
        total_capital_called = sum(r['amount'] for r in capital_calls)
        total_distributions = sum(r['amount'] for r in distributions)
        total_adjustments = sum(r['amount'] for r in adjustments)

        net_pic = total_capital_called + total_adjustments
        dpi = total_distributions / net_pic if net_pic > 0 else 0

        # Verify against expected values from sample PDF
        assert total_capital_called == 11_500_000.0
        assert total_distributions == 4_300_000.0
        assert total_adjustments == -450_000.0
        assert net_pic == 11_050_000.0
        assert abs(dpi - 0.389) < 0.01

        print(f"\n✓ Metrics Calculation Results:")
        print(f"  Total Capital Called: ${total_capital_called:,.2f}")
        print(f"  Total Distributions: ${total_distributions:,.2f}")
        print(f"  Total Adjustments: ${total_adjustments:,.2f}")
        print(f"  Net PIC: ${net_pic:,.2f}")
        print(f"  DPI: {dpi:.3f}")

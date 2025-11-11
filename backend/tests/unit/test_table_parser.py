"""
Unit tests for TableParser

Tests table classification, parsing, and data extraction logic.
Aligned with Evaluation Criteria: Code Quality & Parsing Accuracy
"""
import pytest
from datetime import datetime
from decimal import Decimal
from app.services.table_parser import TableParser


class TestTableClassification:
    """Test suite for table classification (scoring system)"""

    def setup_method(self):
        """Setup test instance"""
        self.parser = TableParser()

    def test_classify_capital_call_table(self, sample_capital_call_table, capital_call_context):
        """Test classification of capital call table"""
        result = self.parser.classify_table(sample_capital_call_table, capital_call_context)
        assert result == 'capital_call', "Should classify as capital_call"

    def test_classify_distribution_table(self, sample_distribution_table, distribution_context):
        """Test classification of distribution table"""
        result = self.parser.classify_table(sample_distribution_table, distribution_context)
        assert result == 'distribution', "Should classify as distribution"

    def test_classify_adjustment_table(self, sample_adjustment_table, adjustment_context):
        """Test classification of adjustment table"""
        result = self.parser.classify_table(sample_adjustment_table, adjustment_context)
        assert result == 'adjustment', "Should classify as adjustment"

    def test_classify_unrelated_table(self, sample_unclassified_table):
        """Test that unrelated tables return None"""
        result = self.parser.classify_table(sample_unclassified_table, "")
        assert result is None, "Should return None for unclassified tables"

    def test_classify_empty_table(self):
        """Test classification with empty table"""
        result = self.parser.classify_table([], "")
        assert result is None, "Should return None for empty table"

    def test_classify_table_without_context(self, sample_capital_call_table):
        """Test classification works even without context"""
        result = self.parser.classify_table(sample_capital_call_table, "")
        assert result == 'capital_call', "Should still classify correctly without context"

    def test_scoring_system_capital_wins(self):
        """Test that scoring system picks highest score"""
        # Table with strong capital call indicators
        table = [
            ['Date', 'Capital Call', 'Amount', 'Drawdown'],
            ['2024-01-01', 'Call 1', '$1000000', 'Investment']
        ]
        result = self.parser.classify_table(table, "Capital contributions funded")
        assert result == 'capital_call', "Should classify as capital_call with highest score"


class TestCapitalCallParsing:
    """Test suite for capital call table parsing"""

    def setup_method(self):
        """Setup test instance"""
        self.parser = TableParser()

    def test_parse_capital_call_table(self, sample_capital_call_table, expected_capital_calls):
        """Test parsing of complete capital call table"""
        result = self.parser.parse_capital_call_table(sample_capital_call_table)

        assert len(result) == 4, "Should parse 4 capital call records"

        for i, record in enumerate(result):
            expected = expected_capital_calls[i]
            assert record['call_date'] == expected['call_date'], f"Record {i}: Date mismatch"
            assert record['call_type'] == expected['call_type'], f"Record {i}: Type mismatch"
            assert record['amount'] == expected['amount'], f"Record {i}: Amount mismatch"
            assert record['description'] == expected['description'], f"Record {i}: Description mismatch"

    def test_parse_empty_table(self):
        """Test parsing empty table returns empty list"""
        result = self.parser.parse_capital_call_table([])
        assert result == [], "Should return empty list for empty table"

    def test_parse_table_with_invalid_rows(self):
        """Test parsing table with some invalid rows"""
        table = [
            ['Date', 'Call Number', 'Amount', 'Description'],
            ['2023-01-15', 'Call 1', '$5,000,000', 'Valid row'],
            ['invalid-date', 'Call 2', '$1,000,000', 'Invalid date'],
            ['2023-06-20', 'Call 3', 'invalid', 'Invalid amount'],
            ['2023-09-01', 'Call 4', '$2,000,000', 'Valid row'],
        ]

        result = self.parser.parse_capital_call_table(table)
        assert len(result) == 2, "Should only parse 2 valid rows"

    def test_total_capital_calls_amount(self, sample_capital_call_table):
        """Test total amount calculation"""
        result = self.parser.parse_capital_call_table(sample_capital_call_table)
        total = sum(r['amount'] for r in result)
        assert total == 11500000.0, "Total capital calls should be $11.5M"


class TestDistributionParsing:
    """Test suite for distribution table parsing"""

    def setup_method(self):
        """Setup test instance"""
        self.parser = TableParser()

    def test_parse_distribution_table(self, sample_distribution_table, expected_distributions):
        """Test parsing of complete distribution table"""
        result = self.parser.parse_distribution_table(sample_distribution_table)

        assert len(result) == 4, "Should parse 4 distribution records"

        for i, record in enumerate(result):
            expected = expected_distributions[i]
            assert record['distribution_date'] == expected['distribution_date'], f"Record {i}: Date mismatch"
            assert record['distribution_type'] == expected['distribution_type'], f"Record {i}: Type mismatch"
            assert record['amount'] == expected['amount'], f"Record {i}: Amount mismatch"
            assert record['is_recallable'] == expected['is_recallable'], f"Record {i}: Recallable flag mismatch"

    def test_recallable_flag_parsing(self):
        """Test parsing of recallable flag with various inputs"""
        test_cases = [
            (['Date', 'Type', 'Amount', 'Recallable', 'Desc'],
             ['2024-01-01', 'Distribution', '$1000000', 'Yes', 'Test'],
             True),
            (['Date', 'Type', 'Amount', 'Recallable', 'Desc'],
             ['2024-01-01', 'Distribution', '$1000000', 'No', 'Test'],
             False),
            (['Date', 'Type', 'Amount', 'Recallable', 'Desc'],
             ['2024-01-01', 'Distribution', '$1000000', 'true', 'Test'],
             True),
            (['Date', 'Type', 'Amount', 'Recallable', 'Desc'],
             ['2024-01-01', 'Distribution', '$1000000', '1', 'Test'],
             True),
        ]

        for header, row, expected_recallable in test_cases:
            table = [header, row]
            result = self.parser.parse_distribution_table(table)
            assert len(result) == 1
            assert result[0]['is_recallable'] == expected_recallable

    def test_total_distributions_amount(self, sample_distribution_table):
        """Test total distribution amount calculation"""
        result = self.parser.parse_distribution_table(sample_distribution_table)
        total = sum(r['amount'] for r in result)
        assert total == 4300000.0, "Total distributions should be $4.3M"


class TestAdjustmentParsing:
    """Test suite for adjustment table parsing"""

    def setup_method(self):
        """Setup test instance"""
        self.parser = TableParser()

    def test_parse_adjustment_table(self, sample_adjustment_table, expected_adjustments):
        """Test parsing of complete adjustment table"""
        result = self.parser.parse_adjustment_table(sample_adjustment_table)

        assert len(result) == 3, "Should parse 3 adjustment records"

        for i, record in enumerate(result):
            expected = expected_adjustments[i]
            assert record['adjustment_date'] == expected['adjustment_date'], f"Record {i}: Date mismatch"
            assert record['adjustment_type'] == expected['adjustment_type'], f"Record {i}: Type mismatch"
            assert record['amount'] == expected['amount'], f"Record {i}: Amount mismatch"
            assert record['category'] == expected['category'], f"Record {i}: Category mismatch"

    def test_negative_amounts_parsed_correctly(self):
        """Test that negative amounts are handled correctly"""
        table = [
            ['Date', 'Type', 'Amount', 'Description'],
            ['2024-01-15', 'Adjustment', '-$500,000', 'Negative adjustment'],
            ['2024-02-15', 'Adjustment', '$100,000', 'Positive adjustment'],
        ]

        result = self.parser.parse_adjustment_table(table)
        assert len(result) == 2
        assert result[0]['amount'] == -500000.0, "Should parse negative amount"
        assert result[1]['amount'] == 100000.0, "Should parse positive amount"

    def test_adjustment_category_classification(self):
        """Test adjustment category classification"""
        test_cases = [
            ('Recallable Distribution', 'Recallable Distribution'),
            ('Capital Call Adjustment', 'Capital Call Adjustment'),
            ('Contribution Adjustment', 'Contribution Adjustment'),
            ('Fee Adjustment', 'Fee Adjustment'),
            ('Other Type', 'Other'),
        ]

        for adj_type, expected_category in test_cases:
            category = self.parser._classify_adjustment_category(adj_type)
            assert category == expected_category, f"Category for '{adj_type}' should be '{expected_category}'"


class TestDateParsing:
    """Test suite for date parsing"""

    def setup_method(self):
        """Setup test instance"""
        self.parser = TableParser()

    def test_parse_various_date_formats(self):
        """Test parsing different date formats"""
        test_dates = [
            ('2023-01-15', datetime(2023, 1, 15)),
            ('01/15/2023', datetime(2023, 1, 15)),
            ('1/15/2023', datetime(2023, 1, 15)),
            ('15-01-2023', datetime(2023, 1, 15)),
        ]

        for date_str, expected_date in test_dates:
            result = self.parser._parse_date(date_str)
            assert result == expected_date, f"Failed to parse '{date_str}'"

    def test_parse_invalid_date(self):
        """Test parsing invalid date returns None"""
        result = self.parser._parse_date('invalid-date')
        assert result is None, "Should return None for invalid date"

    def test_parse_empty_date(self):
        """Test parsing empty date returns None"""
        result = self.parser._parse_date('')
        assert result is None, "Should return None for empty date"

    def test_parse_none_date(self):
        """Test parsing None returns None"""
        result = self.parser._parse_date(None)
        assert result is None, "Should return None for None input"


class TestAmountParsing:
    """Test suite for amount parsing"""

    def setup_method(self):
        """Setup test instance"""
        self.parser = TableParser()

    def test_parse_various_amount_formats(self):
        """Test parsing different amount formats"""
        test_amounts = [
            ('$5,000,000', Decimal('5000000')),
            ('5000000', Decimal('5000000')),
            ('$5000000.00', Decimal('5000000')),
            ('€1,000,000', Decimal('1000000')),
            ('1,234.56', Decimal('1234.56')),
        ]

        for amount_str, expected_amount in test_amounts:
            result = self.parser._parse_amount(amount_str)
            assert result == expected_amount, f"Failed to parse '{amount_str}'"

    def test_parse_negative_amounts(self):
        """Test parsing negative amounts"""
        test_amounts = [
            ('-$500,000', Decimal('-500000')),
            ('($500,000)', Decimal('-500000')),  # Accounting notation
            ('-500000', Decimal('-500000')),
        ]

        for amount_str, expected_amount in test_amounts:
            result = self.parser._parse_amount(amount_str, allow_negative=True)
            assert result == expected_amount, f"Failed to parse negative '{amount_str}'"

    def test_parse_negative_rejected_when_not_allowed(self):
        """Test that negative amounts are rejected when not allowed"""
        result = self.parser._parse_amount('-$500,000', allow_negative=False)
        assert result is None, "Should return None for negative when not allowed"

    def test_parse_invalid_amount(self):
        """Test parsing invalid amount returns None"""
        result = self.parser._parse_amount('invalid')
        assert result is None, "Should return None for invalid amount"

    def test_parse_empty_amount(self):
        """Test parsing empty amount returns None"""
        result = self.parser._parse_amount('')
        assert result is None, "Should return None for empty amount"


class TestTextExtraction:
    """Test suite for text section extraction"""

    def setup_method(self):
        """Setup test instance"""
        self.parser = TableParser()

    def test_extract_text_sections(self, sample_text_for_chunking):
        """Test extraction of meaningful text sections"""
        result = self.parser.extract_text_sections(sample_text_for_chunking)

        assert isinstance(result, dict), "Should return dictionary"
        assert 'strategy' in result, "Should extract strategy section"
        assert 'definitions' in result, "Should extract definitions section"
        assert 'summary' in result, "Should extract summary section"

    def test_extracted_sections_not_empty(self, sample_text_for_chunking):
        """Test that extracted sections have content"""
        result = self.parser.extract_text_sections(sample_text_for_chunking)

        for section_name, content in result.items():
            assert len(content) > 20, f"Section '{section_name}' should have substantial content"
            assert isinstance(content, str), f"Section '{section_name}' should be string"

    def test_extract_from_empty_text(self):
        """Test extraction from empty text"""
        result = self.parser.extract_text_sections("")
        assert result == {}, "Should return empty dict for empty text"


class TestColumnFinding:
    """Test suite for column index finding"""

    def setup_method(self):
        """Setup test instance"""
        self.parser = TableParser()

    def test_find_column_exact_match(self):
        """Test finding column with exact match"""
        header = ['date', 'amount', 'description']
        result = self.parser._find_column_index(header, ['date'])
        assert result == 0, "Should find 'date' at index 0"

    def test_find_column_partial_match(self):
        """Test finding column with partial match"""
        header = ['call date', 'call amount', 'description']
        result = self.parser._find_column_index(header, ['date'])
        assert result == 0, "Should find 'date' in 'call date'"

    def test_find_column_multiple_possibilities(self):
        """Test finding column with multiple possible names"""
        header = ['dist date', 'amount', 'description']
        result = self.parser._find_column_index(header, ['date', 'distribution date', 'dist date'])
        assert result == 0, "Should find first matching possibility"

    def test_find_column_not_found(self):
        """Test finding column that doesn't exist"""
        header = ['col1', 'col2', 'col3']
        result = self.parser._find_column_index(header, ['nonexistent'])
        assert result is None, "Should return None when column not found"

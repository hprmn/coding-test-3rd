"""
Table parsing and classification service

This module handles extraction and classification of tables from PDFs,
identifying capital calls, distributions, and adjustments using a scoring system.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
from decimal import Decimal, InvalidOperation
import logging

logger = logging.getLogger(__name__)


class TableParser:
    """Parse and classify tables extracted from fund performance PDFs"""

    # Keywords for table classification (weighted scoring system)
    # Header keywords (high priority - appear in table title/section)
    CAPITAL_CALL_HEADER_KEYWORDS = [
        'capital call', 'capital calls', 'drawdown', 'drawdowns',
        'capital contribution', 'contributions', 'funded capital'
    ]

    CAPITAL_CALL_KEYWORDS = [
        'call number', 'call date', 'investment call', 'commitment',
        'call', 'funded', 'invested'
    ]

    DISTRIBUTION_HEADER_KEYWORDS = [
        'distribution', 'distributions', 'distribution to lp',
        'distributions to limited partners'
    ]

    DISTRIBUTION_KEYWORDS = [
        'return of capital', 'dividend', 'proceeds', 'payout',
        'returned', 'dist', 'distribution date', 'distribution type'
    ]

    ADJUSTMENT_HEADER_KEYWORDS = [
        'adjustment', 'adjustments', 'rebalancing'
    ]

    ADJUSTMENT_KEYWORDS = [
        'recallable', 'recall', 'rebalance', 'correction',
        'fee adjustment', 'adjustment date', 'adjustment type'
    ]

    # Common date formats
    DATE_PATTERNS = [
        r'\d{4}-\d{2}-\d{2}',  # 2023-01-15
        r'\d{2}/\d{2}/\d{4}',  # 01/15/2023
        r'\d{1,2}-\d{1,2}-\d{4}',  # 1-15-2023
        r'\d{1,2}/\d{1,2}/\d{4}',  # 1/15/2023
    ]

    def classify_table(self, table_data: List[List[str]], context: str = "") -> Optional[str]:
        """
        Classify a table as capital_call, distribution, or adjustment using weighted keyword scoring

        Uses weighted scoring where:
        - Context keywords: 2x weight (section headers)
        - Table header keywords (specific): 3x weight
        - Table header keywords (general): 2x weight
        - Table body keywords: 1x weight

        Args:
            table_data: 2D list of table cells
            context: Surrounding text context for better classification

        Returns:
            Table type: 'capital_call', 'distribution', 'adjustment', or None
        """
        if not table_data or len(table_data) < 2:
            return None

        # Extract different parts of the table
        context_lower = context.lower()
        header = ' '.join(str(cell).lower() for cell in table_data[0] if cell)
        body = ' '.join(' '.join(str(cell).lower() for cell in row if cell) for row in table_data[1:])

        # Weighted scoring for each category
        # Context (section header) - 2x weight
        capital_score = sum(2 for kw in self.CAPITAL_CALL_HEADER_KEYWORDS if kw in context_lower)
        distribution_score = sum(2 for kw in self.DISTRIBUTION_HEADER_KEYWORDS if kw in context_lower)
        adjustment_score = sum(2 for kw in self.ADJUSTMENT_HEADER_KEYWORDS if kw in context_lower)

        # Table header - Header keywords get 3x weight
        capital_score += sum(3 for kw in self.CAPITAL_CALL_HEADER_KEYWORDS if kw in header)
        distribution_score += sum(3 for kw in self.DISTRIBUTION_HEADER_KEYWORDS if kw in header)
        adjustment_score += sum(3 for kw in self.ADJUSTMENT_HEADER_KEYWORDS if kw in header)

        # Table header - General keywords get 2x weight
        capital_score += sum(2 for kw in self.CAPITAL_CALL_KEYWORDS if kw in header)
        distribution_score += sum(2 for kw in self.DISTRIBUTION_KEYWORDS if kw in header)
        adjustment_score += sum(2 for kw in self.ADJUSTMENT_KEYWORDS if kw in header)

        # Table body - Regular keywords get 1x weight
        capital_score += sum(1 for kw in self.CAPITAL_CALL_KEYWORDS if kw in body)
        distribution_score += sum(1 for kw in self.DISTRIBUTION_KEYWORDS if kw in body)
        adjustment_score += sum(1 for kw in self.ADJUSTMENT_KEYWORDS if kw in body)

        logger.debug(f"Table classification scores (weighted) - Capital: {capital_score}, Distribution: {distribution_score}, Adjustment: {adjustment_score}")

        # Determine table type based on highest score
        scores = {
            'capital_call': capital_score,
            'distribution': distribution_score,
            'adjustment': adjustment_score
        }

        max_score = max(scores.values())
        if max_score == 0:
            logger.debug("No matching keywords found, table classification failed")
            return None

        # Return the type with highest score
        # If there's a tie, prefer in order: adjustment > capital_call > distribution
        if adjustment_score == max_score and adjustment_score > 0:
            logger.info(f"Table classified as: adjustment (score: {adjustment_score})")
            return 'adjustment'
        elif capital_score == max_score and capital_score > 0:
            logger.info(f"Table classified as: capital_call (score: {capital_score})")
            return 'capital_call'
        elif distribution_score == max_score and distribution_score > 0:
            logger.info(f"Table classified as: distribution (score: {distribution_score})")
            return 'distribution'

        return None

    def parse_capital_call_table(self, table_data: List[List[str]]) -> List[Dict[str, Any]]:
        """
        Parse capital call table into structured records

        Expected columns: Date, Call Number/Type, Amount, Description

        Returns:
            List of capital call records
        """
        if len(table_data) < 2:
            return []

        header = [str(cell).lower().strip() if cell else '' for cell in table_data[0]]

        # Find column indices
        date_col = self._find_column_index(header, ['date', 'call date'])
        amount_col = self._find_column_index(header, ['amount', 'capital', 'called'])
        type_col = self._find_column_index(header, ['call', 'number', 'type', 'call number'])
        desc_col = self._find_column_index(header, ['description', 'desc', 'note', 'purpose'])

        records = []
        for row_idx, row in enumerate(table_data[1:], start=1):
            if not row or len(row) == 0:
                continue

            try:
                # Extract date
                call_date = self._parse_date(row[date_col] if date_col is not None else None)
                if not call_date:
                    logger.debug(f"Skipping row {row_idx}: Invalid date")
                    continue

                # Extract amount
                amount = self._parse_amount(row[amount_col] if amount_col is not None else None)
                if amount is None or amount == 0:
                    logger.debug(f"Skipping row {row_idx}: Invalid amount")
                    continue

                # Extract call type
                call_type = str(row[type_col]).strip() if type_col is not None and type_col < len(row) else 'Capital Call'

                # Extract description
                description = str(row[desc_col]).strip() if desc_col is not None and desc_col < len(row) else ''

                records.append({
                    'call_date': call_date,
                    'call_type': call_type,
                    'amount': float(amount),
                    'description': description
                })

            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Error parsing capital call row {row_idx}: {e}")
                continue

        logger.info(f"Parsed {len(records)} capital call records")
        return records

    def parse_distribution_table(self, table_data: List[List[str]]) -> List[Dict[str, Any]]:
        """
        Parse distribution table into structured records

        Expected columns: Date, Type, Amount, Recallable, Description

        Returns:
            List of distribution records
        """
        if len(table_data) < 2:
            return []

        header = [str(cell).lower().strip() if cell else '' for cell in table_data[0]]

        # Find column indices
        date_col = self._find_column_index(header, ['date', 'distribution date', 'dist date'])
        amount_col = self._find_column_index(header, ['amount', 'distributed', 'distribution'])
        type_col = self._find_column_index(header, ['type', 'distribution type', 'category'])
        recallable_col = self._find_column_index(header, ['recallable', 'recall'])
        desc_col = self._find_column_index(header, ['description', 'desc', 'note'])

        records = []
        for row_idx, row in enumerate(table_data[1:], start=1):
            if not row or len(row) == 0:
                continue

            try:
                # Extract date
                dist_date = self._parse_date(row[date_col] if date_col is not None else None)
                if not dist_date:
                    logger.debug(f"Skipping row {row_idx}: Invalid date")
                    continue

                # Extract amount
                amount = self._parse_amount(row[amount_col] if amount_col is not None else None)
                if amount is None or amount == 0:
                    logger.debug(f"Skipping row {row_idx}: Invalid amount")
                    continue

                # Extract type
                dist_type = str(row[type_col]).strip() if type_col is not None and type_col < len(row) else 'Distribution'

                # Extract recallable status
                is_recallable = False
                if recallable_col is not None and recallable_col < len(row):
                    recallable_text = str(row[recallable_col]).lower().strip()
                    is_recallable = recallable_text in ['yes', 'true', 'y', '1']

                # Extract description
                description = str(row[desc_col]).strip() if desc_col is not None and desc_col < len(row) else ''

                records.append({
                    'distribution_date': dist_date,
                    'distribution_type': dist_type,
                    'amount': float(amount),
                    'is_recallable': is_recallable,
                    'description': description
                })

            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Error parsing distribution row {row_idx}: {e}")
                continue

        logger.info(f"Parsed {len(records)} distribution records")
        return records

    def parse_adjustment_table(self, table_data: List[List[str]]) -> List[Dict[str, Any]]:
        """
        Parse adjustment table into structured records

        Expected columns: Date, Type, Amount, Description

        Returns:
            List of adjustment records
        """
        if len(table_data) < 2:
            return []

        header = [str(cell).lower().strip() if cell else '' for cell in table_data[0]]

        # Find column indices
        date_col = self._find_column_index(header, ['date', 'adjustment date'])
        amount_col = self._find_column_index(header, ['amount', 'adjustment'])
        type_col = self._find_column_index(header, ['type', 'adjustment type', 'category'])
        desc_col = self._find_column_index(header, ['description', 'desc', 'note'])

        records = []
        for row_idx, row in enumerate(table_data[1:], start=1):
            if not row or len(row) == 0:
                continue

            try:
                # Extract date
                adj_date = self._parse_date(row[date_col] if date_col is not None else None)
                if not adj_date:
                    logger.debug(f"Skipping row {row_idx}: Invalid date")
                    continue

                # Extract amount (can be negative)
                amount = self._parse_amount(row[amount_col] if amount_col is not None else None, allow_negative=True)
                if amount is None:
                    logger.debug(f"Skipping row {row_idx}: Invalid amount")
                    continue

                # Extract type
                adj_type = str(row[type_col]).strip() if type_col is not None and type_col < len(row) else 'Adjustment'

                # Determine category and is_contribution_adjustment flag
                category = self._classify_adjustment_category(adj_type)
                is_contribution_adj = 'capital call' in adj_type.lower() or 'contribution' in adj_type.lower()

                # Extract description
                description = str(row[desc_col]).strip() if desc_col is not None and desc_col < len(row) else ''

                records.append({
                    'adjustment_date': adj_date,
                    'adjustment_type': adj_type,
                    'category': category,
                    'amount': float(amount),
                    'is_contribution_adjustment': is_contribution_adj,
                    'description': description
                })

            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Error parsing adjustment row {row_idx}: {e}")
                continue

        logger.info(f"Parsed {len(records)} adjustment records")
        return records

    def _find_column_index(self, header: List[str], possible_names: List[str]) -> Optional[int]:
        """Find column index by matching possible column names"""
        for i, col_name in enumerate(header):
            for possible_name in possible_names:
                if possible_name in col_name:
                    return i
        return None

    def _parse_date(self, date_str: Any) -> Optional[datetime]:
        """Parse date from various formats"""
        if not date_str:
            return None

        date_str = str(date_str).strip()

        # Try common date formats
        date_formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y/%m/%d',
            '%m-%d-%Y',
            '%d-%m-%Y',
            '%B %d, %Y',
            '%b %d, %Y',
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # Try to extract date using regex
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, date_str)
            if match:
                date_text = match.group(0)
                # Try parsing again with extracted text
                for fmt in date_formats:
                    try:
                        return datetime.strptime(date_text, fmt)
                    except ValueError:
                        continue

        return None

    def _parse_amount(self, amount_str: Any, allow_negative: bool = False) -> Optional[Decimal]:
        """Parse monetary amount from string"""
        if not amount_str:
            return None

        amount_str = str(amount_str).strip()

        # Remove currency symbols and commas
        amount_str = re.sub(r'[$€£¥,\s]', '', amount_str)

        # Handle parentheses as negative (accounting notation)
        is_negative = False
        if '(' in amount_str and ')' in amount_str:
            is_negative = True
            amount_str = amount_str.replace('(', '').replace(')', '')

        # Check for minus sign
        if amount_str.startswith('-'):
            is_negative = True
            amount_str = amount_str[1:]

        try:
            amount = Decimal(amount_str)
            if is_negative:
                amount = -amount

            if not allow_negative and amount < 0:
                return None

            return amount
        except (ValueError, TypeError, InvalidOperation):
            return None

    def _classify_adjustment_category(self, adjustment_type: str) -> str:
        """Classify adjustment into a category"""
        adj_type_lower = adjustment_type.lower()

        if 'recallable' in adj_type_lower or 'recall' in adj_type_lower:
            return 'Recallable Distribution'
        elif 'capital call' in adj_type_lower:
            return 'Capital Call Adjustment'
        elif 'contribution' in adj_type_lower:
            return 'Contribution Adjustment'
        elif 'fee' in adj_type_lower:
            return 'Fee Adjustment'
        else:
            return 'Other'

    def extract_fund_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract fund metadata from PDF text

        Looks for fund information like:
        - Fund Name
        - GP (General Partner) Name
        - Vintage Year
        - Fund Type/Strategy

        Args:
            text: Full text content from PDF

        Returns:
            Dictionary with fund metadata
        """
        metadata = {}

        # Extract Fund Name
        fund_name_patterns = [
            r'(?i)fund\s+name[\s:]+(.+?)(?=\n|$)',
            r'(?i)^(.+?)\s*(?:fund|lp|gp)\s+(?:i{1,3}|iv|v|vi{0,3}|[0-9]+)\s*$',  # Match "Tech Ventures Fund III"
        ]

        for pattern in fund_name_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                fund_name = match.group(1).strip()
                if fund_name and len(fund_name) > 2:
                    metadata['name'] = fund_name
                    break

        # If fund name not found via patterns, use first line if it looks like a fund name
        if 'name' not in metadata:
            first_line = text.split('\n')[0].strip()
            if len(first_line) < 100 and ('fund' in first_line.lower() or 'ventures' in first_line.lower() or 'capital' in first_line.lower()):
                metadata['name'] = first_line

        # Extract GP Name
        gp_patterns = [
            r'(?i)gp[\s:]+(.+?)(?=\n|$)',
            r'(?i)general partner[\s:]+(.+?)(?=\n|$)',
            r'(?i)fund manager[\s:]+(.+?)(?=\n|$)',
        ]

        for pattern in gp_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                gp_name = match.group(1).strip()
                if gp_name and len(gp_name) > 2:
                    metadata['gp_name'] = gp_name
                    break

        # Extract Vintage Year
        vintage_patterns = [
            r'(?i)vintage\s+year[\s:]+(\d{4})',
            r'(?i)vintage[\s:]+(\d{4})',
            r'(?i)inception[\s:]+(\d{4})',
        ]

        for pattern in vintage_patterns:
            match = re.search(pattern, text)
            if match:
                year = int(match.group(1))
                if 1990 <= year <= 2030:  # Sanity check
                    metadata['vintage_year'] = year
                    break

        # Extract Fund Type/Strategy
        strategy_patterns = [
            r'(?i)fund strategy[\s:]+(.+?)(?=\n\n|\n[A-Z][a-z]+:|$)',
            r'(?i)investment strategy[\s:]+(.+?)(?=\n\n|\n[A-Z][a-z]+:|$)',
        ]

        for pattern in strategy_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                strategy = match.group(1).strip()
                if strategy and len(strategy) > 10:
                    # Extract first sentence or first 200 chars for fund_type
                    first_sentence = re.split(r'[.!?]', strategy)[0].strip()
                    if len(first_sentence) < 200:
                        metadata['fund_type'] = first_sentence
                    else:
                        metadata['fund_type'] = strategy[:200].strip()
                    break

        logger.info(f"Extracted fund metadata: {metadata}")
        return metadata

    def extract_text_sections(self, text: str) -> Dict[str, str]:
        """
        Extract meaningful text sections for vector storage

        Args:
            text: Full text content from PDF

        Returns:
            Dictionary of section_name -> section_content
        """
        sections = {}

        # Common section headers
        section_patterns = [
            (r'(?i)(fund strategy|investment strategy|strategy)[\s:]+(.+?)(?=\n\n|\n[A-Z]|$)', 'strategy'),
            (r'(?i)(key definitions|definitions|glossary)[\s:]+(.+?)(?=\n\n|\n[A-Z]|$)', 'definitions'),
            (r'(?i)(performance summary|summary)[\s:]+(.+?)(?=\n\n|\n[A-Z]|$)', 'summary'),
            (r'(?i)(fund information|fund details)[\s:]+(.+?)(?=\n\n|\n[A-Z]|$)', 'fund_info'),
        ]

        for pattern, section_name in section_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                content = match.group(2).strip() if len(match.groups()) > 1 else match.group(1).strip()
                if content and len(content) > 20:  # Minimum content length
                    sections[section_name] = content

        return sections

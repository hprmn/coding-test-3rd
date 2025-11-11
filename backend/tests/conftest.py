"""
Pytest configuration and shared fixtures

This module contains pytest fixtures that are shared across all tests.
"""
import pytest
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add app to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


@pytest.fixture
def sample_capital_call_table() -> List[List[str]]:
    """Sample capital call table data"""
    return [
        ['Date', 'Call Number', 'Amount', 'Description'],
        ['2023-01-15', 'Call 1', '$5,000,000', 'Initial Capital Call'],
        ['2023-06-20', 'Call 2', '$3,000,000', 'Follow-on Investment'],
        ['2024-03-10', 'Call 3', '$2,000,000', 'Bridge Round Funding'],
        ['2024-09-15', 'Call 4', '$1,500,000', 'Additional Capital'],
    ]


@pytest.fixture
def sample_distribution_table() -> List[List[str]]:
    """Sample distribution table data"""
    return [
        ['Date', 'Type', 'Amount', 'Recallable', 'Description'],
        ['2023-12-15', 'Return of Capital', '$1,500,000', 'No', 'Exit: TechCo Inc'],
        ['2024-06-20', 'Income', '$500,000', 'No', 'Dividend Payment'],
        ['2024-09-10', 'Return of Capital', '$2,000,000', 'Yes', 'Partial Exit: DataCorp'],
        ['2024-12-20', 'Income', '$300,000', 'No', 'Year-end Distribution'],
    ]


@pytest.fixture
def sample_adjustment_table() -> List[List[str]]:
    """Sample adjustment table data"""
    return [
        ['Date', 'Type', 'Amount', 'Description'],
        ['2024-01-15', 'Recallable Distribution', '-$500,000', 'Recalled distribution from Q4 2023'],
        ['2024-03-20', 'Capital Call Adjustment', '$100,000', 'Management fee adjustment'],
        ['2024-07-10', 'Contribution Adjustment', '-$50,000', 'Expense reimbursement'],
    ]


@pytest.fixture
def sample_unclassified_table() -> List[List[str]]:
    """Sample table that should not be classified"""
    return [
        ['Name', 'Email', 'Phone'],
        ['John Doe', 'john@example.com', '555-1234'],
        ['Jane Smith', 'jane@example.com', '555-5678'],
    ]


@pytest.fixture
def capital_call_context() -> str:
    """Context text for capital call table"""
    return """
    Capital Calls

    The following table shows all capital calls made to Limited Partners during the reporting period.
    Each call represents a drawdown on committed capital for investment purposes.
    """


@pytest.fixture
def distribution_context() -> str:
    """Context text for distribution table"""
    return """
    Distributions to Limited Partners

    The table below summarizes all distributions made to LPs during the quarter.
    This includes both return of capital and income distributions.
    """


@pytest.fixture
def adjustment_context() -> str:
    """Context text for adjustment table"""
    return """
    Adjustments and Rebalancing

    The following adjustments were made during the period to reflect
    recallable distributions and other capital account corrections.
    """


@pytest.fixture
def sample_text_for_chunking() -> str:
    """Sample text for testing text chunking"""
    return """
    Fund Strategy: The fund focuses on early-stage technology companies in the SaaS,
    fintech, and AI sectors. Our investment thesis centers on identifying companies with
    strong product-market fit and scalable business models. We typically invest between
    $2M to $10M in Series A and Series B rounds.

    Key Definitions:

    DPI (Distributions to Paid-In): Total distributions divided by total paid-in capital.
    Measures cash returned to investors. A DPI of 1.0 means investors have received their
    original investment back. DPI above 1.0 indicates profits have been distributed.

    IRR (Internal Rate of Return): The annualized rate of return that makes the net
    present value of all cash flows equal to zero. IRR accounts for the timing of cash
    flows and provides a time-weighted measure of fund performance.

    TVPI (Total Value to Paid-In): The sum of distributions and residual value divided
    by paid-in capital. Measures total value creation including both realized and
    unrealized gains.

    Performance Summary: As of Q4 2024, the fund has called $11.5M in capital and
    distributed $4.3M to Limited Partners. The current DPI stands at 0.39, with an
    IRR of 12.5%. The fund has made 15 investments across 12 portfolio companies.
    """


@pytest.fixture
def expected_capital_calls() -> List[Dict[str, Any]]:
    """Expected parsed capital call records"""
    return [
        {
            'call_date': datetime(2023, 1, 15),
            'call_type': 'Call 1',
            'amount': 5000000.0,
            'description': 'Initial Capital Call'
        },
        {
            'call_date': datetime(2023, 6, 20),
            'call_type': 'Call 2',
            'amount': 3000000.0,
            'description': 'Follow-on Investment'
        },
        {
            'call_date': datetime(2024, 3, 10),
            'call_type': 'Call 3',
            'amount': 2000000.0,
            'description': 'Bridge Round Funding'
        },
        {
            'call_date': datetime(2024, 9, 15),
            'call_type': 'Call 4',
            'amount': 1500000.0,
            'description': 'Additional Capital'
        },
    ]


@pytest.fixture
def expected_distributions() -> List[Dict[str, Any]]:
    """Expected parsed distribution records"""
    return [
        {
            'distribution_date': datetime(2023, 12, 15),
            'distribution_type': 'Return of Capital',
            'amount': 1500000.0,
            'is_recallable': False,
            'description': 'Exit: TechCo Inc'
        },
        {
            'distribution_date': datetime(2024, 6, 20),
            'distribution_type': 'Income',
            'amount': 500000.0,
            'is_recallable': False,
            'description': 'Dividend Payment'
        },
        {
            'distribution_date': datetime(2024, 9, 10),
            'distribution_type': 'Return of Capital',
            'amount': 2000000.0,
            'is_recallable': True,
            'description': 'Partial Exit: DataCorp'
        },
        {
            'distribution_date': datetime(2024, 12, 20),
            'distribution_type': 'Income',
            'amount': 300000.0,
            'is_recallable': False,
            'description': 'Year-end Distribution'
        },
    ]


@pytest.fixture
def expected_adjustments() -> List[Dict[str, Any]]:
    """Expected parsed adjustment records"""
    return [
        {
            'adjustment_date': datetime(2024, 1, 15),
            'adjustment_type': 'Recallable Distribution',
            'amount': -500000.0,
            'category': 'Recallable Distribution',
            'is_contribution_adjustment': False,
            'description': 'Recalled distribution from Q4 2023'
        },
        {
            'adjustment_date': datetime(2024, 3, 20),
            'adjustment_type': 'Capital Call Adjustment',
            'amount': 100000.0,
            'category': 'Capital Call Adjustment',
            'is_contribution_adjustment': True,
            'description': 'Management fee adjustment'
        },
        {
            'adjustment_date': datetime(2024, 7, 10),
            'adjustment_type': 'Contribution Adjustment',
            'amount': -50000.0,
            'category': 'Contribution Adjustment',
            'is_contribution_adjustment': True,
            'description': 'Expense reimbursement'
        },
    ]


@pytest.fixture
def sample_pdf_path() -> str:
    """Path to sample PDF for integration tests"""
    return str(Path(__file__).parent.parent.parent / "files" / "Sample_Fund_Performance_Report.pdf")


@pytest.fixture(scope="function")
def test_db():
    """Setup SQLite in-memory database for testing"""
    from tests.test_db import init_test_db, reset_test_db, cleanup_test_db, get_test_db

    # Initialize database
    init_test_db()

    # Get database session
    db = get_test_db()

    yield db

    # Cleanup
    db.close()
    cleanup_test_db()


@pytest.fixture(scope="function")
def test_fund(test_db):
    """Create a test fund in the database"""
    from app.models import Fund

    fund = Fund(
        name="Test Fund",
        gp_name="Test GP",
        fund_type="Venture Capital",
        vintage_year=2023
    )
    test_db.add(fund)
    test_db.commit()
    test_db.refresh(fund)

    return fund


@pytest.fixture(scope="function")
def postgres_test_fund():
    """Create a test fund in PostgreSQL database for integration tests"""
    from app.db.session import SessionLocal
    from app.models import Fund, CapitalCall, Distribution, Adjustment

    db = SessionLocal()
    try:
        # Create test fund
        fund = Fund(
            name="Test Fund",
            gp_name="Test GP",
            fund_type="Venture Capital",
            vintage_year=2023
        )
        db.add(fund)
        db.commit()
        db.refresh(fund)

        yield fund

        # Cleanup - delete all related records
        db.query(CapitalCall).filter(CapitalCall.fund_id == fund.id).delete()
        db.query(Distribution).filter(Distribution.fund_id == fund.id).delete()
        db.query(Adjustment).filter(Adjustment.fund_id == fund.id).delete()
        db.query(Fund).filter(Fund.id == fund.id).delete()
        db.commit()
    finally:
        db.close()

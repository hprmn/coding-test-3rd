"""
Multi-fund comparison API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from app.db.session import get_db
from app.models.fund import Fund
from app.services.metrics_calculator import MetricsCalculator

router = APIRouter()


@router.post("/compare")
async def compare_funds(
    fund_ids: List[int] = Query(..., description="List of fund IDs to compare"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Compare multiple funds side-by-side

    Returns:
        Comparison data with metrics for each fund
    """
    if len(fund_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 funds are required for comparison"
        )

    if len(fund_ids) > 5:
        raise HTTPException(
            status_code=400,
            detail="Maximum 5 funds can be compared at once"
        )

    calculator = MetricsCalculator(db)
    comparison_data = []

    for fund_id in fund_ids:
        # Get fund details
        fund = db.query(Fund).filter(Fund.id == fund_id).first()

        if not fund:
            raise HTTPException(
                status_code=404,
                detail=f"Fund with ID {fund_id} not found"
            )

        # Calculate metrics
        metrics = calculator.calculate_all_metrics(fund_id)

        comparison_data.append({
            "fund_id": fund_id,
            "fund_name": fund.name,
            "gp_name": fund.gp_name,
            "fund_type": fund.fund_type,
            "vintage_year": fund.vintage_year,
            "metrics": {
                "dpi": metrics.get("dpi", 0),
                "irr": metrics.get("irr", 0),
                "pic": metrics.get("pic", 0),
                "total_distributions": metrics.get("total_distributions", 0)
            }
        })

    # Calculate averages
    avg_dpi = sum(f["metrics"]["dpi"] for f in comparison_data) / len(comparison_data)
    avg_irr = sum(f["metrics"]["irr"] for f in comparison_data) / len(comparison_data)
    avg_pic = sum(f["metrics"]["pic"] for f in comparison_data) / len(comparison_data)

    # Rank funds by DPI
    sorted_by_dpi = sorted(
        comparison_data,
        key=lambda x: x["metrics"]["dpi"],
        reverse=True
    )

    # Rank funds by IRR
    sorted_by_irr = sorted(
        comparison_data,
        key=lambda x: x["metrics"]["irr"],
        reverse=True
    )

    return {
        "funds": comparison_data,
        "averages": {
            "dpi": round(avg_dpi, 4),
            "irr": round(avg_irr, 2),
            "pic": round(avg_pic, 2)
        },
        "rankings": {
            "by_dpi": [f["fund_id"] for f in sorted_by_dpi],
            "by_irr": [f["fund_id"] for f in sorted_by_irr]
        },
        "insights": {
            "best_dpi": {
                "fund_id": sorted_by_dpi[0]["fund_id"],
                "fund_name": sorted_by_dpi[0]["fund_name"],
                "dpi": sorted_by_dpi[0]["metrics"]["dpi"]
            },
            "best_irr": {
                "fund_id": sorted_by_irr[0]["fund_id"],
                "fund_name": sorted_by_irr[0]["fund_name"],
                "irr": sorted_by_irr[0]["metrics"]["irr"]
            }
        }
    }

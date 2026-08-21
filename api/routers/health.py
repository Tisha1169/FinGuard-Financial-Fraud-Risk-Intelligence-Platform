from fastapi import APIRouter

from database.connection import check_connection

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    healthy, message = check_connection()
    return {"status": "ok" if healthy else "degraded", "database": message}

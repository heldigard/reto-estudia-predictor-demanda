from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.forecasting import build_payload, write_frontend_artifacts, write_payload


def main() -> None:
    payload = build_payload()
    write_payload(payload)
    write_frontend_artifacts(payload)
    print("Precomputed artifacts generated for backend and frontend.")


if __name__ == "__main__":
    main()

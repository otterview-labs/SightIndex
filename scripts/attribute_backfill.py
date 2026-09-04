import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def main() -> None:
    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.services.attribute_backfill import DurableAttributeBackfillService

    parser = argparse.ArgumentParser(
        description="Durably upgrade legacy person-crop tags with structured VLM attributes."
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 500:
        parser.error("--batch-size must be between 1 and 500")
    if args.max_attempts < 1 or args.max_attempts > 10:
        parser.error("--max-attempts must be between 1 and 10")

    with SessionLocal() as db:
        progress = DurableAttributeBackfillService(db, get_settings()).run(
            batch_size=args.batch_size,
            max_attempts=args.max_attempts,
        )
    print(json.dumps(progress.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

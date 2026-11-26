"""Check the PostgreSQL connection. Run as a module from anpr-backend:

    python -m app.test_db

Exits 0 on success, 1 on failure, so a script can trust the exit code.
"""

import sys

from app.database import engine


def main():
    try:
        with engine.connect():
            print("PostgreSQL connected successfully!")
    except Exception as error:
        print("Database connection failed:")
        print(error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Create the database tables (once). Run as a module from anpr-backend:

    python -m app.create_tables
"""

from app.database import engine, Base
from app.models import Detection, Alert  # noqa: F401 - registers the models with Base


def main():
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")


if __name__ == "__main__":
    main()

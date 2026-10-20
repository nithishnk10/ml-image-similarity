from database import get_postgres_connection


CREATE_TABLE_QUERY = """
CREATE TABLE IF NOT EXISTS images (
    id BIGSERIAL PRIMARY KEY,
    qdrant_id BIGINT UNIQUE NOT NULL,
    image_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    artist TEXT,
    style TEXT
);
"""


def create_table():
    connection = get_postgres_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_TABLE_QUERY)

        connection.commit()
        print("PostgreSQL table 'images' created successfully.")

    finally:
        connection.close()


if __name__ == "__main__":
    create_table()
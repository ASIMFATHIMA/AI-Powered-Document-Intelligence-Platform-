
import os
import mysql.connector
from datetime import datetime, timezone
from mysql.connector import Error


def get_connection():
    """Create a connection to the MySQL database."""
    password = os.getenv("DB_PASSWORD")
    if password is None:
        raise RuntimeError("DB_PASSWORD environment variable is required.")

    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=password,
        database=os.getenv("DB_NAME", "document_intelligence")
    )


def init_db():
    """Create the required tables if they do not exist."""
    connection = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                text_length INT NOT NULL,
                uploaded_at VARCHAR(40) NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INT AUTO_INCREMENT PRIMARY KEY,
                document_id INT,
                rating VARCHAR(30) NOT NULL,
                comment TEXT,
                created_at VARCHAR(40) NOT NULL,
                FOREIGN KEY (document_id)
                    REFERENCES documents(id)
                    ON DELETE SET NULL
            )
        """)

        connection.commit()

    except Error as e:
        raise RuntimeError(
            f"MySQL initialization failed: {e}"
        ) from e

    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()


def save_document(filename, text_length):
    """Save document metadata and return its ID."""
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO documents
                (filename, text_length, uploaded_at)
            VALUES (%s, %s, %s)
        """, (
            filename,
            text_length,
            datetime.now(timezone.utc).isoformat()
        ))

        connection.commit()
        return cursor.lastrowid

    except Error as e:
        if connection:
            connection.rollback()
        raise RuntimeError(f"Could not save document: {e}") from e

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


def save_feedback(document_id, rating, comment=""):
    """Save feedback for a processed document."""
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO feedback
                (document_id, rating, comment, created_at)
            VALUES (%s, %s, %s, %s)
        """, (
            document_id,
            rating,
            comment,
            datetime.now(timezone.utc).isoformat()
        ))

        connection.commit()

    except Error as e:
        if connection:
            connection.rollback()
        raise RuntimeError(f"Could not save feedback: {e}") from e

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


def recent_documents(limit=10):
    """Return the most recently uploaded document records."""
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT id, filename, text_length, uploaded_at
            FROM documents
            ORDER BY id DESC
            LIMIT %s
        """, (limit,))

        return cursor.fetchall()

    except Error as e:
        raise RuntimeError(
            f"Could not retrieve documents: {e}"
        ) from e

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

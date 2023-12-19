# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
SQL Storage for Model Metadata
"""

import sqlite3
from typing import Set

from invokeai.app.services.model_records import UnknownModelException
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase

from .fetch import ModelMetadataFetchBase
from .metadata_base import AnyModelRepoMetadata


class ModelMetadataStore:
    """Store, search and fetch model metadata retrieved from remote repositories."""

    _db: SqliteDatabase
    _cursor: sqlite3.Cursor

    def __init__(self, db: SqliteDatabase):
        """
        Initialize a new object from preexisting sqlite3 connection and threading lock objects.

        :param conn: sqlite3 connection object
        :param lock: threading Lock object
        """
        super().__init__()
        self._db = db
        self._cursor = self._db.conn.cursor()
        self._enable_foreign_key_constraints()

    def add_metadata(self, model_key: str, metadata: AnyModelRepoMetadata) -> None:
        """
        Add a block of repo metadata to a model record.

        The model record config must already exist in the database with the
        same key. Otherwise a FOREIGN KEY constraint exception will be raised.

        :param model_key: Existing model key in the `model_config` table
        :param metadata: ModelRepoMetadata object to store
        """
        json_serialized = metadata.model_dump_json()
        with self._db.lock:
            try:
                self._cursor.execute(
                    """--sql
                    INSERT INTO model_metadata(
                       id,
                       metadata
                    )
                    VALUES (?,?);
                    """,
                    (
                        model_key,
                        json_serialized,
                    ),
                )
                self._update_tags(model_key, metadata.tags)
                self._db.conn.commit()
            except sqlite3.Error as e:
                self._db.conn.rollback()
                raise e

    def get_metadata(self, model_key: str) -> AnyModelRepoMetadata:
        """Retrieve the ModelRepoMetadata corresponding to model key."""
        with self._db.lock:
            self._cursor.execute(
                """--sql
                SELECT metadata FROM model_metadata
                WHERE id=?;
                """,
                (model_key,),
            )
            rows = self._cursor.fetchone()
            if not rows:
                raise UnknownModelException("model metadata not found")
            return ModelMetadataFetchBase.from_json(rows[0])

    def update_metadata(self, model_key: str, metadata: AnyModelRepoMetadata) -> AnyModelRepoMetadata:
        """
        Update metadata corresponding to the model with the indicated key.

        :param model_key: Existing model key in the `model_config` table
        :param metadata: ModelRepoMetadata object to update
        """
        json_serialized = metadata.model_dump_json()  # turn it into a json string.
        with self._db.lock:
            try:
                self._cursor.execute(
                    """--sql
                    UPDATE model_metadata
                    SET
                        metadata=?
                    WHERE id=?;
                    """,
                    (json_serialized, model_key),
                )
                if self._cursor.rowcount == 0:
                    raise UnknownModelException("model not found")
                self._update_tags(model_key, metadata.tags)
                self._db.conn.commit()
            except sqlite3.Error as e:
                self._db.conn.rollback()
                raise e

        return self.get_metadata(model_key)

    def search_by_tag(self, tags: Set[str]) -> Set[str]:
        """Return the keys of models containing all of the listed tags."""
        with self._db.lock:
            try:
                matches: Set[str] = set()
                for tag in tags:
                    self._cursor.execute(
                        """--sql
                        SELECT a.id FROM model_tags AS a,
                                           tags AS b
                        WHERE a.tag_id=b.tag_id
                          AND b.tag_text=?;
                        """,
                        (tag,),
                    )
                    model_keys = {x[0] for x in self._cursor.fetchall()}
                    matches = matches.intersection(model_keys) if len(matches) > 0 else model_keys
            except sqlite3.Error as e:
                raise e
        return matches

    def search_by_author(self, author: str) -> Set[str]:
        """Return the keys of models authored by the indicated author."""
        self._cursor.execute(
            """--sql
            SELECT id FROM model_metadata
            WHERE author=?;
            """,
            (author,),
        )
        return {x[0] for x in self._cursor.fetchall()}

    def _update_tags(self, model_key: str, tags: Set[str]) -> None:
        """Update tags for the model referenced by model_key."""
        # remove previous tags from this model
        self._cursor.execute(
            """--sql
            DELETE FROM model_tags
            WHERE id=?;
            """,
            (model_key,),
        )

        for tag in tags:
            self._cursor.execute(
                """--sql
                INSERT OR IGNORE INTO tags (
                  tag_text
                  )
                VALUES (?);
                """,
                (tag,),
            )
            self._cursor.execute(
                """--sql
                SELECT tag_id
                FROM tags
                WHERE tag_text = ?
                LIMIT 1;
                """,
                (tag,),
            )
            tag_id = self._cursor.fetchone()[0]
            self._cursor.execute(
                """--sql
                INSERT OR IGNORE INTO model_tags (
                   id,
                   tag_id
                  )
                VALUES (?,?);
                """,
                (model_key, tag_id),
            )

    def _enable_foreign_key_constraints(self) -> None:
        self._cursor.execute("PRAGMA foreign_keys = ON;")

import logging
import pyodbc
from datetime import datetime
from typing import Optional, Dict, Any


class DatabaseLogHandler(logging.Handler):
    def __init__(self, connection_string: str, table_name: str = "Logs"):
        super().__init__()
        self.connection_string = connection_string
        self.table_name = table_name
        self._ensure_table_exists()

    def _ensure_table_exists(self) -> None:
        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = '{self.table_name}')
                BEGIN
                    CREATE TABLE {self.table_name} (
                        timestamp DATETIME NOT NULL,
                        level VARCHAR(10) NOT NULL,
                        message VARCHAR(1000) NOT NULL
                    )
                END
                """)
                conn.commit()
        except Exception as e:
            print(f"Failed to create log table: {e}")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                timestamp = datetime.fromtimestamp(record.created)
                level = record.levelname
                # Store only the actual message without additional formatting
                message = record.getMessage()

                sql = f"INSERT INTO {self.table_name} (timestamp, level, message) VALUES (?, ?, ?)"
                cursor.execute(sql, (timestamp, level, message))
                conn.commit()
        except Exception as e:
            print(f"Failed to write log to database: {e}")
            print(f"Log: [{record.levelname}] {record.getMessage()}")

    def clear_old_logs(self) -> None:
        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                # Keep last 1000 logs to avoid table growing too large
                cursor.execute(f"""
                    WITH OldLogs AS (
                        SELECT timestamp,
                               ROW_NUMBER() OVER (ORDER BY timestamp DESC) AS RowNum
                        FROM {self.table_name}
                    )
                    DELETE FROM {self.table_name}
                    WHERE timestamp IN (
                        SELECT timestamp FROM OldLogs WHERE RowNum > 1000
                    )
                """)
                conn.commit()
        except Exception as e:
            print(f"Failed to clear old logs: {e}")


class Logger:
    def __init__(
            self,
            name: str = 'app',
            level: int = logging.INFO,
            use_console: bool = True,
            console_level: int = None,
            db_config: Optional[Dict[str, Any]] = None
    ) -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.handlers = []  # Clear any existing handlers

        # Configure console logging if requested
        if use_console:
            console_handler = logging.StreamHandler()
            # If console_level is provided, use it; otherwise use the global level
            handler_level = console_level if console_level is not None else level
            console_handler.setLevel(handler_level)
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

        # Configure database logging if requested
        self.db_handler = None
        if db_config:
            try:
                connection_string = self._build_connection_string(db_config)
                self.db_handler = DatabaseLogHandler(connection_string)
                # Use a simple formatter that doesn't duplicate information
                db_formatter = logging.Formatter('%(message)s')
                self.db_handler.setFormatter(db_formatter)
                # Database handler always uses the global log level
                self.db_handler.setLevel(level)
                self._logger.addHandler(self.db_handler)

                # Clear old logs
                self.db_handler.clear_old_logs()
            except Exception as e:
                print(f"Failed to initialize database logging: {e}")

    def _build_connection_string(self, config: Dict[str, Any]) -> str:
        return f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={config['Host']},{config['Port']};" \
               f"DATABASE={config['Database']};UID={config['User']};PWD={config['Password']}"

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        self._logger.critical(msg, *args, **kwargs)

    def get(self) -> logging.Logger:
        return self._logger

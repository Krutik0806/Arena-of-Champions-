import os
import logging
import json
import math
import random
import time
import asyncio
import aiohttp
import io
import requests
import threading
import re
import html
from datetime import date, datetime
from typing import Optional, Dict, List, Tuple, Union, Any, Callable
from contextlib import contextmanager
from enum import Enum, auto
from aiohttp import web

# ====================================
# ====================================

class GameType(Enum):
    """Game type enumeration"""
    CHASE = "chase"
    GUESS = "guess"
    NIGHTMARE = "nightmare"
    DAILY = "daily"

class AchievementType(Enum):
    """Achievement type enumeration"""
    WINNER = "winner"
    ORANGE_CAP = "orange cap"
    PURPLE_CAP = "purple cap"
    MVP = "mvp"
    DREAM11 = "dream11"
    FIRST_GAME = "first game"
    CENTURY_MASTER = "century master"
    NIGHTMARE_SURVIVOR = "nightmare survivor"
    CHASE_CHAMPION = "chase champion"
    GUESS_MASTER = "guess master"
    DAILY_WARRIOR = "daily warrior"
    WEEKLY_CHAMPION = "weekly champion"
    MONTHLY_LEGEND = "monthly legend"
    SEASON_KING = "season king"
    PERFECT_SCORER = "perfect scorer"
    LUCKY_GUESSER = "lucky guesser"
    STREAK_MASTER = "streak master"
    COMEBACK_KING = "comeback king"
    SPEED_DEMON = "speed demon"
    PATIENCE_MASTER = "patience master"

class GameDifficulty(Enum):
    """Game difficulty levels"""
    BEGINNER = "beginner"
    EASY = "easy" 
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"

class GameOutcome(Enum):
    """Game outcome status"""
    WON = "won"
    LOST = "lost"
    TIMEOUT = "timeout"
    ABANDONED = "abandoned"
    COMPLETED = "completed"
    ONGOING = "ongoing"

class AdminLevel(Enum):
    """Admin authorization levels"""
    SUPER_ADMIN = "SUPER_ADMIN"
    ENV_ADMIN = "ENV_ADMIN"
    DB_ADMIN = "DB_ADMIN"
    NOT_ADMIN = "NOT_ADMIN"

class LogLevel(Enum):
    """Structured logging levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AuctionState(Enum):
    """Auction state enumeration"""
    REGISTRATION = "registration"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class UserRole(Enum):
    """User roles in auction system"""
    HOST = "host"
    CAPTAIN = "captain"
    PLAYER = "player"

class Constants:
    """Application constants"""
    
    ROAST_CACHE_DURATION = 300  # 5 minutes
    LEADERBOARD_CACHE_DURATION = 60  # 1 minute
    PROFILE_CACHE_DURATION = 120  # 2 minutes
    GOAT_CACHE_DURATION = 3600  # 1 hour
    
    DB_POOL_MIN_CONN = 5
    DB_POOL_MAX_CONN = 25
    
    MAX_GUESS_ATTEMPTS = 10
    DEFAULT_GAME_TIME_LIMIT = 60
    MAX_NIGHTMARE_LEVEL = 100
    
    TELEGRAM_MAX_MESSAGE_LENGTH = 4000
    MAX_USERNAME_LENGTH = 32
    MAX_DISPLAY_NAME_LENGTH = 64
    MAX_ACHIEVEMENT_NAME_LENGTH = 100
    
    USERNAME_PATTERN = r'^[a-zA-Z0-9_]{1,32}$'
    ACHIEVEMENT_PATTERN = r'^[a-zA-Z0-9\s\-_\.]{1,100}$'
    
    SHARDS_PER_NEW_ACHIEVEMENT = 5
    SHARDS_PER_GAME_WIN = 2
    SHARDS_PER_DAILY_LOGIN = 1

# ====================================
# ====================================

class StructuredLogger:
    """Structured logging with context and proper formatting"""
    
    def __init__(self, name: str = "BotLogger"):
        self.logger = logging.getLogger(name)
        self.setup_logger()
    
    def setup_logger(self) -> None:
        """Setup logger with structured formatter"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _log(self, level: LogLevel, message: str, **context) -> None:
        """Internal logging with structured context"""
        context_str = " | ".join([f"{k}={v}" for k, v in context.items() if v is not None])
        full_message = f"{message} | {context_str}" if context_str else message
        
        level_map = {
            LogLevel.DEBUG: self.logger.debug,
            LogLevel.INFO: self.logger.info,
            LogLevel.WARNING: self.logger.warning,
            LogLevel.ERROR: self.logger.error,
            LogLevel.CRITICAL: self.logger.critical
        }
        
        level_map[level](full_message)
    
    def debug(self, message: str, **context) -> None:
        """Log debug message with context"""
        self._log(LogLevel.DEBUG, message, **context)
    
    def info(self, message: str, **context) -> None:
        """Log info message with context"""
        self._log(LogLevel.INFO, message, **context)
    
    def warning(self, message: str, **context) -> None:
        """Log warning message with context"""
        self._log(LogLevel.WARNING, message, **context)
    
    def error(self, message: str, **context) -> None:
        """Log error message with context"""
        self._log(LogLevel.ERROR, message, **context)
    
    def critical(self, message: str, **context) -> None:
        """Log critical message with context"""
        self._log(LogLevel.CRITICAL, message, **context)
    
    def game_event(self, event: str, user_id: Optional[int] = None, 
                   game_type: Optional[GameType] = None, 
                   game_id: Optional[str] = None, **extra):
        """Log game-related events"""
        context = {
            'event_type': 'game',
            'user_id': user_id,
            'game_type': game_type.value if game_type else None,
            'game_id': game_id,
            **extra
        }
        self.info(event, **context)
    
    def user_action(self, action: str, user_id: Optional[int] = None, 
                    username: Optional[str] = None, **extra):
        """Log user actions"""
        context = {
            'event_type': 'user_action',
            'user_id': user_id,
            'username': username,
            **extra
        }
        self.info(action, **context)
    
    def database_operation(self, operation: str, table: Optional[str] = None, 
                          user_id: Optional[int] = None, **extra):
        """Log database operations"""
        context = {
            'event_type': 'database',
            'operation': operation,
            'table': table,
            'user_id': user_id,
            **extra
        }
        self.info(f"DB {operation}", **context)
    
    def achievement_event(self, event: str, user_id: Optional[int] = None,
                         achievement_type: Optional[AchievementType] = None,
                         achievement_name: Optional[str] = None, **extra):
        """Log achievement-related events"""
        context = {
            'event_type': 'achievement',
            'user_id': user_id,
            'achievement_type': achievement_type.value if achievement_type else None,
            'achievement_name': achievement_name,
            **extra
        }
        self.info(event, **context)

logger = StructuredLogger("ChatBot")

# ====================================
# ====================================

class TelegramAPIWrapper:
    """Wrapper for Telegram bot API with error handling and retry logic"""
    
    def __init__(self, bot, max_retries: int = 3, retry_delay: float = 1.0):
        self.bot = bot
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    async def _execute_with_retry(self, operation_name: str, func, *args, **kwargs):
        """Execute Telegram API call with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                result = await func(*args, **kwargs)
                if attempt > 0:
                    logger.info("Telegram API retry successful", 
                              operation=operation_name, 
                              attempt=attempt + 1)
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning("Telegram API attempt failed", 
                             operation=operation_name,
                             attempt=attempt + 1,
                             error=str(e))
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        
        logger.error("Telegram API operation failed after retries", 
                   operation=operation_name,
                   max_retries=self.max_retries,
                   error=str(last_exception))
        raise last_exception
    
    async def send_message(self, chat_id: int, text: str, **kwargs) -> bool:
        """Send message with error handling and retry logic"""
        try:
            if len(text) > Constants.TELEGRAM_MAX_MESSAGE_LENGTH:
                text = text[:Constants.TELEGRAM_MAX_MESSAGE_LENGTH - 3] + "..."
            
            await self._execute_with_retry(
                "send_message",
                self.bot.send_message,
                chat_id=chat_id,
                text=text,
                **kwargs
            )
            
            logger.user_action("Message sent successfully", 
                             user_id=chat_id, 
                             message_length=len(text))
            return True
            
        except Exception as e:
            logger.error("Failed to send message", 
                       user_id=chat_id, 
                       error=str(e))
            return False
    
    async def edit_message(self, chat_id: int, message_id: int, text: str, **kwargs) -> bool:
        """Edit message with error handling and retry logic"""
        try:
            if len(text) > Constants.TELEGRAM_MAX_MESSAGE_LENGTH:
                text = text[:Constants.TELEGRAM_MAX_MESSAGE_LENGTH - 3] + "..."
            
            await self._execute_with_retry(
                "edit_message",
                self.bot.edit_message_text,
                text=text,
                chat_id=chat_id,
                message_id=message_id,
                **kwargs
            )
            
            logger.user_action("Message edited successfully", 
                             user_id=chat_id, 
                             message_id=message_id)
            return True
            
        except Exception as e:
            logger.error("Failed to edit message", 
                       user_id=chat_id, 
                       message_id=message_id,
                       error=str(e))
            return False
    
    async def send_photo(self, chat_id: int, photo, caption: str = None, **kwargs) -> bool:
        """Send photo with error handling and retry logic"""
        try:
            await self._execute_with_retry(
                "send_photo",
                self.bot.send_photo,
                chat_id=chat_id,
                photo=photo,
                caption=caption,
                **kwargs
            )
            
            logger.user_action("Photo sent successfully", 
                             user_id=chat_id, 
                             has_caption=bool(caption))
            return True
            
        except Exception as e:
            logger.error("Failed to send photo", 
                       user_id=chat_id, 
                       error=str(e))
            return False
    
    async def answer_callback_query(self, callback_query_id: str, text: str = None, **kwargs) -> bool:
        """Answer callback query with error handling"""
        try:
            await self._execute_with_retry(
                "answer_callback_query",
                self.bot.answer_callback_query,
                callback_query_id=callback_query_id,
                text=text,
                **kwargs
            )
            
            logger.user_action("Callback query answered", 
                             callback_id=callback_query_id)
            return True
            
        except Exception as e:
            logger.error("Failed to answer callback query", 
                       callback_id=callback_query_id,
                       error=str(e))
            return False
    
    async def delete_message(self, chat_id: int, message_id: int) -> bool:
        """Delete message with error handling"""
        try:
            await self._execute_with_retry(
                "delete_message",
                self.bot.delete_message,
                chat_id=chat_id,
                message_id=message_id
            )
            
            logger.user_action("Message deleted successfully", 
                             user_id=chat_id, 
                             message_id=message_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete message", 
                       user_id=chat_id, 
                       message_id=message_id,
                       error=str(e))
            return False

# ====================================
# ====================================

class DeferredTaskScheduler:
    """
    Manages async tasks that may be scheduled before the event loop is available.
    Tasks are queued and executed when the event loop becomes available.
    """
    
    def __init__(self):
        self._pending_tasks = []
        self._loop = None
    
    def set_event_loop(self, loop):
        """Set the event loop and execute any pending tasks"""
        self._loop = loop
        if self._pending_tasks:
            logger.debug(f"Executing {len(self._pending_tasks)} deferred tasks")
            for task_coro in self._pending_tasks:
                try:
                    asyncio.create_task(task_coro)
                except Exception as e:
                    logger.error("Error executing deferred task", 
                               task_type="deferred_task", 
                               error=str(e))
            self._pending_tasks.clear()
    
    def schedule_task(self, coro):
        """
        Schedule a coroutine to run. If event loop is available, run immediately.
        Otherwise, queue for later execution.
        """
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(coro)
        except RuntimeError:
            logger.debug("No event loop available, deferring task")
            self._pending_tasks.append(coro)

deferred_scheduler = DeferredTaskScheduler()

# ====================================
# ====================================

class DatabaseMigration:
    """
    Simple database migration system with version tracking.
    Prevents CREATE TABLE failures in production environments.
    """
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.migrations = []
    
    def add_migration(self, version: int, name: str, sql: str, rollback_sql: str = None):
        """Add a migration to be executed"""
        self.migrations.append({
            'version': version,
            'name': name,
            'sql': sql,
            'rollback_sql': rollback_sql
        })
    
    def init_migration_table(self, conn):
        """Create migration tracking table"""
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                id SERIAL PRIMARY KEY,
                version INTEGER UNIQUE NOT NULL,
                name VARCHAR(255) NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checksum VARCHAR(64)
            )
        """)
        conn.commit()
        cursor.close()
    
    def get_applied_migrations(self, conn) -> set:
        """Get set of applied migration versions"""
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT version FROM _migrations ORDER BY version")
            return {row[0] for row in cursor.fetchall()}
        except Exception:
            return set()
        finally:
            cursor.close()
    
    def apply_migration(self, conn, migration):
        """Apply a single migration"""
        cursor = conn.cursor()
        try:
            logger.database_operation(
                "migration_apply",
                migration_version=migration['version'],
                migration_name=migration['name']
            )
            cursor.execute(migration['sql'])
            
            cursor.execute("""
                INSERT INTO _migrations (version, name) 
                VALUES (%s, %s)
            """, (migration['version'], migration['name']))
            
            conn.commit()
            logger.database_operation(
                "migration_success",
                migration_version=migration['version']
            )
            return True
        except Exception as e:
            logger.database_operation(
                "migration_failed",
                migration_version=migration['version'],
                error=str(e)
            )
            conn.rollback()
            return False
        finally:
            cursor.close()
    
    def run_migrations(self):
        """Run pending migrations"""
        if not self.db_pool:
            logger.error("No database pool available for migrations")
            return False
        
        try:
            conn = self.db_pool.getconn()
            self.init_migration_table(conn)
            applied = self.get_applied_migrations(conn)
            
            pending = [m for m in sorted(self.migrations, key=lambda x: x['version']) 
                      if m['version'] not in applied]
            
            if not pending:
                logger.info("No pending migrations")
                return True
            
            logger.info(f"Running {len(pending)} pending migrations")
            for migration in pending:
                if not self.apply_migration(conn, migration):
                    return False
            
            logger.info("All migrations completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Migration system error: {e}")
            return False
        finally:
            if conn:
                self.db_pool.putconn(conn)

# ====================================
# ====================================

class SecureSQL:
    """
    SQL security utilities to prevent injection attacks.
    Provides table name validation and safe dynamic SQL construction.
    """
    
    ALLOWED_TABLES = {
        'players', 'admins', 'chats', 'achievements', 'cooldowns',
        'chase_games', 'guess_games', 'roast_rotation', 'roast_usage',
        'banned_users', 'daily_goat', 'daily_leaderboard_entries',
        'nightmare_games', 'shard_transactions', 'achievement_confirmations',
        'title_backups', 'system_performance', '_migrations'
    }
    
    @classmethod
    def validate_table_name(cls, table_name: str) -> bool:
        """Validate table name against whitelist"""
        if not table_name or not isinstance(table_name, str):
            return False
        return table_name.lower().strip() in cls.ALLOWED_TABLES
    
    @classmethod
    def safe_table_identifier(cls, table_name: str) -> str:
        """
        Return safe table identifier or raise exception.
        Use this before any dynamic table name usage.
        """
        if not cls.validate_table_name(table_name):
            raise ValueError(f"Invalid or unauthorized table name: {table_name}")
        return table_name.lower().strip()
    
    @classmethod
    def safe_delete_query(cls, table_name: str, where_clause: str = None, params: tuple = None) -> tuple:
        """
        Generate safe DELETE query with validated table name.
        Returns (query, params) tuple ready for cursor.execute()
        """
        safe_table = cls.safe_table_identifier(table_name)
        
        if where_clause:
            query = f"DELETE FROM {safe_table} WHERE {where_clause}"
        else:
            query = f"DELETE FROM {safe_table}"
        
        return query, params or ()
    
    @classmethod
    def safe_truncate_query(cls, table_name: str) -> str:
        """Generate safe TRUNCATE query with validated table name"""
        safe_table = cls.safe_table_identifier(table_name)
        return f"TRUNCATE TABLE {safe_table}"

# ====================================
# ====================================

def paginate_text(text: str, max_length: int = 4000) -> List[str]:
    """
    Paginate long text into chunks that fit in Telegram messages.
    Tries to break at natural points (line breaks) when possible.
    """
    if len(text) <= max_length:
        return [text]
    
    pages = []
    current_page = ""
    
    lines = text.split('\n')
    
    for line in lines:
        if len(current_page + line + '\n') > max_length:
            if current_page:
                pages.append(current_page.rstrip())
                current_page = line + '\n'
            else:
                while len(line) > max_length:
                    pages.append(line[:max_length])
                    line = line[max_length:]
                current_page = line + '\n'
        else:
            current_page += line + '\n'
    
    if current_page:
        pages.append(current_page.rstrip())
    
    return pages

from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup, error as telegram_error
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
from telegram.error import BadRequest
import psycopg2
from psycopg2 import sql, pool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

async def safe_edit_message(query, text: str, **kwargs) -> bool:
    """Safely edit a message with proper error handling"""
    try:
        await query.edit_message_text(text, **kwargs)
        return True
    except BadRequest as e:
        error_msg = str(e).lower()
        if "message is not modified" in error_msg or "message_not_modified" in error_msg:
            logger.debug("Message not modified - content unchanged")
            return True
        elif "message to edit not found" in error_msg or "message can't be edited" in error_msg or "message to delete not found" in error_msg:
            logger.warning(f"Cannot edit message: {e}")
            try:
                await query.message.reply_text(text, **kwargs)
                return True
            except:
                return False
        else:
            logger.error(f"BadRequest while editing message: {e}")
            return False
    except Exception as e:
        logger.error(f"Unexpected error editing message: {e}")
        return False

async def send_paginated_message(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, parse_mode: str = 'HTML', max_length: int = 4000):
    """Send a long message as multiple paginated messages"""
    pages = paginate_text(text, max_length)
    
    for i, page in enumerate(pages):
        if len(pages) > 1:
            page_indicator = f"\n\n📄 <i>Page {i+1} of {len(pages)}</i>"
            if len(page + page_indicator) <= max_length:
                page += page_indicator
        
        await safe_send(update.message.reply_text, page, parse_mode=parse_mode)
        
        if i < len(pages) - 1:
            await asyncio.sleep(0.5)

class InputValidator:
    """
    🛡️ Unified Input Validation Utility
    
    Consolidates all input validation logic to eliminate duplication and ensure
    consistent validation rules across the entire bot application.
    """
    
    USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_]{1,32}$')
    ACHIEVEMENT_PATTERN = re.compile(r'^[a-zA-Z0-9\s\-_\.]{1,100}$')
    CONTROL_CHARS_PATTERN = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
    
    MAX_USERNAME_LENGTH = 32
    MAX_DISPLAY_NAME_LENGTH = 64
    MAX_ACHIEVEMENT_NAME_LENGTH = 100
    DEFAULT_TEXT_LIMIT = 1000
    
    MIN_TELEGRAM_ID = 1
    MAX_TELEGRAM_ID = 10**12
    MIN_CHAT_ID = -10**12
    MAX_CHAT_ID = 10**12
    
    @classmethod
    def username(cls, username: str) -> bool:
        """
        Validate username format according to Telegram standards.
        
        Args:
            username (str): The username to validate
            
        Returns:
            bool: True if username is valid, False otherwise
            
        Rules:
            - Alphanumeric characters and underscores only
            - Length: 1-32 characters
        """
        if not username:
            return False
        return bool(cls.USERNAME_PATTERN.match(username))
    
    @classmethod
    def display_name(cls, name: str) -> bool:
        """
        Validate display name for reasonable length and content.
        
        Args:
            name (str): The display name to validate
            
        Returns:
            bool: True if display name is valid, False otherwise
            
        Rules:
            - Non-empty after stripping whitespace
            - Maximum length: 64 characters
        """
        if not name or len(name.strip()) < 1:
            return False
        return len(name.strip()) <= cls.MAX_DISPLAY_NAME_LENGTH
    
    @classmethod
    def achievement_name(cls, name: str) -> bool:
        """
        Validate achievement name format.
        
        Args:
            name (str): The achievement name to validate
            
        Returns:
            bool: True if achievement name is valid, False otherwise
            
        Rules:
            - Alphanumeric, spaces, hyphens, underscores, dots
            - Length: 1-100 characters
        """
        if not name:
            return False
        return bool(cls.ACHIEVEMENT_PATTERN.match(name))
    
    @classmethod
    def telegram_id(cls, telegram_id: Union[int, str]) -> bool:
        """
        Validate Telegram user ID format.
        
        Args:
            telegram_id (Union[int, str]): The Telegram ID to validate
            
        Returns:
            bool: True if ID is valid, False otherwise
            
        Rules:
            - Must be convertible to integer
            - Range: 1 to 10^12 (reasonable Telegram ID range)
        """
        try:
            tid = int(telegram_id)
            return cls.MIN_TELEGRAM_ID <= tid <= cls.MAX_TELEGRAM_ID
        except (ValueError, TypeError):
            return False
    
    @classmethod
    def chat_id(cls, chat_id: Union[int, str]) -> bool:
        """
        Validate chat/group ID format.
        
        Args:
            chat_id (Union[int, str]): The chat ID to validate
            
        Returns:
            bool: True if chat ID is valid, False otherwise
            
        Rules:
            - Must be convertible to integer  
            - Range: -10^12 to 10^12 (includes negative group IDs)
        """
        try:
            cid = int(chat_id)
            return cls.MIN_CHAT_ID <= cid <= cls.MAX_CHAT_ID
        except (ValueError, TypeError):
            return False
    
    @classmethod
    def sanitize_text(cls, text: str, max_length: int = None, html_escape: bool = True) -> str:
        """
        Sanitize user input to prevent security issues and ensure data integrity.
        
        Args:
            text (str): The input text to sanitize
            max_length (int, optional): Maximum allowed length. Defaults to 1000.
            html_escape (bool): Whether to HTML-escape the text. Defaults to True.
            
        Returns:
            str: Sanitized text
            
        Operations:
            1. Handle None/empty input
            2. Remove control characters and null bytes
            3. Limit length
            4. HTML-escape for safety (optional)
            5. Strip whitespace
        """
        if not text:
            return ""
        
        if max_length is None:
            max_length = cls.DEFAULT_TEXT_LIMIT
        
        text = cls.CONTROL_CHARS_PATTERN.sub('', text)
        
        text = text[:max_length]
        
        if html_escape:
            text = html.escape(text)
        
        return text.strip()
    
    @classmethod
    def is_safe_text(cls, text: str, max_length: int = None, allow_html: bool = False) -> bool:
        """
        Check if text is safe without modifying it.
        
        Args:
            text (str): Text to validate
            max_length (int, optional): Maximum allowed length
            allow_html (bool): Whether to allow HTML characters
            
        Returns:
            bool: True if text is safe, False otherwise
        """
        if not text:
            return True
        
        if max_length is None:
            max_length = cls.DEFAULT_TEXT_LIMIT
        
        if len(text) > max_length:
            return False
        
        if cls.CONTROL_CHARS_PATTERN.search(text):
            return False
        
        if not allow_html and any(char in text for char in '<>&"\''):
            return False
        
        return True
    
    @classmethod
    def truncate_with_ellipsis(cls, text: str, max_length: int, ellipsis: str = "...") -> str:
        """
        Truncate text with ellipsis if it exceeds max_length.
        
        Args:
            text (str): Text to potentially truncate
            max_length (int): Maximum length before truncation
            ellipsis (str): String to append when truncated
            
        Returns:
            str: Original text or truncated text with ellipsis
        """
        if not text:
            return ""
        
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(ellipsis)] + ellipsis
    
    @classmethod
    def validate_command_args(cls, args: list, min_args: int, max_args: int = None) -> tuple[bool, str]:
        """
        Validate command arguments count.
        
        Args:
            args (list): Command arguments list
            min_args (int): Minimum required arguments
            max_args (int, optional): Maximum allowed arguments
            
        Returns:
            tuple[bool, str]: (is_valid, error_message)
        """
        if len(args) < min_args:
            return False, f"❌ **Not enough arguments.** Expected at least {min_args}, got {len(args)}."
        
        if max_args is not None and len(args) > max_args:
            return False, f"❌ **Too many arguments.** Expected at most {max_args}, got {len(args)}."
        
        return True, ""
    
    @classmethod
    def safe_join_args(cls, args: list, start_index: int = 0, default: str = "") -> str:
        """
        Safely join command arguments from a starting index.
        
        Args:
            args (list): Arguments list
            start_index (int): Index to start joining from
            default (str): Default value if no args to join
            
        Returns:
            str: Joined arguments or default
        """
        if not args or start_index >= len(args):
            return default
        
        return " ".join(args[start_index:])
    
    @classmethod 
    def normalize_whitespace(cls, text: str) -> str:
        """
        Normalize whitespace in text (collapse multiple spaces, strip).
        
        Args:
            text (str): Text to normalize
            
        Returns:
            str: Text with normalized whitespace
        """
        if not text:
            return ""
        
        return re.sub(r'\s+', ' ', text).strip()
    
    @classmethod
    def is_empty_or_whitespace(cls, text: str) -> bool:
        """
        Check if text is None, empty, or only whitespace.
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if text is effectively empty
        """
        return not text or not text.strip()

def validate_chat_id(chat_id: Union[int, str]) -> bool:
    """Backward compatibility wrapper for InputValidator.chat_id()"""
    return InputValidator.chat_id(chat_id)

@contextmanager
def safe_db_transaction(connection):
    """Context manager for safe database transactions"""
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise

class SmartCacheManager:
    """
    🧠 Smart Cache Management System
    
    Solves the "cache may become stale forever" problem by implementing:
    - Database change detection via timestamps/hashing
    - Manual cache invalidation for admin operations
    - Selective invalidation by user/data type
    - Global cache versioning
    """
    
    @staticmethod
    def generate_data_hash(data: dict) -> str:
        """Generate hash of profile data to detect changes"""
        import hashlib
        import json
        
        normalized_data = {
            'player_score': data.get('player', {}).get('score', 0),
            'player_shards': data.get('player', {}).get('shards', 0),
            'player_level': data.get('chase_stats', {}).get('highest_level', 0),
            'achievements_count': len(data.get('achievements', [])),
            'last_game_time': data.get('chase_stats', {}).get('last_game_time', 0)
        }
        
        data_string = json.dumps(normalized_data, sort_keys=True)
        return hashlib.md5(data_string.encode()).hexdigest()
    
    @staticmethod
    def should_invalidate_cache(cache_entry: dict, current_time: float, cache_duration: int, 
                               global_invalidated_at: float = 0) -> bool:
        """
        Determine if cache entry should be invalidated
        
        Args:
            cache_entry: Cache entry with 'timestamp' and 'data_hash'
            current_time: Current timestamp
            cache_duration: Cache duration in seconds
            global_invalidated_at: Global cache invalidation timestamp
            
        Returns:
            bool: True if cache should be invalidated
        """
        if current_time - cache_entry.get('timestamp', 0) >= cache_duration:
            return True
        
        if cache_entry.get('timestamp', 0) < global_invalidated_at:
            return True
        
        return False
    
    @staticmethod
    def invalidate_related_caches(bot_instance, operation_type: str, user_id: int = None):
        """
        Invalidate caches based on the type of database operation
        
        Args:
            bot_instance: Bot instance with cache objects
            operation_type: Type of operation ('profile_update', 'achievement', 'admin_action', etc.)
            user_id: Specific user ID if operation affects single user
        """
        current_time = time.time()
        
        if operation_type == "profile_update" and user_id:
            if hasattr(bot_instance, 'profile_cache'):
                if user_id in bot_instance.profile_cache['data']:
                    del bot_instance.profile_cache['data'][user_id]
                if user_id in bot_instance.profile_cache['last_updated']:
                    del bot_instance.profile_cache['last_updated'][user_id]
                if user_id in bot_instance.profile_cache['data_version']:
                    del bot_instance.profile_cache['data_version'][user_id]
                    
        elif operation_type == "leaderboard_change":
            if hasattr(bot_instance, 'leaderboard_cache'):
                bot_instance.leaderboard_cache['last_updated'] = 0
                bot_instance.leaderboard_cache['data'] = []
                
        elif operation_type == "achievement_update":
            if hasattr(bot_instance, 'profile_cache'):
                bot_instance.profile_cache['data'].clear()
                bot_instance.profile_cache['last_updated'].clear()
                bot_instance.profile_cache['data_version'].clear()
                
        elif operation_type == "admin_bulk_action":
            if hasattr(bot_instance, 'profile_cache'):
                bot_instance.profile_cache['global_invalidated_at'] = current_time
                bot_instance.profile_cache['data'].clear()
                bot_instance.profile_cache['last_updated'].clear()
                bot_instance.profile_cache['data_version'].clear()
            
            if hasattr(bot_instance, 'leaderboard_cache'):
                bot_instance.leaderboard_cache['last_updated'] = 0
                bot_instance.leaderboard_cache['data'] = []
        
        logger.debug(f"Cache invalidated: {operation_type}, user_id: {user_id}")

class ThreadSafeDict:
    """
    🔒 Thread-Safe Dictionary Wrapper
    
    Solves race conditions in shared mutable dictionaries by providing
    atomic operations with proper locking mechanisms.
    """
    
    def __init__(self, initial_data: dict = None):
        self._data = initial_data or {}
        self._lock = threading.RLock()  # Reentrant lock for nested operations
    
    def get(self, key, default=None):
        """Thread-safe get operation"""
        with self._lock:
            return self._data.get(key, default)
    
    def __getitem__(self, key):
        """Thread-safe item access"""
        with self._lock:
            return self._data[key]
    
    def __setitem__(self, key, value):
        """Thread-safe item assignment"""
        with self._lock:
            self._data[key] = value
    
    def __delitem__(self, key):
        """Thread-safe item deletion"""
        with self._lock:
            del self._data[key]
    
    def __contains__(self, key):
        """Thread-safe membership test"""
        with self._lock:
            return key in self._data
    
    def __len__(self):
        """Thread-safe length"""
        with self._lock:
            return len(self._data)
    
    def pop(self, key, default=None):
        """Thread-safe pop operation"""
        with self._lock:
            return self._data.pop(key, default)
    
    def update(self, other):
        """Thread-safe update operation"""
        with self._lock:
            self._data.update(other)
    
    def clear(self):
        """Thread-safe clear operation"""
        with self._lock:
            self._data.clear()
    
    def items(self):
        """Thread-safe items iteration (returns copy)"""
        with self._lock:
            return list(self._data.items())
    
    def keys(self):
        """Thread-safe keys iteration (returns copy)"""
        with self._lock:
            return list(self._data.keys())
    
    def values(self):
        """Thread-safe values iteration (returns copy)"""
        with self._lock:
            return list(self._data.values())
    
    def atomic_increment(self, key, increment=1, default=0):
        """Thread-safe atomic increment operation"""
        with self._lock:
            current = self._data.get(key, default)
            new_value = current + increment
            self._data[key] = new_value
            return new_value
    
    def atomic_operation(self, func, *args, **kwargs):
        """Execute a function atomically on the internal data"""
        with self._lock:
            return func(self._data, *args, **kwargs)
    
    def copy(self):
        """Thread-safe copy operation"""
        with self._lock:
            return self._data.copy()
    
    def setdefault(self, key, default):
        """Thread-safe setdefault operation"""
        with self._lock:
            return self._data.setdefault(key, default)
    
    def append_to_list(self, key, value, default_list=None):
        """Thread-safe append to list operation"""
        with self._lock:
            if key not in self._data:
                self._data[key] = default_list or []
            if isinstance(self._data[key], list):
                self._data[key].append(value)
            else:
                raise TypeError(f"Value at key '{key}' is not a list")
    
    def extend_list(self, key, values, default_list=None):
        """Thread-safe extend list operation"""
        with self._lock:
            if key not in self._data:
                self._data[key] = default_list or []
            if isinstance(self._data[key], list):
                self._data[key].extend(values)
            else:
                raise TypeError(f"Value at key '{key}' is not a list")
    
    def pop_from_list(self, key, index=-1):
        """Thread-safe pop from list operation"""
        with self._lock:
            if key in self._data and isinstance(self._data[key], list):
                return self._data[key].pop(index) if self._data[key] else None
            return None
    
    def list_length(self, key):
        """Thread-safe get list length"""
        with self._lock:
            if key in self._data and isinstance(self._data[key], list):
                return len(self._data[key])
            return 0
    
    def update_nested_dict(self, key, nested_key, value, default_dict=None):
        """Thread-safe nested dictionary update"""
        with self._lock:
            if key not in self._data:
                self._data[key] = default_dict or {}
            if isinstance(self._data[key], dict):
                self._data[key][nested_key] = value
            else:
                raise TypeError(f"Value at key '{key}' is not a dictionary")
    
    def get_nested(self, key, nested_key, default=None):
        """Thread-safe nested dictionary get"""
        with self._lock:
            if key in self._data and isinstance(self._data[key], dict):
                return self._data[key].get(nested_key, default)
            return default

class ThreadSafeCounter(ThreadSafeDict):
    """
    🔢 Thread-Safe Counter for Usage Tracking
    
    Specialized thread-safe dictionary for counting operations
    with atomic increment/decrement methods.
    """
    
    def increment(self, key, amount=1):
        """Atomically increment counter for key"""
        return self.atomic_increment(key, amount, 0)
    
    def decrement(self, key, amount=1):
        """Atomically decrement counter for key"""
        return self.atomic_increment(key, -amount, 0)
    
    def get_min_value(self):
        """Get minimum value across all counters"""
        with self._lock:
            return min(self._data.values()) if self._data else 0
    
    def get_keys_with_min_value(self):
        """Get all keys that have the minimum value"""
        with self._lock:
            if not self._data:
                return []
            min_val = min(self._data.values())
            return [key for key, value in self._data.items() if value == min_val]
    
    def reset_counter(self, key):
        """Reset specific counter to 0"""
        with self._lock:
            self._data[key] = 0
    
    def reset_all(self):
        """Reset all counters to 0"""
        with self._lock:
            for key in self._data:
                self._data[key] = 0

def split_message_safely(message: str, max_length: int = 4090) -> List[str]:
    """
    Split long messages at safe boundaries to prevent HTML parsing errors.
    Ensures HTML tags and emojis aren't broken mid-way.
    """
    if len(message) <= max_length:
        return [message]
    
    messages = []
    current_msg = ""
    
    lines = message.split('\n')
    
    for line in lines:
        test_msg = current_msg + ('\n' if current_msg else '') + line
        
        if len(test_msg) > max_length:
            if current_msg:
                current_msg += "\n\n<i>📄 Continued...</i>"
                messages.append(current_msg)
                current_msg = ""
            
            if len(line) > max_length:
                words = line.split(' ')
                for word in words:
                    test_word = current_msg + (' ' if current_msg else '') + word
                    
                    if len(test_word) > max_length - 50:  # Leave buffer
                        if current_msg:
                            current_msg += "\n\n<i>📄 Continued...</i>"
                            messages.append(current_msg)
                            current_msg = word
                        else:
                            current_msg = word[:max_length-50] + "..."
                            messages.append(current_msg)
                            current_msg = "..." + word[max_length-50:]
                    else:
                        current_msg = test_word
            else:
                current_msg = line
        else:
            current_msg = test_msg
    
    if current_msg:
        messages.append(current_msg)
    
    return messages

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def H(s: str | None) -> str:
    """HTML-escape helper for user-supplied content using InputValidator"""
    return InputValidator.sanitize_text(s or "", html_escape=True) if s else ""

async def safe_send(call, *args, **kwargs):
    """Safely send Telegram messages with rate limiting and retry logic"""
    from telegram.error import RetryAfter, TimedOut, NetworkError
    
    retries = 3
    for i in range(retries):
        try:
            return await call(*args, **kwargs)
        except RetryAfter as e:
            await asyncio.sleep(e.retry_after + 1)
        except (TimedOut, NetworkError):
            await asyncio.sleep(1 + i)
        except Exception as e:
            pass
            if i == retries - 1:  # Last attempt
                raise
            await asyncio.sleep(1)
    
    return await call(*args, **kwargs)

def log_exception(context: str, e: Exception, user_id: int = None):
    """Centralized exception logging with context"""
    error_msg = f"[{context}] Error"
    if user_id:
        error_msg += f" for user {user_id}"
    error_msg += f": {type(e).__name__}: {e}"
    logger.error(error_msg, exc_info=True)

ERROR_MESSAGES = {
    'database': "❌ <b>Database temporarily unavailable!</b>\n\nPlease try again in a moment.",
    'permission': "❌ <b>ACCESS DENIED!</b>\n\n🛡️ You don't have permission for this action.",
    'invalid_input': "❌ <b>Invalid input!</b>\n\nPlease check your command and try again.", 
    'rate_limit': "⏳ <b>Please slow down!</b>\n\nToo many requests. Try again in a moment.",
    'generic': "❌ <b>Something went wrong!</b>\n\nPlease try again later."
}

def db_query(read_only=False, return_conn=False, fallback=None):
    """
    Database operation decorator that eliminates repetitive connection logic.
    
    Args:
        read_only (bool): If True, no commit is performed (for SELECT queries)
        return_conn (bool): If True, passes both cursor and connection to the function
        fallback: Value to return if database connection fails or exception occurs
    
    Usage:
        @db_query()
        def my_method(self, cursor, param1, param2):
            cursor.execute("INSERT INTO table VALUES (%s, %s)", (param1, param2))
            return cursor.fetchone()
        
        @db_query(read_only=True)
        def get_user(self, cursor, user_id):
            cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            return cursor.fetchone()
            
        @db_query(read_only=True, fallback={'total': 0})
        def get_stats(self, cursor):
            cursor.execute("SELECT COUNT(*) FROM table")
            return {'total': cursor.fetchone()[0]}
            
        @db_query(return_conn=True)
        def complex_operation(self, cursor, conn, data):
            cursor.execute("INSERT ...", data)
            return cursor.rowcount
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            conn = self.get_db_connection()
            if not conn:
                logger.error("Database connection failed", 
                           function=func.__name__, 
                           event_type="database_connection")
                return fallback
            
            cursor = None
            try:
                cursor = conn.cursor()
                
                if return_conn:
                    result = func(self, cursor, conn, *args, **kwargs)
                else:
                    result = func(self, cursor, *args, **kwargs)
                
                if not read_only:
                    conn.commit()
                    
                return result
                
            except Exception as e:
                if not read_only:
                    conn.rollback()
                logger.error("Database operation error", 
                           function=func.__name__, 
                           error=str(e), 
                           event_type="database_operation")
                return fallback
            finally:
                if cursor:
                    cursor.close()
                self.return_db_connection(conn)
        
        return wrapper
    return decorator

def check_banned(func):
    """Decorator to check if user is banned and registered before executing command"""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user = update.effective_user
        user_id = user.id
        
        global bot_instance
        
        if func.__name__ == 'start_command':
            return await func(update, context, *args, **kwargs)
        
        if hasattr(globals().get('bot_instance'), 'is_banned') and bot_instance.is_banned(user_id):
            await update.message.reply_text(
                "🚫 <b>ACCESS RESTRICTED</b>\n\n"
                "Your access to this bot has been restricted.\n"
                "Contact administrators if you believe this is an error.",
                parse_mode='HTML'
            )
            return
        
        admin_commands = ['admin_status_command', 'broadcast_command', 'add_admin_command', 
                         'remove_admin_command', 'list_admins_command', 'adminpanel_command',
                         'reset_all_command', 'backup_command', 'restart_command']
        
        if func.__name__ not in admin_commands:
            if hasattr(globals().get('bot_instance'), 'get_player_by_telegram_id'):
                player = bot_instance.get_player_by_telegram_id(user_id)
                if not player:
                    await update.message.reply_text(
                        "⚠️ <b>REGISTRATION REQUIRED!</b> ⚠️\n\n"
                        "🏆 <b>Welcome to Arena Of Champions!</b> 🏏\n\n"
                        "You must register first to access bot features.\n\n"
                        "🚀 <b>Click here to start:</b> /start\n\n"
                        "Join the arena and become a champion! 👑",
                        parse_mode='HTML'
                    )
                    return
        
        return await func(update, context, *args, **kwargs)
    
    return wrapper

def handle_database_errors(func):
    """Decorator to handle database errors gracefully"""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        try:
            return await func(update, context, *args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            
            if 'does not exist' in error_msg or 'relation' in error_msg:
                table_name = "unknown"
                if 'cooldowns' in error_msg:
                    table_name = "cooldowns"
                elif 'admins' in error_msg:
                    table_name = "admins"
                elif 'chats' in error_msg:
                    table_name = "chats"
                elif 'achievements' in error_msg:
                    table_name = "achievements"
                elif 'chase_games' in error_msg:
                    table_name = "chase_games"
                elif 'guess_games' in error_msg:
                    table_name = "guess_games"
                elif 'roast_rotation' in error_msg:
                    table_name = "roast_rotation"
                
                await update.message.reply_text(
                    f"🚨 <b>Database Setup Required</b>\n\n"
                    f"❌ Missing table: <code>{table_name}</code>\n\n"
                    f"📋 <b>To fix this:</b>\n"
                    f"1. Run the complete database schema\n"
                    f"2. Or restart the bot to auto-create tables\n"
                    f"3. Contact admin if issue persists\n\n"
                    f"🔧 <i>Bot initialization may be incomplete</i>",
                    parse_mode='HTML'
                )
                
                logger.error(f"Database table missing: {table_name} - Error: {e}")
                return
            
            raise e
    
    return wrapper

class BotWatchdog:
    def __init__(self, url: str, interval: int = 60):
        self.url = url
        self.interval = interval
        self.running = False
        self.task = None

    def start(self):
        """Start watchdog as async task - requires event loop"""
        if self.running:
            return
            
        self.running = True
        try:
            loop = asyncio.get_running_loop()
            self.task = loop.create_task(self._run())
            logger.info(f"Watchdog started as async task for {self.url}")
        except RuntimeError:
            logger.warning("No event loop available - watchdog disabled. Ensure bot.run() is called first.")
            self.running = False

    def stop(self):
        """Stop watchdog task"""
        self.running = False
        if hasattr(self, 'task') and self.task:
            self.task.cancel()
        elif hasattr(self, 'thread') and self.thread:
            self.thread.join()

    async def _run(self):
        """Enhanced async watchdog with error recovery and restart capability"""
        consecutive_failures = 0
        max_failures = 3
        
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                while self.running:
                    try:
                        async with session.get(self.url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                            if response.status != 200:
                                consecutive_failures += 1
                                logger.error(f"Health check failed with status code {response.status} (failure {consecutive_failures}/{max_failures})")
                                
                                if consecutive_failures >= max_failures:
                                    logger.critical("Health check failed multiple times. Triggering restart.")
                                    os._exit(1)
                            else:
                                if consecutive_failures > 0:
                                    logger.info("Health check recovered")
                                    consecutive_failures = 0
                                    
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        consecutive_failures += 1
                        logger.error(f"Health check request failed: {e} (failure {consecutive_failures}/{max_failures})")
                        
                        if consecutive_failures >= max_failures:
                            logger.critical("Health check failed multiple times. Triggering restart.")
                            os._exit(1)
                            
                    except Exception as e:
                        consecutive_failures += 1
                        logger.error(f"Unexpected error in watchdog: {e} (failure {consecutive_failures}/{max_failures})")
                        
                        if consecutive_failures >= max_failures:
                            logger.critical("Watchdog encountered critical errors. Triggering restart.")
                            os._exit(1)
                    
                    if consecutive_failures > 0:
                        sleep_time = self.interval * (1 + consecutive_failures * 0.5)
                        await asyncio.sleep(min(sleep_time, 300))  # Cap at 5 minutes
                    else:
                        await asyncio.sleep(self.interval)
                        
        except ImportError:
            logger.error("aiohttp is required for watchdog functionality. Install with: pip install aiohttp")
            self.running = False
            return  # Exit early - don't block event loop with time.sleep

# ====================================
# ====================================

class AuctionProposal:
    """Represents a proposed auction awaiting admin approval"""
    def __init__(self, proposal_id: int, creator_id: int, creator_name: str):
        self.id = proposal_id
        self.creator_id = creator_id
        self.creator_name = creator_name
        self.name = ""
        self.teams = []  # List of team names
        self.purse = 0  # Same purse for all teams
        self.base_price = 0  # Base price for all players
        self.status = "pending"  # pending, approved, rejected
        self.created_at = datetime.now()
        self.admin_response_at = None
        self.admin_id = None
        self.admin_name = None
        
class ApprovedAuction:
    """Represents an approved auction ready for hosting"""
    def __init__(self, auction_id: int, proposal: AuctionProposal):
        self.id = auction_id
        self.name = proposal.name
        self.creator_id = proposal.creator_id
        self.creator_name = proposal.creator_name
        self.teams = proposal.teams
        self.purse = proposal.purse
        self.base_price = proposal.base_price
        self.status = "setup"  # setup, captain_reg, player_reg, ready, active, completed
        self.created_at = proposal.created_at
        self.approved_at = datetime.now()
        self.group_chat_id = None
        
        self.registered_captains = {}  # Dict[user_id, CaptainRegistration]
        self.approved_captains = {}    # Dict[user_id, ApprovedCaptain]
        self.registered_players = {}   # Dict[user_id, PlayerRegistration]
        self.approved_players = {}     # Dict[user_id, ApprovedPlayer]
        
        self.current_player = None
        self.current_player_index = 0
        self.current_bids = {}
        self.highest_bidder = None
        self.highest_bid = self.base_price
        self.player_queue = []
        self.sold_players = {}
        self.bidding_active = False
        self.unsold_players = {}
        self.countdown_seconds = 30
        self._bid_lock = threading.Lock()  # Proper threading lock for race condition prevention
        self.last_bid_time = None  # Track timing
        self.is_paused = False
        self.randomize_players = True

class CaptainRegistration:
    """Captain registration awaiting host approval"""
    def __init__(self, user_id: int, name: str, team_name: str):
        self.user_id = user_id
        self.name = name
        self.team_name = team_name
        self.registered_at = datetime.now()
        self.status = "pending"  # pending, approved, rejected

class ApprovedCaptain:
    """Approved captain with team and budget"""
    def __init__(self, user_id: int, name: str, team_name: str, purse: int):
        self.user_id = user_id
        self.name = name
        self.team_name = team_name
        self.purse = purse
        self.spent = 0
        self.players = []  # List of bought players
        self.approved_at = datetime.now()

class PlayerRegistration:
    """Player registration awaiting host approval"""
    def __init__(self, user_id: int, name: str, username: str = None):
        self.user_id = user_id
        self.name = name
        self.username = username
        self.registered_at = datetime.now()
        self.status = "pending"  # pending, approved, rejected

class ApprovedPlayer:
    """Approved player in auction pool"""
    def __init__(self, user_id: int, name: str, base_price: int, username: str = None):
        self.user_id = user_id
        self.name = name
        self.username = username
        self.base_price = base_price
        self.current_bid = 0
        self.current_bidder_id = None
        self.current_bidder_name = None
        self.winning_team = None
        self.sold_price = 0
        self.is_sold = False
        self.approved_at = datetime.now()

class AuctionRegistrationState:
    """Tracks the current registration state"""
    def __init__(self):
        self.step = "name"  # name, teams, purse, base_price, complete
        self.data = {}

class ArenaOfChampionsBot:
    def __init__(self):
        self.roast_cache = ThreadSafeDict({
            'lines': [],
            'last_updated': 0,
            'cache_duration': 300,  # 5 minutes cache
            'usage_counts': ThreadSafeCounter()  # Thread-safe counter for roast usage
        })
        
        self.leaderboard_cache = ThreadSafeDict({
            'data': [],
            'last_updated': 0,
            'cache_duration': 60,  # 1 minute cache for leaderboard
            'invalidated_at': 0  # For manual invalidation on game completion
        })
        
        self.profile_cache = {
            'data': ThreadSafeDict(),  # {user_id: profile_data} - thread-safe
            'last_updated': ThreadSafeDict(),  # {user_id: timestamp} - thread-safe
            'data_version': ThreadSafeDict(),  # {user_id: data_hash} - thread-safe
            'cache_duration': 120,  # 2 minutes cache for profiles
            'global_invalidated_at': 0  # Global invalidation timestamp
        }
        
        self.goat_cache = {
            'data': None,
            'date': None,
            'last_updated': 0,
            'cache_duration': 3600  # 1 hour cache for GOAT
        }
        
        self.bot_token = os.getenv('BOT_TOKEN')
        self.db_url = os.getenv('DATABASE_URL')
        self.super_admin_id = int(os.getenv('SUPER_ADMIN_ID', '0'))  # Creator/Super Admin
        self.admin_ids = [int(x.strip()) for x in os.getenv('ADMIN_IDS', '').split(',') if x.strip()]
        
        if not self.bot_token:
            raise ValueError("BOT_TOKEN environment variable is required")
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable is required")
        
        self.logs_bot_token = os.getenv('LOGS_BOT_TOKEN')
        self.logs_chat_id = int(os.getenv('LOGS_CHAT_ID', '0'))
        self.logs_enabled = bool(os.getenv('LOGS_ENABLED', 'True').lower() in ['true', '1', 'yes'])
        
        if self.logs_enabled and not self.logs_bot_token:
            logger.warning("LOGS_BOT_TOKEN not set but logs are enabled. Disabling logs.")
            self.logs_enabled = False
        
        try:
            self.db_pool = psycopg2.pool.ThreadedConnectionPool(
                5, 25,  # Increased pool size for better performance
                self.db_url,
                connect_timeout=15,  # Increased timeout
                application_name="Arena_Of_Champions_Bot",
                keepalives_idle=600,  # Keep connections alive for 10 minutes
                keepalives_interval=30,  # Send keepalive every 30 seconds
                keepalives_count=3  # Try 3 times before considering connection dead
            )
            logger.info("Database connection pool initialized")
            
            self._initialize_database_tables()
            self._setup_migrations()
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            self.db_pool = None
        
        self.api_wrapper = None
        
        self.achievement_emojis = {
            'winner': '🏆',
            'orange cap': '🟧',
            'purple cap': '🟪', 
            'mvp': '🏅'
        }
        
        self.guess_games = ThreadSafeDict()  # {user_id: game_state}
        
        self.guess_difficulties = {
            'beginner': {'emoji': '🟢', 'range': (1, 20), 'attempts': 6, 'time_limit': 30, 'multiplier': 1.0},
            'easy': {'emoji': '🔵', 'range': (1, 50), 'attempts': 8, 'time_limit': 60, 'multiplier': 1.2},
            'medium': {'emoji': '🟡', 'range': (1, 100), 'attempts': 7, 'time_limit': 60, 'multiplier': 1.5},
            'hard': {'emoji': '🟠', 'range': (1, 200), 'attempts': 8, 'time_limit': 90, 'multiplier': 2.0},
            'expert': {'emoji': '🔴', 'range': (1, 500), 'attempts': 10, 'time_limit': 90, 'multiplier': 3.0}
        }
        
        self.auction_proposals = {}      # Dict[proposal_id, AuctionProposal] 
        self.approved_auctions = {}      # Dict[auction_id, ApprovedAuction]
        self.registration_states = {}    # Dict[user_id, AuctionRegistrationState]
        self.proposal_counter = 0
        self.auction_counter = 0
        self.bid_timers = ThreadSafeDict()        # {session_id: timer_task}
        
    def _initialize_database_tables(self) -> None:
        """Initialize required database tables that might be missing"""
        try:
            conn = self.get_db_connection()
            if conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS nightmare_games (
                        game_id SERIAL PRIMARY KEY,
                        player_telegram_id BIGINT NOT NULL,
                        player_name VARCHAR(255),
                        original_number INTEGER,
                        current_number INTEGER,
                        shift_seed INTEGER,
                        encoded_hint TEXT,
                        hint_type VARCHAR(50),
                        decoded_hint TEXT,
                        attempts_used INTEGER DEFAULT 0,
                        max_attempts INTEGER DEFAULT 10,
                        game_range_min INTEGER DEFAULT 1,
                        game_range_max INTEGER DEFAULT 100,
                        is_completed BOOLEAN DEFAULT FALSE,
                        is_won BOOLEAN DEFAULT FALSE,
                        guess_history TEXT DEFAULT '[]',
                        result VARCHAR(50),
                        completed_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                try:
                    cursor.execute("ALTER TABLE nightmare_games ADD COLUMN IF NOT EXISTS result VARCHAR(50)")
                except Exception:
                    pass  # Column might already exist
                
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_nightmare_player_telegram_id ON nightmare_games(player_telegram_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_nightmare_completed ON nightmare_games(is_completed)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_nightmare_created_at ON nightmare_games(created_at)")
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS banned_users (
                        id SERIAL PRIMARY KEY,
                        telegram_id BIGINT UNIQUE NOT NULL,
                        username VARCHAR(255),
                        display_name VARCHAR(255),
                        banned_by_id BIGINT NOT NULL,
                        banned_by_name VARCHAR(255) NOT NULL,
                        ban_reason TEXT DEFAULT 'No reason provided',
                        banned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_banned_users_telegram_id ON banned_users(telegram_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_banned_users_active ON banned_users(is_active)")
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS daily_leaderboard_entries (
                        id SERIAL PRIMARY KEY,
                        player_id INTEGER,
                        player_telegram_id BIGINT NOT NULL,
                        player_name VARCHAR(255) NOT NULL,
                        leaderboard_date DATE NOT NULL DEFAULT CURRENT_DATE,
                        game_type VARCHAR(20) NOT NULL,
                        
                        -- Chase game daily stats
                        chase_games_played INTEGER DEFAULT 0,
                        chase_best_score INTEGER DEFAULT 0,
                        chase_best_level INTEGER DEFAULT 0,
                        chase_total_score INTEGER DEFAULT 0,
                        
                        -- Guess game daily stats  
                        guess_games_played INTEGER DEFAULT 0,
                        guess_best_score INTEGER DEFAULT 0,
                        guess_total_score INTEGER DEFAULT 0,
                        guess_games_won INTEGER DEFAULT 0,
                        
                        -- Timestamps
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        -- Ensure one entry per player per game type per day
                        UNIQUE(player_telegram_id, leaderboard_date, game_type)
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS daily_leaderboard_rewards (
                        id SERIAL PRIMARY KEY,
                        reward_date DATE NOT NULL,
                        game_type VARCHAR(20) NOT NULL,
                        player_id INTEGER,
                        player_telegram_id BIGINT NOT NULL,
                        player_name VARCHAR(255) NOT NULL,
                        rank_position INTEGER NOT NULL,
                        daily_score INTEGER NOT NULL,
                        shard_reward INTEGER NOT NULL,
                        rewarded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        -- Unique constraint to prevent double rewards
                        UNIQUE(reward_date, game_type, player_telegram_id)
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS daily_leaderboard_status (
                        id SERIAL PRIMARY KEY,
                        status_date DATE NOT NULL UNIQUE,
                        chase_rewards_distributed BOOLEAN DEFAULT FALSE,
                        guess_rewards_distributed BOOLEAN DEFAULT FALSE,
                        chase_rewards_at TIMESTAMP,
                        guess_rewards_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_lb_date ON daily_leaderboard_entries(leaderboard_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_lb_player ON daily_leaderboard_entries(player_telegram_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_lb_game_type ON daily_leaderboard_entries(game_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_lb_chase_score ON daily_leaderboard_entries(chase_total_score DESC)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_lb_guess_score ON daily_leaderboard_entries(guess_total_score DESC)")
                
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_rewards_date ON daily_leaderboard_rewards(reward_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_rewards_player ON daily_leaderboard_rewards(player_telegram_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_rewards_game_type ON daily_leaderboard_rewards(game_type)")
                
                cursor.execute("""
                    CREATE OR REPLACE FUNCTION update_daily_leaderboard_entry(
                        p_telegram_id BIGINT,
                        p_player_name VARCHAR(255),
                        p_game_type VARCHAR(20),
                        p_score INTEGER,
                        p_level INTEGER DEFAULT 1,
                        p_won BOOLEAN DEFAULT TRUE
                    ) RETURNS BOOLEAN AS $$
                    DECLARE
                        player_record RECORD;
                        today_date DATE := CURRENT_DATE;
                    BEGIN
                        -- Get player ID (optional, can be NULL if no players table)
                        SELECT id INTO player_record FROM players WHERE telegram_id = p_telegram_id;
                        
                        -- Update or insert daily entry based on game type
                        IF p_game_type = 'chase' THEN
                            INSERT INTO daily_leaderboard_entries (
                                player_id, player_telegram_id, player_name, leaderboard_date, game_type,
                                chase_games_played, chase_best_score, chase_best_level, chase_total_score
                            ) VALUES (
                                COALESCE(player_record.id, NULL), p_telegram_id, p_player_name, today_date, p_game_type,
                                1, p_score, p_level, p_score
                            )
                            ON CONFLICT (player_telegram_id, leaderboard_date, game_type) DO UPDATE SET
                                chase_games_played = daily_leaderboard_entries.chase_games_played + 1,
                                chase_best_score = GREATEST(daily_leaderboard_entries.chase_best_score, p_score),
                                chase_best_level = GREATEST(daily_leaderboard_entries.chase_best_level, p_level),
                                chase_total_score = daily_leaderboard_entries.chase_total_score + p_score,
                                updated_at = CURRENT_TIMESTAMP;
                                
                        ELSIF p_game_type = 'guess' THEN
                            INSERT INTO daily_leaderboard_entries (
                                player_id, player_telegram_id, player_name, leaderboard_date, game_type,
                                guess_games_played, guess_best_score, guess_total_score, guess_games_won
                            ) VALUES (
                                COALESCE(player_record.id, NULL), p_telegram_id, p_player_name, today_date, p_game_type,
                                1, p_score, p_score, CASE WHEN p_won THEN 1 ELSE 0 END
                            )
                            ON CONFLICT (player_telegram_id, leaderboard_date, game_type) DO UPDATE SET
                                guess_games_played = daily_leaderboard_entries.guess_games_played + 1,
                                guess_best_score = GREATEST(daily_leaderboard_entries.guess_best_score, p_score),
                                guess_total_score = daily_leaderboard_entries.guess_total_score + p_score,
                                guess_games_won = daily_leaderboard_entries.guess_games_won + CASE WHEN p_won THEN 1 ELSE 0 END,
                                updated_at = CURRENT_TIMESTAMP;
                                
                        -- Skip nightmare games for daily leaderboard as it doesn't have nightmare columns
                        -- Just update player stats for nightmare games
                        END IF;
                        
                        RETURN TRUE;
                    END;
                    $$ LANGUAGE plpgsql
                """)
                
                cursor.execute("BEGIN")
                try:
                    cursor.execute("DROP VIEW IF EXISTS daily_chase_leaderboard")
                    cursor.execute("DROP VIEW IF EXISTS daily_guess_leaderboard")
                    
                    cursor.execute("""
                        CREATE VIEW daily_chase_leaderboard AS
                        SELECT 
                            player_id,
                            player_telegram_id,
                            player_name,
                            chase_games_played as games_played,
                            chase_best_score as best_score,
                            chase_best_level as level_completed,
                            chase_total_score as total_score,
                            ROW_NUMBER() OVER (ORDER BY chase_best_level DESC, chase_best_score DESC, chase_games_played ASC) as rank
                        FROM daily_leaderboard_entries 
                        WHERE leaderboard_date = CURRENT_DATE AND game_type = 'chase'
                        ORDER BY chase_best_level DESC, chase_best_score DESC, chase_games_played ASC
                        LIMIT 10
                    """)
                
                    cursor.execute("""
                        CREATE VIEW daily_guess_leaderboard AS
                        SELECT 
                            player_id,
                            player_telegram_id,
                            player_name,
                            guess_games_played as games_played,
                            guess_best_score as best_score,
                            guess_total_score as total_score,
                            guess_games_won as games_won,
                            CASE 
                                WHEN guess_games_played > 0 THEN ROUND((guess_games_won * 100.0 / guess_games_played), 2)
                                ELSE 0.00 
                            END as win_percentage,
                            ROW_NUMBER() OVER (ORDER BY guess_total_score DESC, guess_games_won DESC, guess_best_score DESC) as rank
                        FROM daily_leaderboard_entries 
                        WHERE leaderboard_date = CURRENT_DATE AND game_type = 'guess'
                        ORDER BY guess_total_score DESC, guess_games_won DESC, guess_best_score DESC
                        LIMIT 10
                    """)
                    
                    cursor.execute("COMMIT")  # Commit the view creation transaction
                except Exception as view_error:
                    cursor.execute("ROLLBACK")
                    logger.error(f"Error creating views: {view_error}")
                    raise
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS auction_sessions (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        host_id BIGINT NOT NULL,
                        host_name VARCHAR(255) NOT NULL,
                        state VARCHAR(50) DEFAULT 'registration',
                        description TEXT,
                        max_teams INTEGER DEFAULT 8,
                        max_players_per_team INTEGER DEFAULT 11,
                        base_budget INTEGER DEFAULT 1000,
                        bid_increment INTEGER DEFAULT 25,
                        bid_timer INTEGER DEFAULT 60,
                        registration_phase BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS auction_captains (
                        id SERIAL PRIMARY KEY,
                        auction_id INTEGER REFERENCES auction_sessions(id) ON DELETE CASCADE,
                        user_id BIGINT NOT NULL,
                        username VARCHAR(255),
                        display_name VARCHAR(255) NOT NULL,
                        team_name VARCHAR(255) NOT NULL,
                        remaining_budget INTEGER NOT NULL,
                        current_team_size INTEGER DEFAULT 0,
                        joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(auction_id, user_id),
                        UNIQUE(auction_id, team_name)
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS auction_players (
                        id SERIAL PRIMARY KEY,
                        auction_id INTEGER REFERENCES auction_sessions(id) ON DELETE CASCADE,
                        user_id BIGINT NOT NULL,
                        username VARCHAR(255),
                        display_name VARCHAR(255) NOT NULL,
                        registration_message TEXT,
                        is_sold BOOLEAN DEFAULT FALSE,
                        sold_to_captain_id INTEGER REFERENCES auction_captains(id),
                        sold_price INTEGER DEFAULT 0,
                        registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        sold_at TIMESTAMP,
                        UNIQUE(auction_id, user_id)
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS auction_bids (
                        id SERIAL PRIMARY KEY,
                        auction_id INTEGER REFERENCES auction_sessions(id) ON DELETE CASCADE,
                        player_id INTEGER REFERENCES auction_players(id) ON DELETE CASCADE,
                        captain_id INTEGER REFERENCES auction_captains(id) ON DELETE CASCADE,
                        bid_amount INTEGER NOT NULL,
                        is_winning BOOLEAN DEFAULT FALSE,
                        bid_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS auction_notifications (
                        id SERIAL PRIMARY KEY,
                        auction_id INTEGER REFERENCES auction_sessions(id) ON DELETE CASCADE,
                        user_id BIGINT NOT NULL,
                        role VARCHAR(20) NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        subscribed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_auction_sessions_host ON auction_sessions(host_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_auction_sessions_state ON auction_sessions(state)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_auction_captains_auction ON auction_captains(auction_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_auction_captains_user ON auction_captains(user_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_auction_players_auction ON auction_players(auction_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_auction_players_user ON auction_players(user_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_auction_bids_auction ON auction_bids(auction_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_auction_bids_player ON auction_bids(player_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_auction_notifications_user ON auction_notifications(user_id)")
                
                conn.commit()
                cursor.close()
                self.return_db_connection(conn)
                logger.info("Database tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database tables: {e}")
    
    def _setup_migrations(self):
        """Setup and run database migrations"""
        try:
            migration_system = DatabaseMigration(self.db_pool)
            
            migration_system.add_migration(
                version=1,
                name="Add migration tracking",
                sql="SELECT 1 -- Migration system is self-initializing",
            )
            
            migration_system.add_migration(
                version=2,
                name="Add shards columns to players",
                sql="""
                    ALTER TABLE players ADD COLUMN IF NOT EXISTS current_shards INTEGER DEFAULT 0;
                    ALTER TABLE players ADD COLUMN IF NOT EXISTS total_shards_earned INTEGER DEFAULT 0;
                """
            )
            
            migration_system.run_migrations()
            
        except Exception as e:
            logger.error(f"Error setting up migrations: {e}")
        
    def check_cooldown(self, user_id: int, command_type: str, cooldown_seconds: int = 10) -> tuple[bool, int]:
        """Check if user is on cooldown for a command. Returns (is_on_cooldown, seconds_remaining)"""
        with self.get_db_cursor() as cursor_result:
            if cursor_result is None:
                return False, 0  # Allow if DB error
            cursor, conn = cursor_result
            
            cursor.execute("""
                SELECT last_used FROM cooldowns 
                WHERE user_id = %s AND command_type = %s
            """, (user_id, command_type))
            
            result = cursor.fetchone()
            if not result:
                return False, 0
            
            last_used = result[0]
            time_diff = (time.time() - last_used.timestamp())
            
            if time_diff < cooldown_seconds:
                return True, int(cooldown_seconds - time_diff)
            
            return False, 0
    
    def update_cooldown(self, user_id: int, command_type: str):
        """Update user's cooldown for a command"""
        with self.get_db_cursor() as cursor_result:
            if cursor_result is None:
                return
            cursor, conn = cursor_result
            
            cursor.execute("""
                INSERT INTO cooldowns (user_id, command_type, last_used)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, command_type)
                DO UPDATE SET last_used = CURRENT_TIMESTAMP
            """, (user_id, command_type))
        
    async def execute_with_retry(self, operation, *args, retries=2, timeout=5):
        """Execute database operation with retry logic for auction system"""
        for attempt in range(retries + 1):
            try:
                if not callable(operation):
                    raise ValueError("Operation must be callable")
                
                return await asyncio.wait_for(operation(*args), timeout=timeout)
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Database operation timeout, retrying... ({attempt + 1}/{retries + 1})")
                    await asyncio.sleep(0.1 * (attempt + 1))  # Progressive backoff
                    continue
                else:
                    logger.error("Database operation failed after all retries due to timeout")
                    raise
            except Exception as e:
                error_str = str(e).lower()
                if attempt < retries and any(keyword in error_str for keyword in ["timeout", "connection", "pool"]):
                    logger.warning(f"Database error: {e}, retrying... ({attempt + 1}/{retries + 1})")
                    await asyncio.sleep(0.1 * (attempt + 1))
                    continue
                else:
                    logger.error(f"Database operation failed: {e}")
                    raise

    def get_db_connection(self) -> Optional['psycopg2.extensions.connection']:
        """Create database connection from pool with better error handling"""
        try:
            if self.db_pool:
                try:
                    pool_size = self.db_pool.maxconn - len(self.db_pool._pool)
                    if pool_size > 22:  # Warn if pool usage is high (>87% of 25)
                        logger.warning(f"High pool usage: {pool_size}/{self.db_pool.maxconn} connections in use")
                        try:
                            async def send_pool_usage_log():
                                try:
                                    await self.send_admin_log(
                                        'system_event',
                                        f"High database pool usage: {pool_size}/{self.db_pool.maxconn} connections in use ({pool_size/self.db_pool.maxconn*100:.1f}%)",
                                        None,
                                        "System"
                                    )
                                except Exception as e:
                                    logger.debug(f"Could not send pool usage log: {e}")
                            
                            deferred_scheduler.schedule_task(send_pool_usage_log())
                        except Exception as e:
                            logger.debug(f"Could not schedule pool usage log: {e}")
                except (AttributeError, TypeError) as e:
                    logger.debug(f"Could not retrieve pool status: {e}")
                
                conn = self.db_pool.getconn()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()
                    return conn
                else:
                    logger.error("Database connection pool exhausted: connection pool exhausted")
                    try:
                        async def send_pool_exhaustion_log():
                            try:
                                await self.send_admin_log(
                                    'error',
                                    "Database connection pool EXHAUSTED - all connections in use",
                                    None,
                                    "System"
                                )
                            except Exception as e:
                                logger.debug(f"Could not send pool exhaustion log: {e}")
                        
                        deferred_scheduler.schedule_task(send_pool_exhaustion_log())
                    except Exception as e:
                        logger.debug(f"Could not schedule pool exhaustion log: {e}")
                    if self.reset_connection_pool():
                        logger.info("Retrying connection after pool reset...")
                        conn = self.db_pool.getconn()
                        if conn:
                            cursor = conn.cursor()
                            cursor.execute("SELECT 1")
                            cursor.close()
                            return conn
                    return None
            else:
                return psycopg2.connect(
                    self.db_url,
                    connect_timeout=10,
                    application_name="AOC_Bot_Fallback"
                )
        except psycopg2.pool.PoolError as e:
            logger.error(f"Database connection pool exhausted: {e}")
            if "connection pool exhausted" in str(e).lower():
                logger.warning("Attempting to reset exhausted connection pool...")
                if self.reset_connection_pool():
                    try:
                        conn = self.db_pool.getconn()
                        if conn:
                            cursor = conn.cursor()
                            cursor.execute("SELECT 1")
                            cursor.close()
                            return conn
                    except Exception as retry_e:
                        logger.error(f"Retry after pool reset failed: {retry_e}")
            return None
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return None
    
    def return_db_connection(self, conn):
        """Return connection to pool"""
        try:
            if self.db_pool and conn:
                self.db_pool.putconn(conn)
        except Exception as e:
            logger.error(f"Error returning connection to pool: {e}")
    
    def reset_connection_pool(self):
        """Reset the connection pool when it gets exhausted"""
        try:
            logger.warning("Resetting connection pool due to exhaustion...")
            if self.db_pool:
                self.db_pool.closeall()
            
            self.db_pool = psycopg2.pool.ThreadedConnectionPool(
                3, 40,
                self.db_url,
                connect_timeout=10,
                application_name="Arena_Of_Champions_Bot_Reset"
            )
            logger.info("Connection pool reset successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reset connection pool: {e}")
            return False
    
    def get_connection_pool_status(self) -> Dict[str, Union[int, bool]]:
        """Get current connection pool status for monitoring"""
        if not self.db_pool:
            return "Pool not initialized"
        try:
            return f"Pool connections: {len(self.db_pool._pool)} available, {len(self.db_pool._used)} in use"
        except:
            return "Pool status unavailable"
    
    @contextmanager
    def get_db_cursor(self):
        """Context manager for database operations"""
        conn = self.get_db_connection()
        if not conn:
            yield None
            return
        try:
            cursor = conn.cursor()
            yield cursor, conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database operation error: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            self.return_db_connection(conn)
    
    @contextmanager 
    def get_db_connection_ctx(self):
        """Context manager for database connections that ensures proper cleanup"""
        conn = self.get_db_connection()
        if not conn:
            yield None
            return
        try:
            yield conn
        finally:
            self.return_db_connection(conn)
    
    def init_database(self):
        """Initialize database tables with timeout protection"""
        logger.info("Starting database initialization...")
        
        logger.info("Using simplified database initialization...")
        logger.info("Please run complete_database_schema.sql manually for full setup")
        
        conn = self.get_db_connection()
        if not conn:
            logger.error("Failed to get database connection")
            return False
            
        try:
            logger.info("Testing database connection...")
            cursor = conn.cursor()
            
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            logger.info(f"Database connection successful: {version[:50]}...")
            
            logger.info("Creating essential database tables...")
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    id SERIAL PRIMARY KEY,
                    telegram_id BIGINT UNIQUE NOT NULL,
                    username VARCHAR(255),
                    display_name VARCHAR(255) NOT NULL,
                    title VARCHAR(255),
                    unlocked_levels TEXT[] DEFAULT ARRAY['beginner']::TEXT[],
                    highest_score INTEGER DEFAULT 0,
                    total_score INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cooldowns (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    command_type VARCHAR(50) NOT NULL,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, command_type)
                );
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS admins (
                    id SERIAL PRIMARY KEY,
                    telegram_id BIGINT UNIQUE NOT NULL,
                    username VARCHAR(255),
                    display_name VARCHAR(255),
                    added_by BIGINT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id SERIAL PRIMARY KEY,
                    chat_id BIGINT UNIQUE NOT NULL,
                    chat_type VARCHAR(50),
                    title VARCHAR(255),
                    username VARCHAR(255),
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS achievements (
                    id SERIAL PRIMARY KEY,
                    player_id INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
                    achievement_name VARCHAR(255) NOT NULL,
                    count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(player_id, achievement_name)
                );
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chase_games (
                    id SERIAL PRIMARY KEY,
                    telegram_id BIGINT NOT NULL,
                    player_name VARCHAR(255),
                    chat_id BIGINT,
                    final_score INTEGER DEFAULT 0,
                    max_level INTEGER DEFAULT 1,
                    game_outcome VARCHAR(50) DEFAULT 'completed',
                    game_duration INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS guess_games (
                    id SERIAL PRIMARY KEY,
                    player_id INTEGER,
                    telegram_id BIGINT NOT NULL,
                    player_name VARCHAR(255),
                    difficulty VARCHAR(50),
                    target_number INTEGER,
                    guesses_used INTEGER DEFAULT 0,
                    max_guesses INTEGER DEFAULT 10,
                    game_outcome VARCHAR(50) DEFAULT 'ongoing',
                    final_score INTEGER DEFAULT 0,
                    chat_id BIGINT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                );
            """)
            
            conn.commit()
            logger.info("All essential tables created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            logger.error("If tables are missing, run 'complete_database_schema_fixed.sql' manually")
            try:
                conn.rollback()
            except:
                pass
            return False
        finally:
            self.return_db_connection(conn)
    
    def get_all_admin_ids(self) -> list:
        """Get all admin IDs from all sources"""
        admin_ids = set()
        
        if self.super_admin_id:
            admin_ids.add(self.super_admin_id)
        
        admin_ids.update(self.admin_ids)
        
        try:
            with self.get_db_connection_ctx() as conn:
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT telegram_id FROM admins")
                    db_admins = cursor.fetchall()
                    cursor.close()
                    
                    for (admin_id,) in db_admins:
                        admin_ids.add(admin_id)
        except Exception as e:
            logger.warning(f"Error fetching database admins: {e}")
        
        return list(admin_ids)

    def is_admin(self, user_id: int) -> bool:
        """
        Check if user is admin with clear hierarchy:
        1. Super Admin (env: SUPER_ADMIN_ID) - highest authority
        2. Environment Admins (env: ADMIN_IDS) - permanent admins
        3. Database Admins (table: admins) - dynamic admins
        """
        if user_id == self.super_admin_id:
            logger.debug(f"User {user_id} verified as SUPER_ADMIN")
            return True
            
        if user_id in self.admin_ids:
            logger.debug(f"User {user_id} verified as ENV_ADMIN")
            return True
        
        try:
            with self.get_db_connection_ctx() as conn:
                if not conn:
                    logger.warning(f"Database unavailable for admin check user {user_id}, using env fallback")
                    return False
                
                cursor = conn.cursor()
                cursor.execute("SELECT telegram_id FROM admins WHERE telegram_id = %s", (user_id,))
                result = cursor.fetchone() is not None
                cursor.close()
                
                if result:
                    logger.debug(f"User {user_id} verified as DB_ADMIN")
                
                return result
        except Exception as e:
            logger.warning(f"Error checking database admin status for user {user_id}: {e}")
            return False
    
    @db_query(read_only=True, fallback=False)
    def is_banned(self, cursor, user_id: int) -> bool:
        """Check if user is banned"""
        cursor.execute(
            "SELECT id FROM banned_users WHERE telegram_id = %s AND is_active = TRUE",
            (user_id,)
        )
        result = cursor.fetchone()
        return result is not None
    
    @db_query(fallback=False)
    def ban_user(self, cursor, user_id: int, username: str, display_name: str, banned_by_id: int, banned_by_name: str, reason: str = "No reason provided") -> bool:
        """Ban a user from using the bot"""
        cursor.execute("""
            INSERT INTO banned_users (telegram_id, username, display_name, banned_by_id, banned_by_name, ban_reason)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (telegram_id) DO UPDATE SET
                is_active = TRUE,
                banned_by_id = EXCLUDED.banned_by_id,
                banned_by_name = EXCLUDED.banned_by_name,
                ban_reason = EXCLUDED.ban_reason,
                banned_at = CURRENT_TIMESTAMP
        """, (user_id, username, display_name, banned_by_id, banned_by_name, reason))
        return True
    
    @db_query(fallback=False)
    def unban_user(self, cursor, user_id: int) -> bool:
        """Unban a user"""
        cursor.execute(
            "UPDATE banned_users SET is_active = FALSE WHERE telegram_id = %s",
            (user_id,)
        )
        return cursor.rowcount > 0
    
    def is_super_admin(self, user_id: int) -> bool:
        """Check if user is super admin (creator) - highest authority"""
        return user_id == self.super_admin_id
    
    def is_env_admin(self, user_id: int) -> bool:
        """Check if user is environment-defined admin (permanent)"""
        return user_id in self.admin_ids
    
    def is_db_admin(self, user_id: int) -> bool:
        """Check if user is database-defined admin (dynamic)"""
        try:
            with self.get_db_connection_ctx() as conn:
                if not conn:
                    return False
                
                cursor = conn.cursor()
                cursor.execute("SELECT telegram_id FROM admins WHERE telegram_id = %s", (user_id,))
                result = cursor.fetchone() is not None
                cursor.close()
                return result
        except Exception as e:
            logger.error(f"Error checking database admin status: {e}")
            return False
    
    def get_admin_level(self, user_id: int) -> str:
        """Get admin level for user (for logging/debugging)"""
        if self.is_super_admin(user_id):
            return "SUPER_ADMIN"
        elif self.is_env_admin(user_id):
            return "ENV_ADMIN" 
        elif self.is_db_admin(user_id):
            return "DB_ADMIN"
        else:
            return "NOT_ADMIN"
    
    def sync_env_admins_to_db(self) -> bool:
        """Sync environment-defined admins to database (run on startup)"""
        if not self.admin_ids:
            logger.info("No environment admins to sync")
            return True
            
        try:
            with self.get_db_connection_ctx() as conn:
                if not conn:
                    logger.warning("Cannot sync admins: database unavailable")
                    return False
                
                cursor = conn.cursor()
                synced_count = 0
                
                for admin_id in self.admin_ids:
                    try:
                        cursor.execute("SELECT telegram_id FROM admins WHERE telegram_id = %s", (admin_id,))
                        if not cursor.fetchone():
                            cursor.execute("""
                                INSERT INTO admins (telegram_id, username, display_name, added_by)
                                VALUES (%s, %s, %s, %s)
                            """, (admin_id, "ENV_ADMIN", f"Environment Admin {admin_id}", self.super_admin_id))
                            synced_count += 1
                            logger.info(f"Synced environment admin {admin_id} to database")
                    except Exception as e:
                        logger.warning(f"Failed to sync admin {admin_id}: {e}")
                        continue
                
                conn.commit()
                cursor.close()
                
                if synced_count > 0:
                    logger.info(f"Successfully synced {synced_count} environment admins to database")
                else:
                    logger.debug("All environment admins already exist in database")
                
                return True
                
        except Exception as e:
            logger.error(f"Error syncing environment admins to database: {e}")
            return False

    def track_chat(self, chat_id: int, chat_type: str, title: str = None, username: str = None) -> bool:
        """Track chat information for broadcast purposes"""
        conn = self.get_db_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO chats (chat_id, chat_type, title, username, last_activity)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (chat_id) DO UPDATE SET
                    chat_type = EXCLUDED.chat_type,
                    title = EXCLUDED.title,
                    username = EXCLUDED.username,
                    is_active = TRUE,
                    last_activity = CURRENT_TIMESTAMP
            """, (chat_id, chat_type, title, username))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error tracking chat {chat_id}: {e}")
            conn.rollback()
            return False
        finally:
            self.return_db_connection(conn)

    def find_player_by_identifier(self, identifier: str) -> Optional[Dict]:
        """Find player by name, username, or telegram_id"""
        conn = self.get_db_connection()
        if not conn:
            return None
            
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            if identifier.isdigit():
                cursor.execute(
                    "SELECT * FROM players WHERE telegram_id = %s",
                    (int(identifier),)
                )
            elif identifier.startswith('@'):
                username = identifier[1:]  # Remove @
                cursor.execute(
                    "SELECT * FROM players WHERE username ILIKE %s",
                    (username,)
                )
            else:
                cursor.execute(
                    "SELECT * FROM players WHERE display_name ILIKE %s",
                    (identifier,)
                )
            
            player = cursor.fetchone()
            return dict(player) if player else None
            
        except Exception as e:
            logger.error(f"Error finding player: {e}")
            return None
        finally:
            self.return_db_connection(conn)
    
    def create_or_update_player(self, telegram_id: int, username: str, display_name: str) -> tuple[bool, bool]:
        """Create or update player record. Returns (success, is_new_user)"""
        
        if not InputValidator.display_name(display_name):
            logger.warning(f"Invalid display name: {display_name}")
            return False, False
        
        if username and not InputValidator.username(username):
            logger.warning(f"Invalid username: {username}")
            username = None  # Set to None if invalid
        
        with self.get_db_cursor() as cursor_result:
            if cursor_result is None:
                return False, False
            cursor, conn = cursor_result
            
            cursor.execute("SELECT id FROM players WHERE telegram_id = %s", (telegram_id,))
            existing_user = cursor.fetchone()
            is_new_user = existing_user is None
            
            cursor.execute("""
                INSERT INTO players (telegram_id, username, display_name)
                VALUES (%s, %s, %s)
                ON CONFLICT (telegram_id)
                DO UPDATE SET 
                    username = EXCLUDED.username,
                    display_name = EXCLUDED.display_name,
                    updated_at = CURRENT_TIMESTAMP
            """, (telegram_id, username, display_name))
            
            return True, is_new_user
    
    def add_achievement(self, player_id: int, achievement: str, performed_by: int = None, username: str = None) -> bool:
        """Add achievement to player - returns False if player doesn't exist"""
        conn = self.get_db_connection()
        if not conn:
            return False
            
        try:
            cursor = conn.cursor()
            
            cursor.execute("SELECT display_name FROM players WHERE id = %s", (player_id,))
            player_data = cursor.fetchone()
            if not player_data:
                logger.user_action("Player not found - user not registered", 
                                user_id=player_id, status="not_registered")
                return False
            
            player_name = player_data[0]
            
            cursor.execute(
                "SELECT count FROM achievements WHERE player_id = %s AND achievement_name = %s",
                (player_id, achievement.lower())
            )
            
            existing = cursor.fetchone()
            
            is_new_achievement = not existing
            
            if existing:
                cursor.execute(
                    "UPDATE achievements SET count = count + 1, updated_at = CURRENT_TIMESTAMP WHERE player_id = %s AND achievement_name = %s",
                    (player_id, achievement.lower())
                )
            else:
                cursor.execute(
                    "INSERT INTO achievements (player_id, achievement_name) VALUES (%s, %s)",
                    (player_id, achievement.lower())
                )
            
            conn.commit()
            
            if is_new_achievement:
                try:
                    cursor.execute("SELECT telegram_id FROM players WHERE id = %s", (player_id,))
                    telegram_result = cursor.fetchone()
                    
                    if telegram_result:
                        telegram_id = telegram_result[0]
                        shard_reward = 50  # Reduced from 100 - Base reward for achievements
                        
                        achievement_lower = achievement.lower()
                        if any(keyword in achievement_lower for keyword in ['winner', 'champion', 'mvp', 'goat']):
                            shard_reward = 100  # Reduced from 200
                        elif any(keyword in achievement_lower for keyword in ['legend', 'master', 'elite']):
                            shard_reward = 75   # Reduced from 150
                        
                        shard_success = self.award_shards(
                            telegram_id,
                            shard_reward,
                            'achievement',
                            f'New achievement: {achievement}',
                            performed_by
                        )
                        
                        if shard_success:
                            logger.achievement_event(
                                f"Awarded {shard_reward} shards for new achievement",
                                user_id=player_id,
                                achievement_name=achievement,
                                shard_reward=shard_reward
                            )
                        
                except Exception as e:
                    logger.error(f"Error awarding achievement shards: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding achievement: {e}")
            conn.rollback()
            return False
        finally:
            self.return_db_connection(conn)

    @db_query(fallback=False)
    def add_or_increment_achievement(self, cursor, player_id: int, achievement: str, inc: int = 1) -> bool:
        """Safely add or increment achievement using upsert (requires UNIQUE constraint)"""
        cursor.execute("SELECT id FROM players WHERE id = %s", (player_id,))
        if not cursor.fetchone():
            logger.info(f"Player {player_id} not found - user not registered")
            return False
        
        cursor.execute("""
            INSERT INTO achievements (player_id, achievement_name, count)
            VALUES (%s, %s, %s)
            ON CONFLICT (player_id, achievement_name)
            DO UPDATE SET count = achievements.count + EXCLUDED.count,
                          updated_at = CURRENT_TIMESTAMP
        """, (player_id, achievement.lower(), inc))
        
        return True

    @db_query(fallback=False)
    def remove_achievement(self, cursor, player_id: int, achievement: str) -> bool:
        """Remove one instance of achievement from player"""
        cursor.execute(
            "SELECT count FROM achievements WHERE player_id = %s AND achievement_name = %s",
            (player_id, achievement.lower())
        )
        
        result = cursor.fetchone()
        
        if not result:
            return False  # Achievement doesn't exist
        
        current_count = result[0]
        
        if current_count > 1:
            cursor.execute(
                "UPDATE achievements SET count = count - 1, updated_at = CURRENT_TIMESTAMP WHERE player_id = %s AND achievement_name = %s",
                (player_id, achievement.lower())
            )
        else:
            cursor.execute(
                "DELETE FROM achievements WHERE player_id = %s AND achievement_name = %s",
                (player_id, achievement.lower())
            )
        
        return True
    
    def set_player_title(self, player_id: int, title: str) -> bool:
        """Set player title"""
        conn = self.get_db_connection()
        if not conn:
            return False
            
        try:
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE players SET title = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                (title, player_id)
            )
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error setting title: {e}")
            conn.rollback()
            return False
        finally:
            self.return_db_connection(conn)
    
    @db_query(read_only=True, fallback=[])
    def get_player_achievements(self, cursor, player_id: int) -> List[Tuple[str, int]]:
        """Get all achievements for a player"""
        cursor.execute(
            "SELECT achievement_name, count FROM achievements WHERE player_id = %s ORDER BY achievement_name",
            (player_id,)
        )
        return cursor.fetchall()
    
    def get_proper_achievement_name(self, achievement_name: str) -> str:
        """Get achievement name in uppercase for consistent display"""
        # Simply return uppercase version - works for all achievements
        # No need for mapping, handles any achievement name automatically
        return achievement_name.upper().strip()

    def get_achievement_emoji(self, achievement: str) -> str:
        """Get emojis for achievement - supports multiple keywords and returns all matching emojis"""
        achievement_lower = achievement.lower().strip()
        emojis = []
        
        if 'winner' in achievement_lower:
            emojis.append('🏆')
        
        if 'orange' in achievement_lower:
            emojis.append('🟧')
            
        if 'purple' in achievement_lower:
            emojis.append('🟪')
            
        if 'mvp' in achievement_lower:
            emojis.append('🏅')
        
        return ' '.join(emojis) if emojis else ''

    def add_admin(self, telegram_id: int, username: str, display_name: str, added_by: int) -> bool:
        """Add new admin to database"""
        conn = self.get_db_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO admins (telegram_id, username, display_name, added_by)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (telegram_id) DO NOTHING
            """, (telegram_id, username, display_name, added_by))
            
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error adding admin: {e}")
            conn.rollback()
            return False
        finally:
            self.return_db_connection(conn)
    
    def remove_admin(self, telegram_id: int) -> bool:
        """Remove admin from database"""
        conn = self.get_db_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM admins WHERE telegram_id = %s", (telegram_id,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error removing admin: {e}")
            conn.rollback()
            return False
        finally:
            self.return_db_connection(conn)
    
    def get_all_admins(self) -> List[Dict]:
        """Get all admins from database"""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM admins ORDER BY created_at")
            admins = cursor.fetchall()
            return [dict(admin) for admin in admins]
        except Exception as e:
            logger.error(f"Error getting admins: {e}")
            return []
        finally:
            self.return_db_connection(conn)
    
    def record_chase_game(self, telegram_id: int, player_name: str, chat_id: int, final_score: int = 0, max_level: int = 1, game_outcome: str = 'completed', game_duration: int = None) -> bool:
        """Record a completed chase game for statistics"""
        logger.info(f"Attempting to record chase game: player={telegram_id}, name={player_name}, score={final_score}, level={max_level}, outcome={game_outcome}")
        
        conn = self.get_db_connection()
        if not conn:
            logger.error("Failed to get database connection for chase game recording")
            return False
        
        try:
            cursor = conn.cursor()
            
            player_id = None
            cursor.execute("SELECT id FROM players WHERE telegram_id = %s", (telegram_id,))
            player_data = cursor.fetchone()
            if player_data:
                player_id = player_data[0]
                logger.info(f"Found registered player ID: {player_id}")
            else:
                logger.info(f"Player {telegram_id} not registered, using telegram_id")
            
            cursor.execute("""
                INSERT INTO chase_games (
                    player_id, telegram_id, player_name, chat_id, 
                    final_score, max_level, game_outcome, game_duration, completed_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """, (player_id, telegram_id, player_name, chat_id, final_score, max_level, game_outcome, game_duration))
            
            conn.commit()
            logger.info(f"Successfully recorded chase game for player {telegram_id}")
            
            self.invalidate_leaderboard_cache()
            return True
            
        except Exception as e:
            logger.error(f"Error recording chase game: {e}")
            conn.rollback()
            return False
        finally:
            self.return_db_connection(conn)

    @db_query(read_only=True, fallback={'total_games': 0, 'total_players': 0, 'avg_score': 0, 'high_score': 0, 'games_24h': 0, 'games_7d': 0})
    def get_chase_game_stats(self, cursor) -> dict:
        """Get overall chase game statistics"""
        cursor.execute("SELECT COUNT(*) FROM chase_games")
        total_games = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT COUNT(DISTINCT telegram_id) FROM chase_games")
        total_players = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT AVG(final_score) FROM chase_games WHERE final_score > 0")
        avg_score_result = cursor.fetchone()
        avg_score = round(avg_score_result[0]) if avg_score_result[0] else 0
        
        cursor.execute("SELECT MAX(final_score) FROM chase_games")
        high_score = cursor.fetchone()[0] or 0
        
        cursor.execute("""
            SELECT COUNT(*) FROM chase_games 
            WHERE created_at >= NOW() - INTERVAL '24 hours'
        """)
        games_24h = cursor.fetchone()[0] or 0
        
        cursor.execute("""
            SELECT COUNT(*) FROM chase_games 
            WHERE created_at >= NOW() - INTERVAL '7 days'
        """)
        games_7d = cursor.fetchone()[0] or 0
        
        return {
            'total_games': total_games,
            'total_players': total_players,
            'avg_score': avg_score,
            'high_score': high_score,
            'games_24h': games_24h,
            'games_7d': games_7d
        }

    def invalidate_leaderboard_cache(self):
        """Invalidate leaderboard cache to force refresh on next access"""
        self.leaderboard_cache['invalidated_at'] = time.time()
        self.leaderboard_cache['last_updated'] = 0
        logger.debug("Leaderboard cache invalidated")

    def get_chase_leaderboard(self, limit: int = 10) -> List[Dict]:
        """
        Get chase game leaderboard based on best performance per player.
        Ranking: Highest Level > Most Runs at that Level > Fewer Balls Used
        """
        current_time = time.time()
        cache_valid = (
            current_time - self.leaderboard_cache['last_updated'] < self.leaderboard_cache['cache_duration']
            and self.leaderboard_cache['last_updated'] > self.leaderboard_cache['invalidated_at']
            and len(self.leaderboard_cache['data']) > 0
        )
        
        if cache_valid:
            return self.leaderboard_cache['data'][:limit]
        
        with self.get_db_cursor() as cursor_result:
            if cursor_result is None:
                return []
            cursor, conn = cursor_result
            
            cursor.execute("""
                SELECT 
                    telegram_id,
                    COALESCE(player_name, 'Unknown') as player_name,
                    COALESCE(max_level, 0) as highest_level,
                    COALESCE(final_score, 0) as runs_scored,
                    COALESCE(game_duration, 0) as balls_faced,
                    CASE 
                        WHEN COALESCE(game_duration, 0) > 0 THEN 
                            CAST(ROUND((COALESCE(final_score, 0)::NUMERIC / game_duration) * 100, 1) AS FLOAT)
                        ELSE 0.0
                    END as strike_rate
                FROM chase_games
                WHERE game_duration IS NOT NULL AND game_duration > 0
                ORDER BY 
                    max_level DESC, 
                    final_score DESC, 
                    game_duration ASC
                LIMIT %s
            """, (limit,))
            
            results = cursor.fetchall()
            
            leaderboard = []
            for row in results:
                telegram_id = row[0] or 0
                display_name = row[1] or f"User{telegram_id}"
                highest_level = max(0, row[2] or 0)  # Ensure positive
                runs_scored = max(0, row[3] or 0)    # Ensure positive
                balls_faced = max(0, row[4] or 0)    # Ensure positive
                strike_rate = max(0.0, row[5] or 0.0) if len(row) > 5 else 0.0
                
                leaderboard.append({
                    'telegram_id': telegram_id,
                    'display_name': display_name,
                    'username': None,
                    'highest_level': highest_level,
                    'runs_scored': runs_scored,
                    'balls_faced': balls_faced,
                    'strike_rate': strike_rate
                })
            
            self.leaderboard_cache['data'] = leaderboard
            self.leaderboard_cache['last_updated'] = current_time
            
            return leaderboard[:limit]

    def get_player_chase_stats(self, player_id: int) -> dict:
        """Get individual player's chase game statistics with proper connection management"""
        default_stats = {'games_played': 0, 'highest_level': 0, 'highest_score': 0, 'best_sr': 0.0, 'rank': None}
        
        with self.get_db_connection_ctx() as conn:
            if not conn:
                logger.error("Failed to get database connection for chase stats")
                return default_stats
            
            try:
                cursor = conn.cursor()
                
                cursor.execute("SELECT telegram_id FROM players WHERE id = %s", (player_id,))
                player_result = cursor.fetchone()
                if not player_result:
                    logger.warning(f"Player with ID {player_id} not found")
                    return default_stats
                
                telegram_id = player_result[0]
                
                cursor.execute("SELECT COUNT(*) FROM chase_games WHERE telegram_id = %s", (telegram_id,))
                games_played = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT MAX(max_level) FROM chase_games WHERE telegram_id = %s", (telegram_id,))
                highest_level = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT MAX(final_score) FROM chase_games WHERE telegram_id = %s", (telegram_id,))
                highest_score = cursor.fetchone()[0] or 0
                
                cursor.execute("""
                    SELECT MAX(CASE WHEN max_level > 0 THEN final_score::float / max_level ELSE 0 END) 
                    FROM chase_games WHERE telegram_id = %s
                """, (telegram_id,))
                best_sr_result = cursor.fetchone()
                best_sr = round(best_sr_result[0], 1) if best_sr_result[0] else 0.0
                
                rank = None
                try:
                    cursor.execute("""
                        WITH ranked_players AS (
                            SELECT 
                                telegram_id,
                                ROW_NUMBER() OVER (
                                    ORDER BY MAX(max_level) DESC, 
                                            MAX(final_score) DESC
                                ) as rank
                            FROM chase_games 
                            GROUP BY telegram_id
                        )
                        SELECT rank FROM ranked_players WHERE telegram_id = %s
                    """, (telegram_id,))
                    rank_result = cursor.fetchone()
                    rank = rank_result[0] if rank_result else None
                except Exception as e:
                    logger.warning(f"Could not calculate rank for player {player_id}: {e}")
                
                cursor.close()
                return {
                    'games_played': games_played,
                    'highest_level': highest_level,
                    'highest_score': highest_score,
                    'best_sr': best_sr,
                    'rank': rank
                }
                
            except Exception as e:
                logger.error(f"Error getting player chase stats for player {player_id}: {e}")
                return default_stats

    def refresh_roast_cache(self) -> bool:
        """Refresh the roast cache from database with proper connection management"""
        with self.get_db_connection_ctx() as conn:
            if not conn:
                return False
            
            try:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT roast_line, usage_count, last_used_date
                    FROM roast_rotation 
                    ORDER BY usage_count ASC, RANDOM()
                """)
                
                roast_data = cursor.fetchall()
                
                lines_data = [row[0] for row in roast_data]
                usage_data = {row[0]: row[1] for row in roast_data}
                
                self.roast_cache.atomic_operation(
                    lambda cache_data: cache_data.update({
                        'lines': lines_data,
                        'last_updated': time.time()
                    })
                )
                
                self.roast_cache['usage_counts'].clear()
                self.roast_cache['usage_counts'].update(usage_data)
                
                cursor.close()
                logger.info(f"Roast cache refreshed with {len(roast_data)} lines")
                return True
                
            except Exception as e:
                logger.error(f"Error refreshing roast cache: {e}")
                return False
    
    def get_cached_roast_line(self) -> str:
        """Get next roast line using cache for better performance"""
        current_time = time.time()
        
        cache_expired = self.roast_cache.atomic_operation(
            lambda cache_data: (
                current_time - cache_data.get('last_updated', 0) > cache_data.get('cache_duration', 300) or
                not cache_data.get('lines', [])
            )
        )
        
        if cache_expired:
            if not self.refresh_roast_cache():
                return random.choice(GOAT_ROAST_LINES)
        
        cache_data = self.roast_cache.atomic_operation(
            lambda cache_data: {
                'lines': cache_data.get('lines', []).copy(),
                'usage_counts': cache_data.get('usage_counts')
            }
        )
        
        lines = cache_data['lines']
        usage_counts = cache_data['usage_counts']
        
        if not lines:
            return random.choice(GOAT_ROAST_LINES)
        
        min_usage = usage_counts.get_min_value()
        
        available_lines = usage_counts.get_keys_with_min_value()
        
        available_lines = [line for line in available_lines if line in lines]
        
        if not available_lines:
            available_lines = lines
        
        return random.choice(available_lines)
    
    def get_cached_profile_data(self, user_id: int) -> dict:
        """Get cached profile data with smart invalidation"""
        current_time = time.time()
        
        if (user_id in self.profile_cache['data'] and 
            user_id in self.profile_cache['last_updated']):
            
            cache_entry = {
                'timestamp': self.profile_cache['last_updated'][user_id],
                'data_hash': self.profile_cache['data_version'].get(user_id, '')
            }
            
            should_invalidate = SmartCacheManager.should_invalidate_cache(
                cache_entry, 
                current_time, 
                self.profile_cache['cache_duration'],
                self.profile_cache['global_invalidated_at']
            )
            
            if not should_invalidate:
                cached_data = self.profile_cache['data'][user_id]
                current_hash = SmartCacheManager.generate_data_hash(cached_data)
                cached_hash = self.profile_cache['data_version'].get(user_id, '')
                
                if current_hash == cached_hash:
                    logger.debug(f"Profile cache HIT for user {user_id}")
                    return cached_data
                else:
                    logger.debug(f"Profile cache INVALIDATED by data change for user {user_id}")
        
        logger.debug(f"Profile cache MISS for user {user_id} - fetching from database")
        try:
            player = self.find_player_by_identifier(str(user_id))
            if not player:
                return None
            
            achievements = self.get_player_achievements(player['id'])
            
            chase_stats = self.get_player_chase_stats(player['id'])
            
            profile_data = {
                'player': player,
                'achievements': achievements,
                'chase_stats': chase_stats
            }
            
            data_hash = SmartCacheManager.generate_data_hash(profile_data)
            
            self.profile_cache['data'][user_id] = profile_data
            self.profile_cache['last_updated'][user_id] = current_time
            self.profile_cache['data_version'][user_id] = data_hash
            
            logger.debug(f"Profile cached for user {user_id} with hash {data_hash[:8]}...")
            return profile_data
            
        except Exception as e:
            logger.error(f"Error getting profile data for user {user_id}: {e}")
            return None
    
    def update_roast_usage_async(self, roast_line: str):
        """Update roast usage asynchronously to avoid blocking"""
        
        async def update_usage_task():
            try:
                conn = self.get_db_connection()
                if not conn:
                    return
                
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE roast_rotation 
                    SET usage_count = usage_count + 1, 
                        last_used_date = CURRENT_DATE 
                    WHERE roast_line = %s
                """, (roast_line,))
                
                conn.commit()
                self.return_db_connection(conn)
                
                if roast_line in self.roast_cache['usage_counts']:
                    self.roast_cache['usage_counts'].increment(roast_line)
                
            except Exception as e:
                logger.error(f"Error updating roast usage: {e}")
        
        deferred_scheduler.schedule_task(update_usage_task())

    # =========== GUESS THE NUMBER GAME METHODS =========== 
    
    def create_guess_game(self, user_id: int, difficulty: str, player_name: str, chat_id: int) -> dict:
        """Create a new guess game session"""
        if difficulty not in self.guess_difficulties:
            return None
            
        config = self.guess_difficulties[difficulty]
        target = random.randint(config['range'][0], config['range'][1])
        
        game_state = {
            'user_id': user_id,
            'player_name': player_name,
            'chat_id': chat_id,
            'difficulty': difficulty,
            'target_number': target,
            'attempts_used': 0,
            'max_attempts': config['attempts'],
            'time_limit': config['time_limit'],
            'start_time': time.time(),
            'hint_used': False,
            'guesses': [],  # Track all guesses
            'game_active': True,
            'range_min': config['range'][0],
            'range_max': config['range'][1]
        }
        
        self.guess_games[user_id] = game_state
        return game_state
    
    def get_guess_game(self, user_id: int) -> dict:
        """Get active guess game for user"""
        return self.guess_games.get(user_id)
    
    def end_guess_game(self, user_id: int, outcome: str) -> bool:
        """End a guess game and record statistics"""
        if user_id not in self.guess_games:
            return False
            
        game = self.guess_games[user_id]
        game['game_active'] = False
        
        time_taken = int(time.time() - game['start_time'])
        final_score = self.calculate_guess_score(game, outcome, time_taken)
        
        is_daily_challenge = game.get('is_daily_challenge', False)
        challenge_date = game.get('challenge_date', None)
        
        success = self.record_guess_game(
            player_id=None,
            telegram_id=user_id,
            player_name=game['player_name'],
            chat_id=game['chat_id'],
            difficulty=game['difficulty'],
            target_number=game['target_number'],
            guesses_used=game['attempts_used'],
            max_guesses=game['max_attempts'],
            time_taken=time_taken,
            time_limit=game['time_limit'],
            final_score=final_score,
            hint_used=game['hint_used'],
            game_outcome=outcome,
            daily_challenge=is_daily_challenge,
            challenge_date=challenge_date
        )
        
        if outcome == 'won' and final_score > 0:
            self.update_player_scores(user_id, final_score)
            unlocked_level = self.unlock_next_level(user_id, game['difficulty'])
            if unlocked_level:
                game['new_level_unlocked'] = unlocked_level
        
        if success:
            try:
                is_daily = game.get('is_daily_challenge', False)
                game_type = 'daily_guess' if is_daily else 'guess_game'
                won = outcome == 'won'
                
                shard_reward = self.calculate_game_shard_reward(game_type, final_score, 1, won)
                
                if shard_reward > 0:
                    shard_success = self.award_shards(
                        user_id,
                        shard_reward,
                        game_type,
                        f'Score: {final_score}, Outcome: {outcome}, Difficulty: {game["difficulty"]}'
                    )
                    
                    if shard_success:
                        logger.info(f"Awarded {shard_reward} shards to player {user_id} for {game_type}")
                        game['shard_reward'] = shard_reward
                    else:
                        logger.warning(f"Failed to award shards to player {user_id}")
            
            except Exception as e:
                logger.error(f"Error awarding guess game shards: {e}")
            
            try:
                daily_success = self.update_daily_leaderboard(
                    user_id,
                    game['player_name'],
                    'guess',
                    final_score,
                    1,  # Guess games don't have levels
                    outcome == 'won'
                )
                
                if daily_success:
                    logger.info(f"Updated daily guess leaderboard for player {user_id}")
                else:
                    logger.warning(f"Failed to update daily guess leaderboard for player {user_id}")
            
            except Exception as e:
                logger.error(f"Error updating daily guess leaderboard: {e}")
        
        del self.guess_games[user_id]
        
        return success
    
    def calculate_guess_score(self, game: dict, outcome: str, time_taken: int) -> int:
        """Calculate final score for guess game"""
        if outcome != 'won':
            return 0
            
        config = self.guess_difficulties[game['difficulty']]
        base_score = int(100 * config['multiplier'])
        
        speed_bonus = max(0, 50 - (time_taken // 4))
        
        attempt_bonus = max(0, (game['max_attempts'] - game['attempts_used']) * 10)
        
        perfect_bonus = 25 if game['attempts_used'] == 1 else 0

        if game['hint_used']:
            final_score = (base_score + speed_bonus + attempt_bonus + perfect_bonus) // 2
        else:
            final_score = base_score + speed_bonus + attempt_bonus + perfect_bonus
        
        if game.get('is_daily_challenge', False):
            final_score = int(final_score * 1.5)
        
        return max(0, final_score)
    
    def get_unlocked_levels(self, telegram_id: int) -> list:
        """Get list of unlocked difficulty levels for a player"""
        conn = self.get_db_connection()
        if not conn:
            return ['beginner']  # Default to beginner only
            
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT unlocked_levels FROM players WHERE telegram_id = %s
            """, (telegram_id,))
            result = cursor.fetchone()
            
            if result and result[0]:
                return result[0]
            else:
                return ['beginner']
        except Exception as e:
            logger.error(f"Error getting unlocked levels for {telegram_id}: {e}")
            return ['beginner']
        finally:
            self.return_db_connection(conn)
    
    def unlock_next_level(self, telegram_id: int, current_difficulty: str) -> str:
        """Unlock next difficulty level when player wins. Returns next unlocked level or None"""
        difficulty_order = ['beginner', 'easy', 'medium', 'hard', 'expert']
        
        if current_difficulty not in difficulty_order:
            return None
            
        current_index = difficulty_order.index(current_difficulty)
        if current_index >= len(difficulty_order) - 1:
            return None  # Already at max level
            
        next_level = difficulty_order[current_index + 1]
        unlocked_levels = self.get_unlocked_levels(telegram_id)
        
        if next_level not in unlocked_levels:
            unlocked_levels.append(next_level)
            self.update_unlocked_levels(telegram_id, unlocked_levels)
            return next_level
        
        return None  # Level already unlocked
    
    def update_unlocked_levels(self, telegram_id: int, levels: list):
        """Update unlocked levels for a player"""
        conn = self.get_db_connection()
        if not conn:
            return
            
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE players SET unlocked_levels = %s WHERE telegram_id = %s
            """, (levels, telegram_id))
            conn.commit()
        except Exception as e:
            logger.error(f"Error updating unlocked levels for {telegram_id}: {e}")
        finally:
            self.return_db_connection(conn)
    
    def update_player_scores(self, telegram_id: int, new_score: int):
        """Update player's highest and total scores"""
        conn = self.get_db_connection()
        if not conn:
            return
            
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT highest_score, total_score FROM players WHERE telegram_id = %s
            """, (telegram_id,))
            result = cursor.fetchone()
            
            current_highest = result[0] if result and result[0] else 0
            current_total = result[1] if result and result[1] else 0
            
            new_highest = max(current_highest, new_score)
            new_total = current_total + new_score
            
            cursor.execute("""
                UPDATE players SET 
                    highest_score = %s, 
                    total_score = %s 
                WHERE telegram_id = %s
            """, (new_highest, new_total, telegram_id))
            conn.commit()
        except Exception as e:
            logger.error(f"Error updating scores for {telegram_id}: {e}")
        finally:
            self.return_db_connection(conn)
    
    def record_guess_game(self, player_id: int, telegram_id: int, player_name: str, 
                         chat_id: int, difficulty: str, target_number: int, 
                         guesses_used: int, max_guesses: int, time_taken: int, 
                         time_limit: int, final_score: int, hint_used: bool, 
                         game_outcome: str, daily_challenge: bool = False, 
                         challenge_date: str = None) -> bool:
        """Record a completed guess game"""
        conn = self.get_db_connection()
        if not conn:
            return False
            
        try:
            cursor = conn.cursor()
            
            if not player_id:
                cursor.execute("SELECT id FROM players WHERE telegram_id = %s", (telegram_id,))
                player_result = cursor.fetchone()
                player_id = player_result[0] if player_result else None
            
            cursor.execute("""
                INSERT INTO guess_games (
                    player_id, telegram_id, player_name, chat_id, difficulty,
                    target_number, guesses_used, max_guesses, time_taken, time_limit,
                    final_score, hint_used, game_outcome, daily_challenge,
                    challenge_date, completed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """, (player_id, telegram_id, player_name, chat_id, difficulty,
                  target_number, guesses_used, max_guesses, time_taken, time_limit,
                  final_score, hint_used, game_outcome, daily_challenge, challenge_date))
            
            conn.commit()
            logger.info(f"Successfully recorded guess game for player {telegram_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording guess game: {e}")
            conn.rollback()
            return False
        finally:
            self.return_db_connection(conn)
    
    def get_guess_game_stats(self, telegram_id: int = None) -> dict:
        """Get guess game statistics (global or for specific player)"""
        conn = self.get_db_connection()
        if not conn:
            return {}
            
        try:
            cursor = conn.cursor()
            
            if telegram_id:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as games_played,
                        COUNT(CASE WHEN game_outcome = 'won' THEN 1 END) as games_won,
                        MAX(final_score) as best_score,
                        AVG(time_taken) as avg_time,
                        AVG(guesses_used) as avg_attempts,
                        COUNT(CASE WHEN guesses_used = 1 AND game_outcome = 'won' THEN 1 END) as perfect_guesses
                    FROM guess_games 
                    WHERE telegram_id = %s
                """, (telegram_id,))
                
                result = cursor.fetchone()
                if result:
                    games_played, games_won, best_score, avg_time, avg_attempts, perfect_guesses = result
                    win_rate = (games_won / games_played * 100) if games_played > 0 else 0
                    
                    return {
                        'games_played': games_played or 0,
                        'games_won': games_won or 0,
                        'win_rate': round(win_rate, 1),
                        'best_score': best_score or 0,
                        'avg_time': round(avg_time) if avg_time else 0,
                        'avg_attempts': round(avg_attempts, 1) if avg_attempts else 0,
                        'perfect_guesses': perfect_guesses or 0
                    }
            else:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_games,
                        COUNT(DISTINCT telegram_id) as total_players,
                        MAX(final_score) as highest_score,
                        AVG(final_score) as avg_score
                    FROM guess_games
                """)
                
                result = cursor.fetchone()
                if result:
                    return {
                        'total_games': result[0] or 0,
                        'total_players': result[1] or 0, 
                        'highest_score': result[2] or 0,
                        'avg_score': round(result[3]) if result[3] else 0
                    }
            
        except Exception as e:
            logger.error(f"Error getting guess game stats: {e}")
        finally:
            self.return_db_connection(conn)
            
        return {}
    
    def get_guess_leaderboard(self, category: str = 'score', limit: int = 10) -> List[dict]:
        """Get guess game leaderboard"""
        conn = self.get_db_connection()
        if not conn:
            return []
            
        try:
            cursor = conn.cursor()
            
            if category == 'score':
                cursor.execute("""
                    SELECT telegram_id, player_name, MAX(final_score) as best_score,
                           COUNT(*) as games_played
                    FROM guess_games 
                    WHERE game_outcome = 'won'
                    GROUP BY telegram_id, player_name
                    ORDER BY best_score DESC, games_played DESC
                    LIMIT %s
                """, (limit,))
            elif category == 'speed':
                cursor.execute("""
                    SELECT telegram_id, player_name, MIN(time_taken) as fastest_time,
                           COUNT(*) as games_played
                    FROM guess_games 
                    WHERE game_outcome = 'won' AND time_taken > 0
                    GROUP BY telegram_id, player_name
                    ORDER BY fastest_time ASC, games_played DESC
                    LIMIT %s
                """, (limit,))
            elif category == 'games':
                cursor.execute("""
                    SELECT telegram_id, player_name, COUNT(*) as games_played,
                           COUNT(CASE WHEN game_outcome = 'won' THEN 1 END) as games_won
                    FROM guess_games 
                    GROUP BY telegram_id, player_name
                    ORDER BY games_played DESC, games_won DESC
                    LIMIT %s
                """, (limit,))
            else:
                return []
                
            results = cursor.fetchall()
            return [dict(zip([desc[0] for desc in cursor.description], row)) for row in results]
            
        except Exception as e:
            logger.error(f"Error getting guess leaderboard: {e}")
            return []
        finally:
            self.return_db_connection(conn)
    
    def generate_hint(self, game: dict) -> str:
        """Generate a smart hint for the guess game"""
        target = game['target_number']
        range_min = game['range_min'] 
        range_max = game['range_max']
        
        for guess in game['guesses']:
            if guess < target:
                range_min = max(range_min, guess + 1)
            else:
                range_max = min(range_max, guess - 1)
        
        hint_type = random.choice(['divisible', 'range', 'odd_even', 'digit'])
        
        if hint_type == 'divisible':
            for div in [2, 3, 5, 7]:
                if target % div == 0:
                    return f"🍀 The number is divisible by {div}"
            return f"🍀 The number is not divisible by 2, 3, or 5"
            
        elif hint_type == 'range':
            span = range_max - range_min
            if span > 20:
                center = target
                new_min = max(range_min, center - 10)
                new_max = min(range_max, center + 10) 
                return f"🍀 The number is between {new_min}-{new_max}"
            else:
                return f"🍀 The number is between {range_min}-{range_max}"
                
        elif hint_type == 'odd_even':
            return f"🍀 The number is {'odd' if target % 2 == 1 else 'even'}"
            
        else:  # digit
            last_digit = target % 10
            return f"🍀 The number ends with {last_digit}"
    
    def cleanup_expired_guess_games(self) -> int:
        """Clean up expired guess games"""
        current_time = time.time()
        expired_users = []
        
        for user_id, game in self.guess_games.items():
            if not game.get('game_active', True):
                expired_users.append(user_id)
                continue
                
            elapsed = current_time - game['start_time']
            if elapsed > game['time_limit']:
                self.end_guess_game(user_id, 'timeout')
                expired_users.append(user_id)
        
        for user_id in expired_users:
            if user_id in self.guess_games:
                del self.guess_games[user_id]
                
        return len(expired_users)

    def reset_player_data(self, player_id: int, performed_by: int) -> bool:
        """Reset all player data (Super Admin only)"""
        conn = self.get_db_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            cursor.execute("SELECT display_name, title FROM players WHERE id = %s", (player_id,))
            player_data = cursor.fetchone()
            if not player_data:
                return False
            
            player_name, current_title = player_data
            
            cursor.execute("SELECT achievement_name, count FROM achievements WHERE player_id = %s", (player_id,))
            achievements = cursor.fetchall()
            
            for achievement, count in achievements:
                self.backup_achievement_action(player_id, player_name, achievement, count, 'RESET', performed_by)
            
            cursor.execute("DELETE FROM achievements WHERE player_id = %s", (player_id,))
            
            cursor.execute("UPDATE players SET title = NULL WHERE id = %s", (player_id,))
            
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error resetting player data: {e}")
            conn.rollback()
            return False
        finally:
            self.return_db_connection(conn)
    
    def format_achievements_message(self, player: Dict, achievements: List[Tuple[str, int]]) -> str:
        """Format achievements display message - matches user's desired format"""
        name = player["display_name"] or player["username"] or f"User {player['telegram_id']}"
        
        message = f"🏆 <b>ARENA OF CHAMPIONS ACHIEVEMENTS</b>\n<b>━━━━━━━━━━━━━━━</b>\n👤 <b>Player:</b> <b>{name}</b>\n"
        
        if player.get("title"):
            message += f"👑 <b>Title:</b> <b>{player['title']}</b>\n"
        
        message += "━━━━━━━━━━━━━━━\n\n"
        
        if not achievements:
            message += "🚫 <b>No achievements yet!</b>\n💪 <b>Keep playing to earn your first award!</b> 🎯"
            return message
        
        message += "🏆 <b>ACHIEVEMENT LIST:</b>\n\n"
        
        total_awards = 0
        for index, (achievement, count) in enumerate(achievements, 1):
            proper_name = self.get_proper_achievement_name(achievement)
            emoji = self.get_achievement_emoji(achievement)
            emoji_display = f"{emoji} " if emoji else "🎖️ "
            count_display = f" <b>(×{count})</b>" if count > 1 else ""
            message += f"<b>{index}.</b> {emoji_display}<b>{proper_name}</b>{count_display}\n"
            total_awards += count
        
        message += f"\n<b>━━━━━━━━━━━━━━━</b>\n📊 <b>Total Awards:</b> <b>{total_awards}</b> 🎖"
        
        return message

    # ====================================
    # ====================================
    
    def get_all_active_games(self, telegram_id: int) -> dict:
        """Get all active games for a user across all game modes"""
        active_games = {
            'guess': None,
            'nightmare': None,
            'chase': []  # Can have multiple chase games
        }
        
        guess_game = self.get_guess_game(telegram_id)
        if guess_game and guess_game.get('game_active'):
            active_games['guess'] = {
                'type': 'Guess Game',
                'difficulty': guess_game.get('difficulty', '').title(),
                'attempts_left': guess_game.get('max_attempts', 0) - guess_game.get('attempts_used', 0),
                'time_left': max(0, guess_game.get('time_limit', 0) - int(time.time() - guess_game.get('start_time', 0))),
                'range': f"{guess_game.get('range_min', 1)}-{guess_game.get('range_max', 100)}"
            }
        
        nightmare_game = self.get_nightmare_game(telegram_id)
        if nightmare_game:
            active_games['nightmare'] = {
                'type': 'Nightmare Mode',
                'attempts_left': nightmare_game.get('max_attempts', 0) - nightmare_game.get('attempts_used', 0),
                'hint': nightmare_game.get('decoded_hint', 'Mystery number awaits...')
            }
        
        for key, game_state in ACTIVE_CHASE_GAMES.items():
            if game_state.get('player_id') == telegram_id:
                chase_info = {
                    'type': 'Chase Game',
                    'level': game_state.get('level', 1),
                    'score': f"{game_state.get('runs', 0)}/{game_state.get('target', 0)}",
                    'balls_left': game_state.get('balls_remaining', 0),
                    'wickets_left': game_state.get('wickets_remaining', 0)
                }
                active_games['chase'].append(chase_info)
        
        return active_games
    
    def create_game_switch_message(self, telegram_id: int, new_game_type: str) -> tuple[str, bool]:
        """Create a helpful message when switching between games. Returns (message, has_conflicts)"""
        active_games = self.get_all_active_games(telegram_id)
        
        total_active = 0
        if active_games['guess']:
            total_active += 1
        if active_games['nightmare']:
            total_active += 1
        total_active += len(active_games['chase'])
        
        if total_active == 0:
            return "", False  # No conflicts
        
        message = f"🎮 <b>Active Games Status</b>\n\n"
        message += f"You currently have <b>{total_active}</b> active game{'s' if total_active > 1 else ''}:\n\n"
        
        if active_games['guess']:
            g = active_games['guess']
            message += f"🎯 <b>{g['type']}</b> ({g['difficulty']})\n"
            message += f"   • {g['attempts_left']} attempts left\n"
            message += f"   • {g['time_left']}s remaining\n"
            message += f"   • Range: {g['range']}\n\n"
        
        if active_games['nightmare']:
            n = active_games['nightmare']
            message += f"💀 <b>{n['type']}</b>\n"
            message += f"   • {n['attempts_left']} attempts left\n"
            message += f"   • Hint: {InputValidator.truncate_with_ellipsis(n['hint'], 30)}\n\n"
        
        for i, chase in enumerate(active_games['chase']):
            message += f"🏏 <b>{chase['type']}</b> #{i+1}\n"
            message += f"   • Level {chase['level']}\n"
            message += f"   • Score: {chase['score']}\n"
            message += f"   • {chase['balls_left']} balls, {chase['wickets_left']} wickets left\n\n"
        
        message += f"💡 <b>Options:</b>\n"
        message += f"• Continue any active game\n"
        message += f"• Use /quit to end current games\n"
        message += f"• Or start {new_game_type} anyway (games auto-cleanup after timeout)\n\n"
        message += f"🎯 <b>All games earn shards - no progress is lost!</b>"
        
        return message, True
    
    # ====================================
    # ====================================
    
    def get_player_shards(self, telegram_id: int) -> tuple[int, int]:
        """Get player's current shards and total earned. Returns (current, total_earned)"""
        conn = self.get_db_connection()
        if not conn:
            return 0, 0
        
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT shards, total_shards_earned FROM players WHERE telegram_id = %s", 
                (telegram_id,)
            )
            result = cursor.fetchone()
            if result:
                return result[0] or 0, result[1] or 0
            return 0, 0
        except Exception as e:
            logger.error(f"Error getting player shards for {telegram_id}: {e}")
            return 0, 0
        finally:
            self.return_db_connection(conn)
    
    def award_shards(self, telegram_id: int, amount: int, source: str, source_details: str = None, performed_by: int = None, notes: str = None) -> bool:
        """Award shards to a player with transaction logging"""
        if amount <= 0:
            return False
            
        conn = self.get_db_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            cursor.execute("SELECT shards, total_shards_earned, display_name, id FROM players WHERE telegram_id = %s", (telegram_id,))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"Player {telegram_id} not found during shard award - auto-creating")
                cursor.execute("""
                    INSERT INTO players (telegram_id, username, display_name)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (telegram_id) DO NOTHING
                """, (telegram_id, None, source_details or f"User {telegram_id}"))
                
                cursor.execute("SELECT shards, total_shards_earned, display_name, id FROM players WHERE telegram_id = %s", (telegram_id,))
                result = cursor.fetchone()
                
                if not result:
                    logger.error(f"Failed to create player {telegram_id} for shard award")
                    return False
                
            current_shards, total_earned, player_name, player_id = result
            new_balance = (current_shards or 0) + amount
            new_total = (total_earned or 0) + amount
            
            performed_by_id = None
            if performed_by:
                cursor.execute("SELECT id FROM players WHERE telegram_id = %s", (performed_by,))
                admin_result = cursor.fetchone()
                performed_by_id = admin_result[0] if admin_result else None
            
            cursor.execute("""
                UPDATE players 
                SET shards = %s, total_shards_earned = %s 
                WHERE telegram_id = %s
            """, (new_balance, new_total, telegram_id))
            
            cursor.execute("""
                INSERT INTO shard_transactions 
                (player_id, player_telegram_id, player_name, transaction_type, amount, source, source_details, 
                 balance_before, balance_after, performed_by, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (player_id, telegram_id, player_name or f"User {telegram_id}", 'EARN', amount, source, source_details,
                  current_shards or 0, new_balance, performed_by_id, notes))
            
            conn.commit()
            
            SmartCacheManager.invalidate_related_caches(self, "profile_update", telegram_id)
            SmartCacheManager.invalidate_related_caches(self, "leaderboard_change")
            
            logger.info(f"Awarded {amount} shards to {telegram_id} from {source}")
            return True
            
        except Exception as e:
            logger.error(f"Error awarding shards to {telegram_id}: {e}")
            conn.rollback()
            return False
        finally:
            self.return_db_connection(conn)
    
    def remove_shards(self, telegram_id: int, amount: int, source: str, source_details: str = None, performed_by: int = None, notes: str = None) -> bool:
        """Remove shards from a player (admin only) with transaction logging"""
        if amount <= 0:
            return False
            
        conn = self.get_db_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            cursor.execute("SELECT shards, display_name, id FROM players WHERE telegram_id = %s", (telegram_id,))
            result = cursor.fetchone()
            
            if not result:
                logger.error(f"Player {telegram_id} not found for shard removal")
                return False
                
            current_shards, player_name, player_id = result
            current_shards = current_shards or 0
            
            if current_shards < amount:
                logger.error(f"Insufficient shards: {telegram_id} has {current_shards}, tried to remove {amount}")
                return False
                
            new_balance = current_shards - amount
            
            performed_by_id = None
            if performed_by:
                cursor.execute("SELECT id FROM players WHERE telegram_id = %s", (performed_by,))
                admin_result = cursor.fetchone()
                performed_by_id = admin_result[0] if admin_result else None
            
            cursor.execute("""
                UPDATE players 
                SET shards = %s 
                WHERE telegram_id = %s
            """, (new_balance, telegram_id))
            
            cursor.execute("""
                INSERT INTO shard_transactions 
                (player_id, player_telegram_id, player_name, transaction_type, amount, source, source_details, 
                 balance_before, balance_after, performed_by, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (player_id, telegram_id, player_name or f"User {telegram_id}", 'ADMIN_REMOVE', -amount, source, source_details,
                  current_shards, new_balance, performed_by_id, notes))
            
            conn.commit()
            
            SmartCacheManager.invalidate_related_caches(self, "profile_update", telegram_id)
            SmartCacheManager.invalidate_related_caches(self, "leaderboard_change")
            
            logger.info(f"Removed {amount} shards from {telegram_id} by admin {performed_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing shards from {telegram_id}: {e}")
            conn.rollback()
            return False
        finally:
            self.return_db_connection(conn)
    
    def give_admin_shards(self, telegram_id: int, amount: int, performed_by: int, notes: str = None) -> bool:
        """Admin command to give shards to a player"""
        if amount <= 0:
            return False
            
        MAX_SHARD_AMOUNT = 2147483647  # PostgreSQL INTEGER max value
        if amount > MAX_SHARD_AMOUNT:
            logger.error(f"Amount {amount} too large, max allowed: {MAX_SHARD_AMOUNT}")
            return False
            
        conn = self.get_db_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            cursor.execute("SELECT shards, total_shards_earned, display_name, id FROM players WHERE telegram_id = %s", (telegram_id,))
            result = cursor.fetchone()
            
            if not result:
                logger.error(f"Player {telegram_id} not found for shard award")
                return False
                
            current_shards, total_earned, player_name, player_id = result
            new_balance = current_shards + amount
            new_total = (total_earned or 0) + amount
            
            if new_balance > MAX_SHARD_AMOUNT or new_total > MAX_SHARD_AMOUNT:
                logger.error(f"Balance would overflow: new_balance={new_balance}, new_total={new_total}")
                return False
            
            cursor.execute("SELECT id FROM players WHERE telegram_id = %s", (performed_by,))
            admin_result = cursor.fetchone()
            admin_player_id = admin_result[0] if admin_result else None
            
            cursor.execute("""
                UPDATE players 
                SET shards = %s, total_shards_earned = %s 
                WHERE telegram_id = %s
            """, (new_balance, new_total, telegram_id))
            
            cursor.execute("""
                INSERT INTO shard_transactions 
                (player_id, player_telegram_id, player_name, transaction_type, amount, source, source_details, 
                 balance_before, balance_after, performed_by, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (player_id, telegram_id, player_name or f"User {telegram_id}", 'ADMIN_GIVE', amount, 'admin', f'Admin award: {amount} shards',
                  current_shards, new_balance, admin_player_id, notes))
            
            conn.commit()
            logger.info(f"Admin {performed_by} gave {amount} shards to {telegram_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error admin giving shards to {telegram_id}: {e}")
            conn.rollback()
            return False
        finally:
            self.return_db_connection(conn)
    
    def claim_daily_shards(self, telegram_id: int) -> tuple[bool, int, int, str]:
        """
        Claim daily shards bonus. Returns (success, amount_awarded, streak_days, message)
        """
        conn = self.get_db_connection()
        if not conn:
            return False, 0, 0, "Database connection failed"
        
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT last_daily_claim FROM players WHERE telegram_id = %s
            """, (telegram_id,))
            
            result = cursor.fetchone()
            if not result:
                return False, 0, 0, "Player not found"
            
            last_claim = result[0]
            today = date.today()
            
            if last_claim == today:
                return False, 0, 0, "Already claimed today! Come back tomorrow."
            
            yesterday = today.replace(day=today.day - 1) if today.day > 1 else today.replace(month=today.month - 1 if today.month > 1 else 12, day=31 if today.month > 1 else 30)
            
            streak_days = 1
            if last_claim == yesterday:
                cursor.execute("""
                    SELECT streak_days FROM daily_shard_bonuses 
                    WHERE player_id = (SELECT id FROM players WHERE telegram_id = %s)
                    ORDER BY claim_date DESC LIMIT 1
                """, (telegram_id,))
                
                streak_result = cursor.fetchone()
                if streak_result:
                    streak_days = streak_result[0] + 1
            
            base_amount = 50  # Base daily bonus
            streak_bonus = min((streak_days - 1) * 10, 200)  # Up to 20 days = +200 bonus
            total_amount = base_amount + streak_bonus
            
            success = self.award_shards(
                telegram_id, 
                total_amount, 
                'daily_bonus', 
                f'Day {streak_days} streak bonus',
                None,
                f'Base: {base_amount}, Streak bonus: {streak_bonus}'
            )
            
            if success:
                cursor.execute("""
                    UPDATE players SET last_daily_claim = %s WHERE telegram_id = %s
                """, (today, telegram_id))
                
                cursor.execute("""
                    INSERT INTO daily_shard_bonuses (
                        player_id, claim_date, base_amount, streak_bonus, total_amount, streak_days
                    ) VALUES (
                        (SELECT id FROM players WHERE telegram_id = %s),
                        %s, %s, %s, %s, %s
                    )
                """, (telegram_id, today, base_amount, streak_bonus, total_amount, streak_days))
                
                conn.commit()
                
                message = f"🎉 Daily bonus claimed!\n💠 +{total_amount} shards"
                if streak_days > 1:
                    message += f"\n🔥 {streak_days} day streak! (+{streak_bonus} bonus)"
                
                return True, total_amount, streak_days, message
            else:
                return False, 0, 0, "Failed to award shards"
                
        except Exception as e:
            logger.error(f"Error claiming daily shards for {telegram_id}: {e}")
            conn.rollback()
            return False, 0, 0, f"Error: {e}"
        finally:
            self.return_db_connection(conn)
    
    def get_shard_transactions(self, telegram_id: int, limit: int = 10) -> List[Dict]:
        """Get recent shard transactions for a player"""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT transaction_type, amount, source, source_details, 
                       balance_before, balance_after, performed_at, notes
                FROM shard_transactions 
                WHERE player_telegram_id = %s 
                ORDER BY performed_at DESC 
                LIMIT %s
            """, (telegram_id, limit))
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Error getting shard transactions for {telegram_id}: {e}")
            return []
        finally:
            self.return_db_connection(conn)
    
    def get_shard_economy_stats(self) -> dict:
        """Get comprehensive shard economy statistics"""
        conn = self.get_db_connection()
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COALESCE(SUM(shards), 0) FROM players")
            total_circulation = cursor.fetchone()[0]
            
            cursor.execute("SELECT COALESCE(SUM(total_shards_earned), 0) FROM players")
            total_earned = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM shard_transactions")
            total_transactions = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT source, COUNT(*), COALESCE(SUM(amount), 0)
                FROM shard_transactions 
                WHERE transaction_type = 'earned'
                GROUP BY source
                ORDER BY SUM(amount) DESC
            """)
            earn_sources = cursor.fetchall()
            
            cursor.execute("""
                SELECT display_name, shards, total_shards_earned
                FROM players 
                WHERE shards > 0
                ORDER BY shards DESC 
                LIMIT 10
            """)
            top_holders = cursor.fetchall()
            
            cursor.execute("""
                SELECT DATE(performed_at) as date, COUNT(*), SUM(amount)
                FROM shard_transactions 
                WHERE performed_at >= CURRENT_DATE - INTERVAL '7 days'
                AND transaction_type = 'earned'
                GROUP BY DATE(performed_at)
                ORDER BY date DESC
            """)
            daily_activity = cursor.fetchall()
            
            return {
                'total_circulation': total_circulation,
                'total_earned': total_earned,
                'total_transactions': total_transactions,
                'earn_sources': earn_sources,
                'top_holders': top_holders,
                'daily_activity': daily_activity
            }
            
        except Exception as e:
            logger.error(f"Error getting shard economy stats: {e}")
            return {}
        finally:
            self.return_db_connection(conn)
    
    def calculate_game_shard_reward(self, game_type: str, score: int, level: int = 1, won: bool = True) -> int:
        """Calculate shard reward based on game performance"""
        base_rewards = {
            'chase_game': 15,    # Reduced from 30 - Base reward for chase games
            'guess_game': 12,    # Reduced from 25 - Base reward for guess games
            'daily_guess': 20,   # Reduced from 40 - Higher reward for daily challenges
        }
        
        base_reward = base_rewards.get(game_type, 10)
        
        if not won:
            return max(2, base_reward // 5)  # Reduced consolation
        
        score_bonus = min(score // 150, 25)  # Reduced from 100->50 to 150->25
        level_bonus = min((level - 1) * 3, 15)  # Reduced from 5->30 to 3->15
        
        total_reward = base_reward + score_bonus + level_bonus
        
        max_reward = base_reward * 2  # Reduced from 3x to 2x
        return min(total_reward, max_reward)

    # ====================================
    # ====================================
    
    def update_daily_leaderboard(self, telegram_id: int, player_name: str, game_type: str, score: int, level: int = 1, won: bool = True) -> bool:
        """Update player's daily leaderboard entry"""
        conn = self.get_db_connection()
        if not conn:
            logger.error("Failed to get database connection for daily leaderboard update")
            return False
        
        try:
            cursor = conn.cursor()
            
            clean_player_name = player_name.encode('utf-8', 'ignore').decode('utf-8', 'ignore')[:100]
            
            cursor.execute("""
                SELECT update_daily_leaderboard_entry(%s, %s, %s, %s, %s, %s)
            """, (telegram_id, clean_player_name, game_type, score, level, won))
            
            result = cursor.fetchone()[0]
            conn.commit()
            
            if result:
                logger.info(f"Updated daily leaderboard for {telegram_id} ({clean_player_name}): {game_type} score {score}, level {level}")
            else:
                logger.warning(f"Daily leaderboard update returned False for {telegram_id}: {game_type}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error updating daily leaderboard for {telegram_id} ({player_name}): {e}")
            conn.rollback()
            return False
        finally:
            self.return_db_connection(conn)
    
    def get_daily_leaderboard(self, game_type: str, limit: int = 10) -> List[Dict]:
        """Get current daily leaderboard for specified game type"""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            if game_type == 'chase':
                cursor.execute("""
                    SELECT player_telegram_id, player_name, games_played,
                           best_score, level_completed, total_score, rank
                    FROM daily_chase_leaderboard
                    ORDER BY rank
                    LIMIT %s
                """, (limit,))
            elif game_type == 'guess':
                cursor.execute("""
                    SELECT player_telegram_id, player_name, games_played,
                           best_score, total_score, games_won, win_percentage, rank
                    FROM daily_guess_leaderboard  
                    ORDER BY rank
                    LIMIT %s
                """, (limit,))
            else:
                return []
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Error getting daily leaderboard for {game_type}: {e}")
            return []
        finally:
            self.return_db_connection(conn)
    
    def get_player_daily_rank(self, telegram_id: int, game_type: str) -> tuple[int, dict]:
        """Get player's current daily rank and stats. Returns (rank, stats)"""
        conn = self.get_db_connection()
        if not conn:
            return 0, {}
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            if game_type == 'chase':
                cursor.execute("""
                    SELECT dle.*, 
                           RANK() OVER (ORDER BY chase_total_score DESC, chase_best_level DESC, chase_best_score DESC) as rank
                    FROM daily_leaderboard_entries dle
                    WHERE leaderboard_date = CURRENT_DATE AND game_type = %s AND player_telegram_id = %s
                """, (game_type, telegram_id))
            elif game_type == 'guess':
                cursor.execute("""
                    SELECT dle.*,
                           RANK() OVER (ORDER BY guess_total_score DESC, guess_games_won DESC, guess_best_score DESC) as rank
                    FROM daily_leaderboard_entries dle
                    WHERE leaderboard_date = CURRENT_DATE AND game_type = %s AND player_telegram_id = %s
                """, (game_type, telegram_id))
            else:
                return 0, {}
            
            result = cursor.fetchone()
            if result:
                return result['rank'], dict(result)
            return 0, {}
            
        except Exception as e:
            logger.error(f"Error getting daily rank for {telegram_id}: {e}")
            return 0, {}
        finally:
            self.return_db_connection(conn)
    
    def distribute_daily_leaderboard_rewards(self, game_type: str, reward_date: str = None) -> int:
        """Distribute rewards for daily leaderboard. Returns number of players rewarded."""
        conn = self.get_db_connection()
        if not conn:
            return 0
        
        try:
            cursor = conn.cursor()
            
            reward_count = 0
            target_date = reward_date if reward_date else datetime.now().date()
            
            rewards = [100, 100, 100, 80, 80, 80, 60, 60, 40, 20]
            
            leaderboard = self.get_daily_leaderboard(game_type, 10)
            
            if not leaderboard:
                logger.info(f"No players found for {game_type} leaderboard on {target_date}")
                return 0
                
            for position, player in enumerate(leaderboard[:10], 1):
                if position <= len(rewards):
                    reward_amount = rewards[position - 1]  # Array is 0-indexed
                    telegram_id = player.get('player_telegram_id')
                    player_name = player.get('player_name', 'Unknown Player')
                    score = player.get('total_score', 0)
                    
                    if not telegram_id:
                        logger.error(f"No telegram_id found for {game_type} position #{position}: {player}")
                        continue
                    
                    success = self.award_shards(
                        telegram_id,
                        reward_amount,
                        f'daily_{game_type}_leaderboard',
                        f'Position #{position} - Score: {score}',
                        None,
                        f'Daily {game_type.title()} Leaderboard Reward'
                    )
                    
                    if success:
                        reward_count += 1
                        logger.info(f"Awarded {reward_amount} shards to {player_name} ({telegram_id}) for daily {game_type} position #{position}")
            
            if reward_count > 0:
                logger.info(f"Distributed daily {game_type} leaderboard rewards to {reward_count} players")
                
                cursor.execute("""
                    INSERT INTO daily_leaderboard_status (status_date, chase_rewards_distributed, guess_rewards_distributed, chase_rewards_at, guess_rewards_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (status_date) DO UPDATE SET
                        chase_rewards_distributed = CASE WHEN %s = 'chase' THEN TRUE ELSE daily_leaderboard_status.chase_rewards_distributed END,
                        guess_rewards_distributed = CASE WHEN %s = 'guess' THEN TRUE ELSE daily_leaderboard_status.guess_rewards_distributed END,
                        chase_rewards_at = CASE WHEN %s = 'chase' THEN NOW() ELSE daily_leaderboard_status.chase_rewards_at END,
                        guess_rewards_at = CASE WHEN %s = 'guess' THEN NOW() ELSE daily_leaderboard_status.guess_rewards_at END
                """, (
                    target_date,
                    game_type == 'chase', game_type == 'guess',
                    datetime.now() if game_type == 'chase' else None,
                    datetime.now() if game_type == 'guess' else None,
                    game_type, game_type, game_type, game_type
                ))
                
                conn.commit()
            
            return reward_count
            
        except Exception as e:
            logger.error(f"Error distributing daily rewards for {game_type}: {e}")
            conn.rollback()
            return 0
        finally:
            self.return_db_connection(conn)
    
    def check_daily_rewards_status(self) -> dict:
        """Check if daily rewards have been distributed today"""
        conn = self.get_db_connection()
        if not conn:
            return {'chase': False, 'guess': False}
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT chase_rewards_distributed, guess_rewards_distributed,
                       chase_rewards_at, guess_rewards_at
                FROM daily_leaderboard_status 
                WHERE status_date = CURRENT_DATE
            """)
            
            result = cursor.fetchone()
            if result:
                return {
                    'chase': result[0],
                    'guess': result[1], 
                    'chase_at': result[2],
                    'guess_at': result[3]
                }
            return {'chase': False, 'guess': False}
            
        except Exception as e:
            logger.error(f"Error checking daily rewards status: {e}")
            return {'chase': False, 'guess': False}
        finally:
            self.return_db_connection(conn)
    
    def reset_daily_leaderboards(self, target_date: str = None) -> bool:
        """Reset daily leaderboards for specified date (or current date)"""
        conn = self.get_db_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            if target_date:
                cursor.execute("DELETE FROM daily_leaderboard_entries WHERE leaderboard_date = %s", (target_date,))
                cursor.execute("DELETE FROM daily_leaderboard_status WHERE status_date = %s", (target_date,))
            else:
                cursor.execute("DELETE FROM daily_leaderboard_entries WHERE leaderboard_date = CURRENT_DATE")
                cursor.execute("DELETE FROM daily_leaderboard_status WHERE status_date = CURRENT_DATE")
            
            conn.commit()
            logger.info(f"Reset daily leaderboards for {target_date or 'today'}")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting daily leaderboards: {e}")
            conn.rollback()
            return False
        finally:
            self.return_db_connection(conn)

    # ====================================
    # ====================================
    
    async def send_admin_log(self, log_type: str, message: str, user_id: int = None, username: str = None) -> bool:
        """Send log messages to admin logs group chat"""
        if not self.logs_enabled:
            return True
            
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            emoji_map = {
                'admin_action': '🔧',
                'nightmare_win': '👑',
                'achievement': '🏆', 
                'system_event': '⚙️',
                'error': '🚨',
                'security': '🔒',
                'database': '💾',
                'leaderboard': '📊'
            }
            
            emoji = emoji_map.get(log_type, '📝')
            
            user_info = ""
            if user_id:
                if username:
                    user_info = f"\n👤 <b>User:</b> @{username} (ID: {user_id})"
                else:
                    user_info = f"\n👤 <b>User ID:</b> {user_id}"
            
            formatted_message = f"{emoji} <b>[{log_type.upper().replace('_', ' ')}]</b>\n" \
                              f"🕒 <b>Time:</b> {timestamp}" \
                              f"{user_info}\n" \
                              f"📋 <b>Details:</b> {message}"
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                url = f"https://api.telegram.org/bot{self.logs_bot_token}/sendMessage"
                data = {
                    'chat_id': self.logs_chat_id,
                    'text': formatted_message,
                    'parse_mode': 'HTML'
                }
                async with session.post(url, data=data) as response:
                    if response.status == 200:
                        return True
                    else:
                        logger.error(f"Failed to send admin log: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending admin log: {e}")
            return False

    # ====================================
    # ====================================
    
    def generate_nightmare_hint(self, number: int) -> dict:
        """Generate strategic mathematical hints - MORE CRYPTIC for 20K range!"""
        import random
        
        number_str = str(number)
        digits = [int(d) for d in number_str]
        
        hint_variants = []
        
        hint_variants.append(f"🔢 This number has {len(number_str)} digits")
        hint_variants.append(f"➕ The sum of all digits is {sum(digits)}")
        
        if number % 2 == 0:
            hint_variants.append("⚡ This number is EVEN")
        else:
            hint_variants.append("⚡ This number is ODD")
        
        if number > 8000:
            hint_variants.append("🎯 Very close to the maximum (>8,000)")
        elif number < 1000:
            hint_variants.append("🔢 A relatively small number (<1,000)")
        
        if len(digits) > 1:
            hint_variants.append(f"🎯 First digit: {digits[0]}, Last digit: {digits[-1]}")
            hint_variants.append(f"🔍 Contains the digit {random.choice(digits)}")
            
            if digits[0] == digits[-1]:
                hint_variants.append("🔄 First and last digits are the same")
            
            if len(set(digits)) == 1:
                hint_variants.append("🎭 All digits are identical")
            elif len(set(digits)) == len(digits):
                hint_variants.append("🌟 All digits are different")
            
            if sum(digits) > 20:
                hint_variants.append("🔥 Sum of digits is greater than 20")
            elif sum(digits) < 8:
                hint_variants.append("❄️ Sum of digits is less than 8")
            
            if any(d > 7 for d in digits):
                hint_variants.append("🎯 Contains digit 8 or 9")
            if all(d < 5 for d in digits):
                hint_variants.append("🌊 All digits are 0, 1, 2, 3, or 4")
        
        divisibility_hints = []
        if number % 3 == 0:
            divisibility_hints.append("🔺 Divisible by 3")
        if number % 4 == 0:
            divisibility_hints.append("🎳 Divisible by 4")
        if number % 5 == 0:
            divisibility_hints.append("🔮 Divisible by 5")
        if number % 6 == 0:
            divisibility_hints.append("⚡ Divisible by 6")
        if number % 8 == 0:
            divisibility_hints.append("🎯 Divisible by 8")
        if number % 9 == 0:
            divisibility_hints.append("🌟 Divisible by 9")
        if number % 11 == 0:
            divisibility_hints.append("🔥 Divisible by 11")
        if number % 25 == 0:
            divisibility_hints.append("💎 Divisible by 25")
        if number % 12 == 0:
            divisibility_hints.append("⭐ Divisible by 12")
        if number % 15 == 0:
            divisibility_hints.append("🎪 Divisible by 15")
        if number % 20 == 0:
            divisibility_hints.append("🏆 Divisible by 20")
        
        if divisibility_hints:
            hint_variants.extend(random.sample(divisibility_hints, min(3, len(divisibility_hints))))
        
        if number > 10000:
            if len(digits) == 5:
                hint_variants.append("🏆 A majestic 5-digit number")
                if digits[0] == 2:
                    hint_variants.append("🚀 Begins with the number of decades in this range")
                elif digits[0] == 1:
                    hint_variants.append("🎯 Starts with unity, the beginning of all")
        
        if number == int(str(number)[::-1]):  # Palindrome
            hint_variants.append("🪞 This number reads the same forwards and backwards")
        
        digit_product = 1
        for d in digits:
            digit_product *= d
        if digit_product > 0:  # Avoid hint if contains zero
            if digit_product < 50:
                hint_variants.append(f"🎲 The product of digits is {digit_product}")
        
        selected_hint = random.choice(hint_variants)
        
        return {
            'encoded_hint': selected_hint,  # No encoding - direct strategic hint!
            'hint_type': 'CLEAR'  # Clear hint type for strategic hints
        }
    
    def start_nightmare_game(self, telegram_id: int, player_name: str) -> dict:
        """Start new nightmare mode game with Python-based hint generation"""
        import random
        
        conn = self.get_db_connection()
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT game_id FROM nightmare_games 
                WHERE player_telegram_id = %s AND NOT is_completed
            """, (telegram_id,))
            
            if cursor.fetchone():
                other_games = self.get_all_active_games(telegram_id)
                if other_games['guess'] or other_games['chase']:
                    transition_msg, _ = self.create_game_switch_message(telegram_id, "Nightmare Mode")
                    return {'error': transition_msg}
                else:
                    return {'error': 'You have an active nightmare mode game! Use /nightmare to continue it.'}
            
            range_min = 1
            range_max = 10000  # Fixed range as requested
            target_number = random.randint(range_min, range_max)
            max_attempts = 3  # Fixed to 3 attempts as requested
            
            hint_data = self.generate_nightmare_hint(target_number)
            
            cursor.execute("""
                INSERT INTO nightmare_games 
                (player_telegram_id, player_name, original_number, current_number, shift_seed, 
                 encoded_hint, hint_type, decoded_hint, attempts_used, max_attempts, 
                 game_range_min, game_range_max, is_completed, is_won, guess_history)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING game_id
            """, (telegram_id, player_name, target_number, target_number, random.randint(1, 1000),
                  hint_data['encoded_hint'], hint_data['hint_type'], hint_data['encoded_hint'],
                  0, max_attempts, range_min, range_max, False, False, '[]'))
            
            result = cursor.fetchone()
            if result:
                game_id = result['game_id']  # Access by column name for RealDictCursor
                conn.commit()
                logger.info(f"Successfully created nightmare game {game_id} for player {telegram_id}")
                
                return {
                    'game_id': game_id,
                    'encoded_hint': hint_data['encoded_hint'],
                    'hint_type': hint_data['hint_type'],
                    'range_min': range_min,
                    'range_max': range_max,
                    'max_attempts': max_attempts
                }
            else:
                logger.error(f"INSERT returned no result for nightmare game creation for {telegram_id}")
                return {'error': 'Failed to create nightmare game - database did not return game ID'}
            
        except Exception as e:
            logger.error(f"Error starting nightmare game for {telegram_id}: {e}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            conn.rollback()
            
            error_str = str(e).lower()
            if "function" in error_str and "does not exist" in error_str:
                return {'error': 'Database configuration issue. Please contact admin.'}
            elif "integer out of range" in error_str:
                return {'error': 'Number range error. Please try again.'}
            elif "column" in error_str and "does not exist" in error_str:
                return {'error': 'Database schema mismatch. Please contact admin.'}
            else:
                return {'error': f'Failed to start nightmare game: {str(e)}'}
        finally:
            self.return_db_connection(conn)
    
    def make_nightmare_guess(self, game_id: int, guess: int) -> dict:
        """Make a guess in nightmare mode with Python logic"""
        import random
        
        conn = self.get_db_connection()
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM nightmare_games 
                WHERE game_id = %s AND NOT is_completed
            """, (game_id,))
            game = cursor.fetchone()
            
            if not game:
                return {'error': 'Game not found or already completed'}
            
            is_correct = (guess == game['current_number'])
            attempts_used = game['attempts_used'] + 1
            attempts_remaining = game['max_attempts'] - attempts_used
            game_over = is_correct or attempts_remaining <= 0
            won = is_correct
            
            if game_over:
                cursor.execute("""
                    UPDATE nightmare_games 
                    SET attempts_used = %s, is_completed = TRUE, completed_at = NOW()
                    WHERE game_id = %s
                """, (attempts_used, game_id))
                
                if won:
                    self.award_shards(game['player_telegram_id'], 10000, 'Nightmare Mode Victory', 
                                    game['player_name'])
                    
                    try:
                        async def send_nightmare_victory_log():
                            try:
                                await self.send_admin_log(
                                    'nightmare_win',
                                    f"🔥 NIGHTMARE MODE VICTORY! [20K/2ATT] Player: {game['player_name']} | Target: {game['current_number']:,} | Guess: {guess:,} | Attempts: {attempts_used}/2 | Reward: 10,000 shards",
                                    game['player_telegram_id'],
                                    None
                                )
                            except Exception as e:
                                logger.debug(f"Could not send nightmare victory log: {e}")
                        
                        deferred_scheduler.schedule_task(send_nightmare_victory_log())
                    except Exception as e:
                        logger.debug(f"Could not schedule nightmare victory log: {e}")
                    
                    cursor.execute("""
                        UPDATE players SET 
                            nightmare_games_played = nightmare_games_played + 1,
                            nightmare_games_won = nightmare_games_won + 1,
                            nightmare_best_attempts = CASE 
                                WHEN nightmare_best_attempts = 0 OR %s < nightmare_best_attempts 
                                THEN %s ELSE nightmare_best_attempts END,
                            nightmare_total_attempts = nightmare_total_attempts + %s,
                            nightmare_last_played = NOW(),
                            has_shard_mastermind = TRUE
                        WHERE telegram_id = %s
                    """, (attempts_used, attempts_used, attempts_used, game['player_telegram_id']))
                else:
                    cursor.execute("""
                        UPDATE players SET 
                            nightmare_games_played = nightmare_games_played + 1,
                            nightmare_total_attempts = nightmare_total_attempts + %s,
                            nightmare_last_played = NOW()
                        WHERE telegram_id = %s
                    """, (attempts_used, game['player_telegram_id']))
            else:
                cursor.execute("""
                    UPDATE nightmare_games 
                    SET attempts_used = %s
                    WHERE game_id = %s
                """, (attempts_used, game_id))
                
                guess_direction = "high" if guess > game['current_number'] else "low"
                
                if abs(guess - game['current_number']) <= 500:
                    shift_amount = random.randint(10, 100)
                    shift_info = "🎯 You were close! Small shift applied."
                elif abs(guess - game['current_number']) <= 2000:
                    shift_amount = random.randint(100, 500)
                    shift_info = "🎲 Moderate distance. Medium shift applied."
                else:
                    shift_amount = random.randint(500, 1500)
                    shift_info = "🌪️ Way off target! Large shift applied."
                
                if guess_direction == "high":
                    new_target = (game['current_number'] - shift_amount) % (game['game_range_max'] - game['game_range_min'] + 1) + game['game_range_min']
                else:
                    new_target = (game['current_number'] + shift_amount) % (game['game_range_max'] - game['game_range_min'] + 1) + game['game_range_min']
                
                hint_data = self.generate_nightmare_hint(new_target)
                
                cursor.execute("""
                    UPDATE nightmare_games 
                    SET current_number = %s, encoded_hint = %s, hint_type = %s, decoded_hint = %s
                    WHERE game_id = %s
                """, (new_target, hint_data['encoded_hint'], hint_data['hint_type'], hint_data['encoded_hint'], game_id))
            
            conn.commit()
            
            if won:
                message = "🎉 CONGRATULATIONS! You've conquered the nightmare!"
            elif game_over:
                message = f"💀 Game Over! The number was {game['current_number']:,}"
            else:
                message = f"❌ Wrong! {shift_info if 'shift_info' in locals() else 'The number has shifted.'}"
            
            return {
                'is_correct': is_correct,
                'attempts_remaining': attempts_remaining,
                'game_over': game_over,
                'won': won,
                'current_number': game['current_number'],
                'new_hint': hint_data['encoded_hint'] if not won and not game_over else None,
                'shift_info': shift_info if 'shift_info' in locals() else None,
                'guess_direction': guess_direction if 'guess_direction' in locals() else None,
                'message': message
            }
            
        except Exception as e:
            logger.error(f"Error making nightmare guess for game {game_id}: {e}")
            conn.rollback()
            return {'error': str(e)}
        finally:
            self.return_db_connection(conn)
    
    def get_nightmare_game(self, telegram_id: int) -> dict:
        """Get active nightmare game for player"""
        conn = self.get_db_connection()
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM nightmare_games 
                WHERE player_telegram_id = %s AND NOT is_completed
            """, (telegram_id,))
            
            result = cursor.fetchone()
            return dict(result) if result else {}
            
        except Exception as e:
            logger.error(f"Error getting nightmare game for {telegram_id}: {e}")
            return {}
        finally:
            self.return_db_connection(conn)
    
    def get_nightmare_stats(self, telegram_id: int) -> dict:
        """Get nightmare mode statistics for player"""
        conn = self.get_db_connection()
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT nightmare_games_played, nightmare_games_won, 
                       nightmare_best_attempts, nightmare_total_attempts,
                       nightmare_last_played, has_shard_mastermind
                FROM players WHERE telegram_id = %s
            """, (telegram_id,))
            
            result = cursor.fetchone()
            return dict(result) if result else {}
            
        except Exception as e:
            logger.error(f"Error getting nightmare stats for {telegram_id}: {e}")
            return {}
        finally:
            self.return_db_connection(conn)
    
    def get_nightmare_leaderboard(self, limit: int = 10) -> list:
        """Get nightmare mode leaderboard"""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM nightmare_leaderboard LIMIT %s
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Error getting nightmare leaderboard: {e}")
            return []
        finally:
            self.return_db_connection(conn)
    
    def decode_nightmare_hint(self, encoded_hint: str, hint_type: str) -> str:
        """Decode nightmare mode hint"""
        try:
            if hint_type.upper() == 'CLEAR':
                return encoded_hint
            elif hint_type.upper() == 'ROT13':
                import codecs
                return codecs.decode(encoded_hint, 'rot13')
            elif hint_type.upper() == 'BASE64':
                import base64
                return base64.b64decode(encoded_hint.encode()).decode('utf-8')
            else:
                return "Unknown encoding type"
        except Exception as e:
            logger.error(f"Error decoding hint: {e}")
            return "Failed to decode hint"
    
    def award_shard_mastermind_achievement(self, telegram_id: int) -> bool:
        """Request admin confirmation for Shard Mastermind achievement"""
        conn = self.get_db_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT has_shard_mastermind FROM players 
                WHERE telegram_id = %s
            """, (telegram_id,))
            
            result = cursor.fetchone()
            if result and result[0]:
                return True  # Already has achievement
            
            player = self.find_player_by_telegram_id(telegram_id)
            if not player:
                return False
            
            details = {
                'achievement': 'Shard Mastermind',
                'completed_nightmare': True,
                'timestamp': str(datetime.now()),
                'reward': '10,000 shards + exclusive title'
            }
            
            cursor.execute("SELECT request_admin_confirmation(%s, %s, %s, %s)", 
                          (telegram_id, player['display_name'], 'Shard Mastermind', json.dumps(details)))
            
            confirmation_id = cursor.fetchone()[0]
            conn.commit()
            
            self._notify_admins_achievement_request(confirmation_id, player, details)
            
            logger.info(f"Requested admin confirmation for Shard Mastermind achievement - Player: {telegram_id}, Confirmation ID: {confirmation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error requesting Shard Mastermind confirmation for {telegram_id}: {e}")
            conn.rollback()
            return False
        finally:
            self.return_db_connection(conn)
    
    def _notify_admins_achievement_request(self, confirmation_id: int, player: dict, details: dict):
        """Send notification to admins about achievement confirmation request"""
        try:
            admin_list = self.get_admin_list()
            
            message = f"🚨 <b>SPECIAL ACHIEVEMENT CONFIRMATION NEEDED</b> 🚨\n"
            message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            message += f"🧠⚡ <b>Achievement:</b> Shard Mastermind ⚡🧠\n"
            message += f"👤 <b>Player:</b> {H(player['display_name'])}\n"
            message += f"🆔 <b>Telegram ID:</b> <code>{player['telegram_id']}</code>\n"
            message += f"🎮 <b>Accomplished:</b> Conquered Nightmare Mode\n"
            message += f"💠 <b>Reward:</b> 10,000 shards + exclusive title\n"
            message += f"🕒 <b>Requested:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            message += f"⚠️ <b>This is a legendary achievement requiring admin approval!</b>\n"
            message += f"🔍 <b>Confirmation ID:</b> <code>{confirmation_id}</code>\n\n"
            message += f"Use /confirmachievement {confirmation_id} to approve"
            
            for admin_id in admin_list:
                try:
                    deferred_scheduler.schedule_task(
                        self.application.bot.send_message(
                            chat_id=admin_id,
                            text=message,
                            parse_mode='HTML'
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to notify admin {admin_id} about achievement request: {e}")
                    
        except Exception as e:
            logger.error(f"Error notifying admins about achievement request: {e}")
    
    def get_pending_achievement_confirmations(self) -> list:
        """Get list of pending achievement confirmations"""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM admin_confirmations 
                WHERE NOT is_processed 
                ORDER BY created_at DESC
            """)
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Error getting pending confirmations: {e}")
            return []
        finally:
            self.return_db_connection(conn)
    
    def confirm_achievement_admin(self, confirmation_id: int, admin_telegram_id: int, notes: str = None) -> bool:
        """Admin confirms special achievement"""
        conn = self.get_db_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            cursor.execute("SELECT confirm_special_achievement(%s, %s, %s)", 
                          (confirmation_id, admin_telegram_id, notes))
            
            success = cursor.fetchone()[0]
            conn.commit()
            
            if success:
                logger.info(f"Admin {admin_telegram_id} confirmed achievement (ID: {confirmation_id})")
            
            return success
            
        except Exception as e:
            logger.error(f"Error confirming achievement: {e}")
            conn.rollback()
            return False
        finally:
            self.return_db_connection(conn)

    def get_comprehensive_stats(self) -> dict:
        """Get comprehensive bot statistics for admin panel"""
        conn = self.get_db_connection()
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM players")
            total_players = cursor.fetchone()[0] or 0
            
            cursor.execute("""
                SELECT COUNT(DISTINCT telegram_id) 
                FROM shard_transactions 
                WHERE performed_at >= CURRENT_DATE - INTERVAL '7 days'
            """)
            active_players = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT COUNT(*) FROM chase_games")
            chase_games = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT COUNT(*) FROM guess_games")
            guess_games = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT COUNT(*) FROM nightmare_games")
            nightmare_games = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT COALESCE(SUM(current_shards), 0) FROM players")
            total_shards = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT COALESCE(AVG(current_shards), 0) FROM players WHERE current_shards > 0")
            avg_balance = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT COUNT(*) FROM achievements")
            total_achievements = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT COUNT(DISTINCT achievement_name) FROM achievements")
            unique_achievements = cursor.fetchone()[0] or 0
            
            return {
                'total_players': total_players,
                'active_players': active_players,
                'chase_games': chase_games,
                'guess_games': guess_games,
                'nightmare_games': nightmare_games,
                'total_shards': total_shards,
                'avg_balance': round(avg_balance, 1),
                'total_achievements': total_achievements,
                'unique_achievements': unique_achievements
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive stats: {e}")
            return {}
        finally:
            self.return_db_connection(conn)

    def get_all_players_with_shards(self) -> List[Dict]:
        """Get all players with their shard balances"""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT telegram_id, username, display_name, current_shards, total_shards_earned
                FROM players
                ORDER BY current_shards DESC
            """)
            
            results = cursor.fetchall()
            players = []
            
            for row in results:
                players.append({
                    'telegram_id': row[0],
                    'username': row[1] or '',
                    'display_name': row[2] or f"User {row[0]}",
                    'shards': row[3] or 0,
                    'total_earned': row[4] or 0
                })
                
            return players
            
        except Exception as e:
            logger.error(f"Error getting all players with shards: {e}")
            return []
        finally:
            self.return_db_connection(conn)

    def get_shard_leaderboard(self, limit: int = 10) -> List[Dict]:
        """Get shard leaderboard"""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT telegram_id, username, display_name, current_shards, total_shards_earned
                FROM players
                WHERE current_shards > 0
                ORDER BY current_shards DESC
                LIMIT %s
            """, (limit,))
            
            results = cursor.fetchall()
            players = []
            
            for row in results:
                players.append({
                    'telegram_id': row[0],
                    'username': row[1] or '',
                    'display_name': row[2] or f"User {row[0]}",
                    'shards': row[3] or 0,
                    'total_earned': row[4] or 0
                })
                
            return players
            
        except Exception as e:
            logger.error(f"Error getting shard leaderboard: {e}")
            return []
        finally:
            self.return_db_connection(conn)

    def get_all_players(self) -> List[Dict]:
        """Get all registered players"""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, telegram_id, username, display_name, title, created_at, updated_at
                FROM players 
                ORDER BY created_at DESC
            """)
            
            results = cursor.fetchall()
            players = []
            
            for row in results:
                players.append({
                    'id': row[0],
                    'telegram_id': row[1],
                    'username': row[2] or '',
                    'display_name': row[3] or f"User {row[1]}",
                    'title': row[4] or '',
                    'created_at': row[5],
                    'updated_at': row[6]
                })
                
            return players
            
        except Exception as e:
            logger.error(f"Error getting all players: {e}")
            return []
        finally:
            self.return_db_connection(conn)

    def get_all_chase_games(self) -> List[Dict]:
        """Get all active chase games"""
        return []  # Chase games stored in memory, returning empty for now

    # ====================================
    # ====================================
    
    def create_auction_proposal(self, creator_id: int, creator_name: str) -> int:
        """Create a new auction proposal"""
        self.proposal_counter += 1
        proposal_id = self.proposal_counter
        
        proposal = AuctionProposal(proposal_id, creator_id, creator_name)
        self.auction_proposals[proposal_id] = proposal
        
        logger.info(f"Created auction proposal {proposal_id} by {creator_name}")
        return proposal_id
    
    def update_proposal_data(self, proposal_id: int, data: dict) -> bool:
        """Update proposal with registration data"""
        if proposal_id not in self.auction_proposals:
            return False
        
        proposal = self.auction_proposals[proposal_id]
        
        if 'name' in data:
            proposal.name = data['name']
        if 'teams' in data:
            proposal.teams = data['teams']
        if 'purse' in data:
            proposal.purse = data['purse']
        if 'base_price' in data:
            proposal.base_price = data['base_price']
        
        return True
    
    def send_proposal_to_admins(self, proposal_id: int) -> bool:
        """Send completed proposal to admins for approval"""
        if proposal_id not in self.auction_proposals:
            return False
        
        proposal = self.auction_proposals[proposal_id]
        logger.info(f"Proposal {proposal_id} sent to admins for approval")
        return True
    
    def approve_auction_proposal(self, proposal_id: int, admin_id: int, admin_name: str) -> Optional[int]:
        """Approve a proposal and create approved auction"""
        if proposal_id not in self.auction_proposals:
            return None
        
        proposal = self.auction_proposals[proposal_id]
        proposal.status = "approved"
        proposal.admin_response_at = datetime.now()
        proposal.admin_id = admin_id
        proposal.admin_name = admin_name
        
        self.auction_counter += 1
        auction_id = self.auction_counter
        
        approved_auction = ApprovedAuction(auction_id, proposal)
        self.approved_auctions[auction_id] = approved_auction
        
        logger.info(f"Approved auction {auction_id} from proposal {proposal_id}")
        return auction_id
    
    def reject_auction_proposal(self, proposal_id: int, admin_id: int, admin_name: str) -> bool:
        """Reject a proposal"""
        if proposal_id not in self.auction_proposals:
            return False
        
        proposal = self.auction_proposals[proposal_id]
        proposal.status = "rejected"
        proposal.admin_response_at = datetime.now()
        proposal.admin_id = admin_id
        proposal.admin_name = admin_name
        
        logger.info(f"Rejected proposal {proposal_id}")
        return True
    
    def get_approved_auction(self, auction_id: int) -> Optional[ApprovedAuction]:
        """Get approved auction by ID"""
        return self.approved_auctions.get(auction_id)
    
    def start_captain_registration(self, auction_id: int) -> bool:
        """Start captain registration phase"""
        auction = self.get_approved_auction(auction_id)
        if not auction or auction.status != "setup":
            return False
        
        auction.status = "captain_reg"
        logger.info(f"Started captain registration for auction {auction_id}")
        return True
    
    def register_captain_request(self, auction_id: int, user_id: int, name: str, team_name: str) -> bool:
        """Register captain request for host approval"""
        auction = self.get_approved_auction(auction_id)
        if not auction or auction.status not in ["captain_reg", "player_reg", "ready"]:
            return False
        
        if user_id in auction.registered_captains:
            return False  # Already registered
        
        if user_id in auction.registered_players or user_id in auction.approved_players:
            return False  # Cannot be both captain and player
        
        registration = CaptainRegistration(user_id, name, team_name)
        auction.registered_captains[user_id] = registration
        
        logger.info(f"Captain registration request: {name} for team {team_name} in auction {auction_id}")
        return True
    
    def approve_captain(self, auction_id: int, user_id: int) -> bool:
        """Host approves a captain registration"""
        auction = self.get_approved_auction(auction_id)
        if not auction or user_id not in auction.registered_captains:
            return False
        
        registration = auction.registered_captains[user_id]
        registration.status = "approved"
        
        captain = ApprovedCaptain(user_id, registration.name, registration.team_name, auction.purse)
        auction.approved_captains[user_id] = captain
        
        logger.info(f"Approved captain {registration.name} for auction {auction_id}")
        return True
    
    def reject_captain(self, auction_id: int, user_id: int) -> bool:
        """Host rejects a captain registration"""
        auction = self.get_approved_auction(auction_id)
        if not auction or user_id not in auction.registered_captains:
            return False
        
        registration = auction.registered_captains[user_id]
        registration.status = "rejected"
        
        logger.info(f"Rejected captain {registration.name} for auction {auction_id}")
        return True
    
    def start_player_registration(self, auction_id: int) -> bool:
        """Start player registration phase (runs alongside captain registration)"""
        auction = self.get_approved_auction(auction_id)
        if not auction or auction.status not in ["captain_reg", "player_reg"]:
            return False
        
        auction.status = "player_reg"
        
        logger.info(f"Started player registration for auction {auction_id}")
        return True
    
    def register_player_request(self, auction_id: int, user_id: int, name: str, username: str = None) -> bool:
        """Register player request for host approval"""
        auction = self.get_approved_auction(auction_id)
        if not auction or auction.status not in ["captain_reg", "player_reg", "ready"]:
            return False
        
        if user_id in auction.registered_players:
            return False  # Already registered
        
        if user_id in auction.registered_captains or user_id in auction.approved_captains:
            return False  # Cannot be both captain and player
        
        registration = PlayerRegistration(user_id, name, username)
        auction.registered_players[user_id] = registration
        
        logger.info(f"Player registration request: {name} in auction {auction_id}")
        return True
    
    def approve_player(self, auction_id: int, user_id: int) -> bool:
        """Host approves a player registration"""
        auction = self.get_approved_auction(auction_id)
        if not auction or user_id not in auction.registered_players:
            return False
        
        registration = auction.registered_players[user_id]
        registration.status = "approved"
        
        player = ApprovedPlayer(user_id, registration.name, auction.base_price, registration.username)
        auction.approved_players[user_id] = player
        
        auction.player_queue.append(player)
        
        logger.info(f"Approved player {registration.name} for auction {auction_id}")
        return True
    
    def reject_player(self, auction_id: int, user_id: int) -> bool:
        """Host rejects a player registration"""
        auction = self.get_approved_auction(auction_id)
        if not auction or user_id not in auction.registered_players:
            return False
        
        registration = auction.registered_players[user_id]
        registration.status = "rejected"
        
        logger.info(f"Rejected player {registration.name} for auction {auction_id}")
        return True
    
    def close_registration(self, auction_id: int) -> bool:
        """Close registration and prepare for auction"""
        auction = self.get_approved_auction(auction_id)
        if not auction or auction.status not in ["captain_reg", "player_reg"]:
            return False
        
        auction.status = "ready"
        if auction.randomize_players:
            import random
            random.shuffle(auction.player_queue)
            logger.info(f"Closed registration for auction {auction_id}, {len(auction.player_queue)} players ready (randomized)")
        else:
            logger.info(f"Closed registration for auction {auction_id}, {len(auction.player_queue)} players ready (sequential order)")
        
        return True
    
    def start_auction_bidding(self, auction_id: int) -> bool:
        """Start the auction bidding process"""
        auction = self.get_approved_auction(auction_id)
        if not auction or auction.status != "ready":
            return False

        if not auction.approved_players:
            return False
        
        if not auction.player_queue:
            import random
            auction.player_queue = list(auction.approved_players.values())
            random.shuffle(auction.player_queue)
        
        auction.status = "active"
        auction.current_player_index = 0
        auction.current_player = auction.player_queue[0]
        auction.current_bids = {}
        auction.highest_bidder = None
        auction.highest_bid = auction.base_price
        
        logger.info(f"Started auction bidding for auction {auction_id}, shuffled {len(auction.player_queue)} players")
        return True
    
    def place_bid(self, auction_id: int, captain_id: int, amount: int) -> bool:
        """Place a bid in active auction"""
        auction = self.get_approved_auction(auction_id)
        if not auction or auction.status != "active" or not auction.current_player:
            return False
        
        captain = auction.approved_captains.get(captain_id)
        if not captain or captain.purse < amount:
            return False
        
        auction.current_bids[captain_id] = {
            'captain': captain,
            'amount': amount,
            'timestamp': datetime.now()
        }
        
        logger.info(f"Bid placed: {captain.name} bid {amount}Cr for {auction.current_player.name}")
        return True
    
    def captain_out(self, auction_id: int, captain_id: int) -> bool:
        """Captain exits bidding for current player"""
        auction = self.get_approved_auction(auction_id)
        if not auction or auction.status != "active":
            return False
        
        if captain_id in auction.current_bids:
            del auction.current_bids[captain_id]
        
        logger.info(f"Captain {captain_id} opted out of current player")
        return True
    
    def sell_current_player(self, auction_id: int) -> bool:
        """Sell current player to highest bidder"""
        auction = self.get_approved_auction(auction_id)
        if not auction or auction.status != "active" or not auction.current_player:
            return False
        
        lock_acquired = auction._bid_lock.acquire(blocking=True, timeout=5)
        if not lock_acquired:
            logger.error("Failed to acquire lock for sell_current_player")
            return False
        
        try:
            sold_player = auction.current_player
            
            if not auction.current_bids:
                if not hasattr(auction, 'sold_players'):
                    auction.sold_players = {}
                auction.sold_players[sold_player.user_id] = {
                    'player': sold_player,
                    'team': 'UNSOLD',
                    'amount': 0,
                    'captain': None
                }
                
                auction.player_queue.remove(sold_player)
                if auction.player_queue:
                    auction.current_player = auction.player_queue[0]
                    auction.current_bids = {}
                    auction.highest_bidder = None  # Sync with current_bids
                    auction.highest_bid = auction.base_price
                else:
                    auction.current_player = None
                    auction.status = "completed"
                logger.info(f"Player unsold: {sold_player.name}")
                return True
            
            highest_bid_data = max(auction.current_bids.values(), key=lambda x: x['amount'])
            captain = highest_bid_data['captain']
            amount = highest_bid_data['amount']
            
            if captain.purse < amount:
                logger.error(f"PURSE CHECK FAILED: {captain.team_name} has {captain.purse} but bid {amount}")
                return False
            
            sold_player.is_sold = True
            sold_player.winning_team = captain.team_name
            sold_player.sold_price = amount
            
            captain.purse -= amount
            captain.spent = getattr(captain, 'spent', 0) + amount
            if not hasattr(captain, 'players'):
                captain.players = []
            captain.players.append(sold_player.name)
            
            if not hasattr(auction, 'sold_players'):
                auction.sold_players = {}
            auction.sold_players[sold_player.user_id] = {
                'player': sold_player,
                'team': captain.team_name,
                'amount': amount,
                'captain': captain.name
            }
            
            auction.player_queue.remove(sold_player)
            if auction.player_queue:
                auction.current_player = auction.player_queue[0]
                auction.current_bids = {}
                auction.highest_bidder = None  # Sync with current_bids reset
                auction.highest_bid = auction.base_price
            else:
                auction.current_player = None
                auction.status = "completed"
            
            logger.info(f"Player sold: {sold_player.name} to {captain.team_name} for {amount}Cr")
            return True
        finally:
            auction._bid_lock.release()
    
    def skip_current_player(self, auction_id: int) -> bool:
        """Mark current player as unsold (skip)"""
        auction = self.get_approved_auction(auction_id)
        if not auction or auction.status != "active" or not auction.current_player:
            return False
        
        lock_acquired = auction._bid_lock.acquire(blocking=True, timeout=5)
        if not lock_acquired:
            logger.error("Failed to acquire lock for skip_current_player")
            return False
        
        try:
            current = auction.current_player
            auction.player_queue.remove(current)
            auction.unsold_players[current.user_id] = current
            
            if auction.player_queue:
                auction.current_player = auction.player_queue[0]
                auction.current_bids = {}
                auction.highest_bidder = None
                auction.highest_bid = auction.base_price
            else:
                auction.current_player = None
            
            logger.info(f"Marked player {current.name} as UNSOLD")
            return True
        finally:
            auction._bid_lock.release()
    
    def assign_player_manually(self, auction_id: int, player_id: int, captain_id: int, amount: int) -> bool:
        """Manually assign player to captain"""
        auction = self.get_approved_auction(auction_id)
        if not auction:
            return False
        
        player = auction.approved_players.get(player_id)
        captain = auction.approved_captains.get(captain_id)
        
        if not player or not captain or captain.purse < amount:
            logger.error(f"Manual assignment validation failed: player={player is not None}, captain={captain is not None}, purse_check={captain.purse >= amount if captain else False}")
            return False
        
        lock_acquired = auction._bid_lock.acquire(blocking=True, timeout=5)
        if not lock_acquired:
            logger.error("Failed to acquire lock for assign_player_manually")
            return False
        
        try:
            player.is_sold = True
            player.winning_team = captain.team_name
            player.sold_price = amount
            
            captain.purse -= amount
            captain.spent = getattr(captain, 'spent', 0) + amount
            if not hasattr(captain, 'players'):
                captain.players = []
            captain.players.append(player.name)
            
            if not hasattr(auction, 'sold_players'):
                auction.sold_players = {}
            auction.sold_players[player_id] = {
                'player': player,
                'team': captain.team_name,
                'amount': amount,
                'captain': captain.name
            }
            
            if auction.current_player and auction.current_player.user_id == player_id:
                auction.player_queue.remove(player)
                if auction.player_queue:
                    auction.current_player = auction.player_queue[0]
                    auction.current_bids = {}
                    auction.highest_bidder = None  # Sync with current_bids reset
                    auction.highest_bid = auction.base_price
                else:
                    auction.current_player = None
                    auction.status = "completed"
            
            logger.info(f"Manual assignment: {player.name} to {captain.team_name} for {amount}Cr")
            return True
        finally:
            auction._bid_lock.release()
    
    def end_auction(self, auction_id: int) -> bool:
        """End auction and finalize results"""
        auction = self.get_approved_auction(auction_id)
        if not auction:
            return False
        
        auction.status = "completed"
        auction.current_player = None
        
        logger.info(f"Auction {auction_id} completed")
        return True
    
    def pause_auction(self, auction_id: int) -> bool:
        """Pause or resume an active auction"""
        auction = self.get_approved_auction(auction_id)
        if not auction or auction.status != "active":
            return False
        
        auction.is_paused = not auction.is_paused
        status = "paused" if auction.is_paused else "resumed"
        logger.info(f"Auction {auction_id} {status}")
        return True
    
    def rebid_player(self, auction_id: int, player_username: str) -> tuple:
        """Bring back any player (sold or unsold) for rebidding by username"""
        auction = self.get_approved_auction(auction_id)
        if not auction or auction.status != "active":
            return (False, "Auction not found or not active", None)
        
        player = None
        player_id = None
        
        for pid, player_data in auction.sold_players.items():
            p = player_data['player']
            p_username = getattr(p, 'username', '').lower() if hasattr(p, 'username') else ''
            if p_username == player_username.lower().replace('@', '') or p.name.lower() == player_username.lower().replace('@', ''):
                player = p
                player_id = pid
                captain = player_data['captain']
                captain.purse += player_data['amount']
                captain.spent -= player_data['amount']
                captain.players.remove(player)
                auction.sold_players.pop(pid)
                break
        
        if not player:
            for pid, p in auction.unsold_players.items():
                p_username = getattr(p, 'username', '').lower() if hasattr(p, 'username') else ''
                if p_username == player_username.lower().replace('@', '') or p.name.lower() == player_username.lower().replace('@', ''):
                    player = p
                    player_id = pid
                    auction.unsold_players.pop(pid)
                    break
        
        if not player:
            for pid, p in auction.approved_players.items():
                p_username = getattr(p, 'username', '').lower() if hasattr(p, 'username') else ''
                if p_username == player_username.lower().replace('@', '') or p.name.lower() == player_username.lower().replace('@', ''):
                    player = p
                    player_id = pid
                    break
        
        if not player:
            return (False, f"Player '{player_username}' not found", None)
        
        auction.player_queue.insert(0, player)
        auction.current_player = player
        auction.current_bids = {}
        auction.highest_bidder = None
        auction.highest_bid = auction.base_price
        
        logger.info(f"Player {player.name} brought back for rebidding in auction {auction_id}")
        return (True, "Success", player)
    
    def assign_unsold_player(self, auction_id: int, player_username: str, team_name: str, amount: int) -> tuple:
        """Assign an unsold player to a captain by username and team name"""
        auction = self.get_approved_auction(auction_id)
        if not auction:
            return (False, "Auction not found", None, None)
        
        player = None
        player_id = None
        for pid, p in auction.unsold_players.items():
            p_username = getattr(p, 'username', '').lower() if hasattr(p, 'username') else ''
            if p_username == player_username.lower().replace('@', '') or p.name.lower() == player_username.lower().replace('@', ''):
                player = p
                player_id = pid
                break
        
        if not player:
            return (False, "Player not found in unsold list", None, None)
        
        captain = None
        for cap in auction.approved_captains.values():
            if cap.team_name.lower() == team_name.lower():
                captain = cap
                break
        
        if not captain:
            return (False, f"Team '{team_name}' not found", None, None)
        
        if captain.purse < amount:
            return (False, "Insufficient funds", player, captain)
        
        auction.unsold_players.pop(player_id)
        
        captain.purse -= amount
        captain.spent += amount
        captain.players.append(player)
        
        auction.sold_players[player_id] = {
            'player': player,
            'captain': captain,
            'amount': amount,
            'timestamp': datetime.now()
        }
        
        logger.info(f"Unsold player {player.name} manually assigned to {captain.name} for {amount}Cr")
        return (True, "Success", player, captain)
    
    def add_player_to_auction(self, auction_id: int, user_id: int, name: str, username: str = None) -> bool:
        """Add a new player to ongoing auction"""
        auction = self.get_approved_auction(auction_id)
        if not auction:
            return False
        
        if user_id in auction.approved_players:
            return False
        
        player = ApprovedPlayer(user_id, name, auction.base_price, username)
        auction.approved_players[user_id] = player
        auction.player_queue.append(player)
        
        logger.info(f"Added player {name} to ongoing auction {auction_id}")
        return True

# ====================================
# ====================================

@check_banned
async def daily_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Claim daily shard bonus"""
    try:
        user_id = update.effective_user.id
        user = update.effective_user
        
        username = user.username or ""
        display_name = user.full_name or user.first_name or f"User{user.id}"
        bot_instance.create_or_update_player(user_id, username, display_name)
        
        success, amount, streak_days, message = bot_instance.claim_daily_shards(user_id)
        
        if success:
            current_shards, total_earned = bot_instance.get_player_shards(user_id)
            
            response = (
                f"🎁 <b>DAILY ARENA REWARD</b> 🏆\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"⚡ <b>BONUS COLLECTED!</b> ⚡\n\n"
                f"👑 <b>CHAMPION:</b> {H(display_name)}\n"
                f"💎 <b>TODAY'S REWARD:</b> +{amount:,} shards\n"
                f"🔥 <b>STREAK POWER:</b> {streak_days} day{'s' if streak_days > 1 else ''}\n"
                f"💠 <b>TOTAL WEALTH:</b> {current_shards:,} shards\n\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
            )
            
            if streak_days == 1:
                response += f"💡 <i>Come back tomorrow for a streak bonus!</i>\n"
            elif streak_days < 7:
                response += f"💪 <i>Keep the streak going! +{min((streak_days * 10), 200)} bonus tomorrow</i>\n"
            else:
                response += f"🏆 <i>Amazing streak! Maximum bonus achieved!</i>\n"
            
            response += f"🎮 <i>Play games to earn even more shards!</i>"
            
            await safe_send(update.message.reply_text, response, parse_mode='HTML')
        else:
            await safe_send(update.message.reply_text, 
                           f"⏰ <b>Daily Bonus</b>\n\n"
                           f"❌ {message}\n\n"
                           f"🕒 <i>Daily bonuses reset at midnight UTC</i>", 
                           parse_mode='HTML')
        
    except Exception as e:
        log_exception("daily_command", e, update.effective_user.id)
        await safe_send(update.message.reply_text, 
                       "❌ Error claiming daily bonus. Please try again.", 
                       parse_mode='HTML')

@check_banned
async def shards_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show detailed shard balance and last 5 transactions"""
    try:
        user_id = update.effective_user.id
        user = update.effective_user
        
        target_user_id = user_id
        if context.args:
            target_identifier = context.args[0]
            target_player = bot_instance.find_player_by_identifier(target_identifier)
            if target_player:
                target_user_id = target_player['telegram_id']
            else:
                await safe_send(update.message.reply_text, 
                               f"❌ Player '{H(target_identifier)}' not found.")
                return
        
        if target_user_id == user_id:
            username = user.username or ""
            display_name = user.full_name or user.first_name or f"User{user.id}"
            bot_instance.create_or_update_player(user_id, username, display_name)
            target_player = {'display_name': display_name, 'telegram_id': user_id}
        
        current_shards, total_earned = bot_instance.get_player_shards(target_user_id)
        
        transactions = bot_instance.get_shard_transactions(target_user_id, 5)
        
        conn = bot_instance.get_db_connection()
        if not conn:
            await safe_send(update.message.reply_text, "❌ Database connection failed")
            return
            
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) + 1 as rank
                FROM players 
                WHERE shards > %s AND (shards > 0 OR total_shards_earned > 0)
            """, (current_shards,))
            
            user_rank = cursor.fetchone()[0]
            
            if target_user_id == user_id:
                message = f"💎 <b>YOUR SHARD BALANCE</b> 💎\n"
            else:
                message = f"💎 <b>{H(target_player['display_name'][:20])}'S SHARDS</b> 💎\n"
            
            message += f"━━━━━━━━━━━━━━━━━━━━\n"
            message += f"💠 <b>Current Balance:</b> {current_shards:,} shards\n"
            message += f"📈 <b>Total Earned:</b> {total_earned:,} shards\n"
            message += f"🏅 <b>Global Rank:</b> #{user_rank}\n\n"
            
            if transactions:
                message += f"📊 <b>Recent Activity (Last 5):</b>\n"
                for transaction in transactions:
                    trans_type = transaction.get('transaction_type', 'UNKNOWN')
                    amount = transaction.get('amount', 0)
                    source = transaction.get('source', 'Unknown')
                    created = transaction.get('created_at')
                    
                    if created:
                        try:
                            if isinstance(created, str):
                                created = datetime.fromisoformat(created.replace('Z', '+00:00'))
                            date_str = created.strftime("%m/%d %H:%M")
                        except:
                            date_str = "Recent"
                    else:
                        date_str = "Recent"
                    
                    if trans_type == 'EARN':
                        symbol = "+"
                        emoji = "🎮"
                        desc = source.title()
                    elif trans_type == 'SPEND':
                        symbol = "-"
                        emoji = "💸"
                        desc = source.title()
                    elif trans_type == 'ADMIN_GIVE':
                        symbol = "+"
                        emoji = "🎁"
                        desc = "Admin Award"
                    elif trans_type == 'ADMIN_REMOVE':
                        symbol = "-"
                        emoji = "⚖️"
                        desc = "Admin Action"
                    elif trans_type == 'DAILY':
                        symbol = "+"
                        emoji = "📅"
                        desc = "Daily Bonus"
                    else:
                        symbol = "±"
                        emoji = "💠"
                        desc = trans_type.title()
                    
                    message += f"{emoji} {symbol}{amount:,} - {desc} ({date_str})\n"
                
                message += f"\n━━━━━━━━━━━━━━━━━━━━\n"
            else:
                message += f"📊 <b>No transaction history available</b>\n\n"
                message += f"━━━━━━━━━━━━━━━━━━━━\n"
            
            message += f"💡 <b>How to Earn More Shards:</b>\n"
            message += f"🎮 Play games: 25-90 per game\n"
            message += f"📅 Daily bonus: 50-250 per day\n"
            message += f"🏆 Achievements: 100-200 each\n"
            message += f"🐐 Daily GOAT: 300 bonus\n\n"
            message += f"Use /shardlb to see top holders!"
            
            await safe_send(update.message.reply_text, message, parse_mode='HTML')
            
        finally:
            bot_instance.return_db_connection(conn)
        
    except Exception as e:
        log_exception("shards_command", e, update.effective_user.id)
        await safe_send(update.message.reply_text, 
                       "❌ Error retrieving shard balance. Please try again.", 
                       parse_mode='HTML')

@check_banned

async def give_shards_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin command to give shards to a player"""
    if not bot_instance.is_admin(update.effective_user.id):
        await safe_send(update.message.reply_text, 
                       "❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can give shards!", 
                       parse_mode='HTML')
        return
    
    try:
        args_valid, error_msg = InputValidator.validate_command_args(context.args, 2)
        if not args_valid:
            await safe_send(update.message.reply_text,
                           f"{error_msg}\n\n"
                           "📝 <b>Usage:</b>\n"
                           "/giveshards &lt;player&gt; &lt;amount&gt; [reason]\n\n"
                           "<b>Examples:</b>\n"
                           "• /giveshards @user 500\n"
                           "• /giveshards 123456789 1000 Tournament winner\n"
                           "• /giveshards PlayerName 250 Event participation",
                           parse_mode='HTML')
            return
        
        player_identifier = context.args[0]
        try:
            amount = int(context.args[1])
            if amount <= 0:
                raise ValueError("Amount must be positive")
            if amount > 2000000000:  # 2 billion limit
                raise ValueError("Amount too large")
        except ValueError as e:
            if "too large" in str(e):
                await safe_send(update.message.reply_text, "❌ Amount too large. Maximum allowed is 2,000,000,000 shards.")
            else:
                await safe_send(update.message.reply_text, "❌ Invalid amount. Must be a positive number.")
            return
        
        reason = InputValidator.safe_join_args(context.args, 2, "Admin award")
        
        player = bot_instance.find_player_by_identifier(player_identifier)
        if not player:
            await safe_send(update.message.reply_text, 
                           f"❌ Player '{H(player_identifier)}' not found.\n\n"
                           "💡 Make sure they've used the bot before!")
            return
        
        success = bot_instance.give_admin_shards(
            player['telegram_id'], 
            amount, 
            update.effective_user.id, 
            reason
        )
        
        if success:
            current_shards, total_earned = bot_instance.get_player_shards(player['telegram_id'])
            
            await bot_instance.send_admin_log(
                'admin_action',
                f"Shards awarded: +{amount:,} to {player['display_name']} (ID: {player['telegram_id']}) | Reason: {reason} | New balance: {current_shards:,}",
                update.effective_user.id,
                update.effective_user.username
            )
            
            await safe_send(update.message.reply_text,
                           f"✅ <b>SHARDS AWARDED!</b>\n\n"
                           f"👤 <b>Player:</b> {H(player['display_name'])}\n"
                           f"💠 <b>Amount:</b> +{amount:,} shards\n"
                           f"💰 <b>New Balance:</b> {current_shards:,} shards\n"
                           f"📝 <b>Reason:</b> {H(reason)}\n"
                           f"👨‍💼 <b>Given by:</b> {H(update.effective_user.full_name or 'Admin')}",
                           parse_mode='HTML')
        else:
            await safe_send(update.message.reply_text, 
                           "❌ Failed to give shards. This could be due to:\n"
                           "• Amount too large (over 2 billion)\n"
                           "• Player balance would overflow\n"
                           "• Database error\n\n"
                           "Please check the logs or try a smaller amount.")
        
    except Exception as e:
        log_exception("give_shards_command", e, update.effective_user.id)
        await safe_send(update.message.reply_text, 
                       "❌ Error giving shards. Please try again.", 
                       parse_mode='HTML')

async def remove_shards_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin command to remove shards from a player"""
    if not bot_instance.is_admin(update.effective_user.id):
        await safe_send(update.message.reply_text, 
                       "❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can remove shards!", 
                       parse_mode='HTML')
        return
    
    try:
        args_valid, error_msg = InputValidator.validate_command_args(context.args, 2)
        if not args_valid:
            await safe_send(update.message.reply_text,
                           f"{error_msg}\n\n"
                           "📝 <b>Usage:</b>\n"
                           "/removeshards &lt;player&gt; &lt;amount&gt; [reason]\n\n"
                           "<b>Examples:</b>\n"
                           "• /removeshards @user 500\n"
                           "• /removeshards 123456789 1000 Rule violation\n"
                           "• /removeshards PlayerName 250 Correction",
                           parse_mode='HTML')
            return
        
        player_identifier = context.args[0]
        try:
            amount = int(context.args[1])
            if amount <= 0:
                raise ValueError("Amount must be positive")
        except ValueError:
            await safe_send(update.message.reply_text, "❌ Invalid amount. Must be a positive number.")
            return
        
        reason = InputValidator.safe_join_args(context.args, 2, "Admin removal")
        
        player = bot_instance.find_player_by_identifier(player_identifier)
        if not player:
            await safe_send(update.message.reply_text, 
                           f"❌ Player '{H(player_identifier)}' not found.\n\n"
                           "💡 Make sure they've used the bot before!")
            return
        
        current_shards, _ = bot_instance.get_player_shards(player['telegram_id'])
        
        success = bot_instance.remove_shards(
            player['telegram_id'], 
            amount, 
            'admin_removal',
            reason,
            update.effective_user.id, 
            reason
        )
        
        if success:
            new_shards, total_earned = bot_instance.get_player_shards(player['telegram_id'])
            actual_removed = current_shards - new_shards
            
            await safe_send(update.message.reply_text,
                           f"✅ <b>SHARDS REMOVED!</b>\n\n"
                           f"👤 <b>Player:</b> {H(player['display_name'])}\n"
                           f"💠 <b>Amount:</b> -{actual_removed:,} shards\n"
                           f"💰 <b>New Balance:</b> {new_shards:,} shards\n"
                           f"📝 <b>Reason:</b> {H(reason)}\n"
                           f"👨‍💼 <b>Removed by:</b> {H(update.effective_user.full_name or 'Admin')}",
                           parse_mode='HTML')
        else:
            await safe_send(update.message.reply_text, "❌ Failed to remove shards. Please try again.")
        
    except Exception as e:
        log_exception("remove_shards_command", e, update.effective_user.id)
        await safe_send(update.message.reply_text, 
                       "❌ Error removing shards. Please try again.", 
                       parse_mode='HTML')

async def distribute_daily_rewards_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manually distribute daily rewards to top 10 users from /dailylb (Admin only)"""
    user_id = update.effective_user.id
    
    if not bot_instance.is_admin(user_id):
        await safe_send(update.message.reply_text, 
                       "❌ <b>Admin Only Command</b>\n\nThis command is restricted to administrators only.", 
                       parse_mode='HTML')
        return
    
    try:
        chase_lb = bot_instance.get_daily_leaderboard('chase', 10)
        guess_lb = bot_instance.get_daily_leaderboard('guess', 10)
        
        if not chase_lb and not guess_lb:
            await safe_send(update.message.reply_text, 
                           "📭 <b>No Leaderboard Data</b>\n\nNo players found in today's daily leaderboards.\n\nUse /dailylb to check current standings.", 
                           parse_mode='HTML')
            return
            
        conn = bot_instance.get_db_connection()
        already_distributed = False
        if conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT chase_rewards_distributed, guess_rewards_distributed FROM daily_leaderboard_status WHERE status_date = CURRENT_DATE"
            )
            status = cursor.fetchone()
            bot_instance.return_db_connection(conn)
            
            if status and (status[0] or status[1]):
                already_distributed = True
        
        if already_distributed:
            await safe_send(update.message.reply_text, 
                           "⚠️ <b>Already Distributed</b>\n\nRewards have already been distributed today!\n\n📊 Use /dailylb to see current standings.", 
                           parse_mode='HTML')
            return
        
        await safe_send(update.message.reply_text, 
                       "⏳ <b>Distributing Daily Rewards...</b>\n\n💠 Processing top 10 players from both Chase and Guess leaderboards...",
                       parse_mode='HTML')
        
        chase_count = 0
        guess_count = 0
        
        if chase_lb:
            chase_count = bot_instance.distribute_daily_leaderboard_rewards('chase')
        
        if guess_lb:
            guess_count = bot_instance.distribute_daily_leaderboard_rewards('guess')
        
        message = f"🎉 <b>DAILY REWARDS DISTRIBUTED!</b>\n\n"
        message += f"🏏 <b>Chase Leaderboard:</b> {chase_count} players rewarded\n"
        message += f"� <b>Guess Leaderboard:</b> {guess_count} players rewarded\n\n"
        message += f"💎 <b>Reward Structure:</b>\n"
        message += f"🥇 1st-3rd: 100 💠 each\n"
        message += f"🥈 4th-6th: 80 💠 each\n"
        message += f"🥉 7th-8th: 60 💠 each\n"
        message += f"🏅 9th: 40 💠 • 10th: 20 💠\n\n"
        message += f"📊 Use /dailylb to see the leaderboard!"
        
        await safe_send(update.message.reply_text, message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Error in distribute_daily_rewards_command: {e}")
        await safe_send(update.message.reply_text, 
                       "❌ <b>Distribution Failed</b>\n\nAn error occurred while distributing rewards. Please try again later.", 
                       parse_mode='HTML')
        log_exception("distribute_daily_rewards_command", e, update.effective_user.id)
        await safe_send(update.message.reply_text, 
                       "❌ Error distributing daily rewards. Please try again.", 
                       parse_mode='HTML')

async def reset_daily_leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset daily leaderboard data (Admin only)"""
    user_id = update.effective_user.id
    
    if not bot_instance.is_admin(user_id):
        await safe_send(update.message.reply_text, 
                       "❌ This command is only available to admins.", 
                       parse_mode='HTML')
        return
    
    try:
        args = context.args
        if not args or args[0].upper() != "CONFIRM":
            await safe_send(update.message.reply_text, 
                           "⚠️ <b>RESET DAILY LEADERBOARD</b>\n\n"
                           "This will clear all today's leaderboard data!\n"
                           "Use: <code>/resetdailylb CONFIRM</code> to proceed.",
                           parse_mode='HTML')
            return
        
        success = bot_instance.reset_daily_leaderboards()
        
        if success:
            await safe_send(update.message.reply_text, 
                           "✅ <b>Daily leaderboard reset successfully!</b>\n\n"
                           "🗑️ All today's leaderboard entries have been cleared.\n"
                           "🎮 Players can start fresh with new games.",
                           parse_mode='HTML')
        else:
            await safe_send(update.message.reply_text, 
                           "❌ Failed to reset daily leaderboard. Please try again.", 
                           parse_mode='HTML')
        
    except Exception as e:
        log_exception("reset_daily_leaderboard_command", e, update.effective_user.id)
        await safe_send(update.message.reply_text, 
                       "❌ Error resetting daily leaderboard. Please try again.", 
                       parse_mode='HTML')

async def daily_leaderboard_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show daily leaderboard statistics (Admin only)"""
    user_id = update.effective_user.id
    
    if not bot_instance.is_admin(user_id):
        await safe_send(update.message.reply_text, 
                       "❌ This command is only available to admins.", 
                       parse_mode='HTML')
        return
    
    try:
        from datetime import datetime
        today = datetime.now().strftime("%B %d, %Y")
        
        chase_lb = bot_instance.get_daily_leaderboard('chase', 50)  # Get more for stats
        guess_lb = bot_instance.get_daily_leaderboard('guess', 50)
        
        total_chase_players = len(chase_lb)
        total_guess_players = len(guess_lb)
        total_chase_games = sum(p.get('chase_games_played', 0) for p in chase_lb)
        total_guess_games = sum(p.get('guess_games_played', 0) for p in guess_lb)
        total_chase_score = sum(p.get('chase_total_score', 0) for p in chase_lb)
        total_guess_score = sum(p.get('guess_total_score', 0) for p in guess_lb)
        
        top_chase = chase_lb[0] if chase_lb else None
        top_guess = guess_lb[0] if guess_lb else None
        
        message = f"📊 <b>DAILY LEADERBOARD STATISTICS</b>\n"
        message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        message += f"📅 <b>Date:</b> {today}\n\n"
        
        message += f"🏏 <b>CHASE GAME STATS:</b>\n"
        message += f"👥 Players: {total_chase_players}\n"
        message += f"🎮 Total Games: {total_chase_games:,}\n"
        message += f"📊 Total Score: {total_chase_score:,}\n"
        if top_chase:
            message += f"👑 Leader: {H(top_chase['player_name'])} ({top_chase.get('chase_total_score', 0):,})\n"
        message += f"\n"
        
        message += f"🎲 <b>GUESS GAME STATS:</b>\n"
        message += f"👥 Players: {total_guess_players}\n"
        message += f"🎮 Total Games: {total_guess_games:,}\n"
        message += f"📊 Total Score: {total_guess_score:,}\n"
        if top_guess:
            message += f"👑 Leader: {H(top_guess['player_name'])} ({top_guess.get('guess_total_score', 0):,})\n"
        message += f"\n"
        
        message += f"🎯 <b>OVERALL ACTIVITY:</b>\n"
        total_active_players = len(set([p['player_id'] for p in chase_lb] + [p['player_id'] for p in guess_lb]))
        message += f"👥 Unique Active Players: {total_active_players}\n"
        message += f"🎮 Total Games Played: {total_chase_games + total_guess_games:,}\n"
        message += f"💠 Potential Rewards: {min(10, total_chase_players) * 680 + min(10, total_guess_players) * 680:,} shards\n\n"
        
        message += f"⏰ <b>Next reward distribution:</b> 10:00 PM\n"
        message += f"🏆 Use /ddrlb to trigger manually"
        
        await safe_send(update.message.reply_text, message, parse_mode='HTML')
        
    except Exception as e:
        log_exception("daily_leaderboard_stats_command", e, update.effective_user.id)
        await safe_send(update.message.reply_text, 
                       "❌ Error retrieving daily leaderboard stats. Please try again.", 
                       parse_mode='HTML')

async def confirm_achievement_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Confirm special achievement (Admin only)"""
    user_id = update.effective_user.id
    
    if not bot_instance.is_admin(user_id):
        await safe_send(update.message.reply_text, 
                       "❌ This command is only available to admins.", 
                       parse_mode='HTML')
        return
    
    try:
        args = context.args
        if not args:
            await safe_send(update.message.reply_text, 
                           "📋 <b>PENDING ACHIEVEMENT CONFIRMATIONS</b>\n\n"
                           "Use: <code>/confirmachievement [confirmation_id] [optional_notes]</code>\n\n"
                           "Use /listpending to see all pending confirmations.",
                           parse_mode='HTML')
            return
        
        try:
            confirmation_id = int(args[0])
            notes = ' '.join(args[1:]) if len(args) > 1 else None
        except ValueError:
            await safe_send(update.message.reply_text, 
                           "❌ Invalid confirmation ID. Please provide a valid number.", 
                           parse_mode='HTML')
            return
        
        success = bot_instance.confirm_achievement_admin(confirmation_id, user_id, notes)
        
        if success:
            await safe_send(update.message.reply_text, 
                           f"✅ <b>Achievement confirmed successfully!</b>\n\n"
                           f"🏆 Confirmation ID: {confirmation_id}\n"
                           f"👨‍💼 Confirmed by: {H(update.effective_user.full_name or 'Admin')}\n"
                           f"📝 Notes: {H(notes) if notes else 'None'}\n\n"
                           f"The player has been awarded their achievement and rewards!",
                           parse_mode='HTML')
        else:
            await safe_send(update.message.reply_text, 
                           f"❌ Failed to confirm achievement.\n\n"
                           f"Possible reasons:\n"
                           f"• Invalid confirmation ID\n"
                           f"• Already processed\n"
                           f"• Database error",
                           parse_mode='HTML')
        
    except Exception as e:
        log_exception("confirm_achievement_command", e, update.effective_user.id)
        await safe_send(update.message.reply_text, 
                       "❌ Error confirming achievement. Please try again.", 
                       parse_mode='HTML')

async def list_pending_confirmations_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List pending achievement confirmations (Admin only)"""
    user_id = update.effective_user.id
    
    if not bot_instance.is_admin(user_id):
        await safe_send(update.message.reply_text, 
                       "❌ This command is only available to admins.", 
                       parse_mode='HTML')
        return
    
    try:
        pending = bot_instance.get_pending_achievement_confirmations()
        
        if not pending:
            message = f"📋 <b>PENDING CONFIRMATIONS</b>\n"
            message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            message += f"✅ No pending achievement confirmations!\n"
            message += f"🎉 All special achievements are up to date."
        else:
            message = f"📋 <b>PENDING ACHIEVEMENT CONFIRMATIONS</b>\n"
            message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            for conf in pending:
                details = json.loads(conf['achievement_details']) if conf['achievement_details'] else {}
                
                message += f"🆔 <b>ID:</b> {conf['confirmation_id']}\n"
                message += f"🧠⚡ <b>Achievement:</b> {conf['achievement_name']} ⚡🧠\n"
                message += f"👤 <b>Player:</b> {H(conf['player_name'])}\n"
                message += f"📞 <b>Telegram ID:</b> <code>{conf['player_telegram_id']}</code>\n"
                message += f"🕒 <b>Requested:</b> {conf['created_at'].strftime('%Y-%m-%d %H:%M')}\n"
                message += f"💠 <b>Reward:</b> {details.get('reward', '10,000 shards + title')}\n\n"
                message += f"<code>/confirmachievement {conf['confirmation_id']}</code>\n"
                message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        await safe_send(update.message.reply_text, message, parse_mode='HTML')
        
    except Exception as e:
        log_exception("list_pending_confirmations_command", e, update.effective_user.id)
        await safe_send(update.message.reply_text, 
                       "❌ Error retrieving pending confirmations. Please try again.", 
                       parse_mode='HTML')

# ====================================
# ====================================

@check_banned
async def nightmare_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start or continue nightmare mode game"""
    try:
        user_id = update.effective_user.id
        user = update.effective_user
        player_name = user.first_name or "Unknown"
        
        bot_instance.create_or_update_player(user_id, user.username or "", player_name)
        
        active_game = bot_instance.get_nightmare_game(user_id)
        
        if active_game:
            attempts_left = active_game['max_attempts'] - active_game['attempts_used']
            
            message = f"🧩 <b>Nightmare Mode</b>\n"
            message += f"━━━━━━━━━━━━━━━━━━━━\n\n"
            message += f"🎯 <b>Range:</b> 1 to 10,000\n"
            message += f"🎯 <b>Attempts left:</b> {attempts_left}\n\n"
            message += f"💡 <b>Your Hint:</b>\n"
            message += f"{active_game['encoded_hint']}\n\n"
            message += f"🎮 <b>Just type your guess!</b>"
            
            keyboard = [
                [InlineKeyboardButton("❓ Need Help?", callback_data="nightmare_help")],
                [InlineKeyboardButton("❌ Quit", callback_data=f"nightmare_quit_{active_game['game_id']}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
        else:
            game_data = bot_instance.start_nightmare_game(user_id, player_name)
            
            if 'error' in game_data:
                await safe_send(update.message.reply_text, 
                               f"❌ {game_data['error']}", 
                               parse_mode='HTML')
                return
            
            message = f"💀 <b>NIGHTMARE ARENA</b> 💀\n"
            message += f"━━━━━━━━━━━━━━━━━━━━\n"
            message += f"⚡ <b>ULTIMATE CHALLENGE MODE</b> ⚡\n\n"
            message += f"🚨 <b>DANGER ZONE ACTIVATED</b> 🚨\n"
            message += f"🔥 High Risk • High Reward\n\n"
            message += f"━━━━━━━━━━━━━━━━━━━━\n"
            message += f"📊 <b>NIGHTMARE STATS</b>\n"
            message += f"🎯 Number Range: 1-10,000\n"
            message += f"💀 Lives: 3 attempts only\n"
            message += f"👑 Jackpot Prize: 10,000 shards\n"
            message += f"━━━━━━━━━━━━━━━━━━━━\n\n"
            message += f"🧩 <b>YOUR HINT:</b>\n"
            
            hint = game_data.get('encoded_hint', 'No hint available')
            message += f"{hint}\n\n"
            message += f"⚠️ <b>Challenge:</b> The number changes slightly after each wrong guess!\n"
            message += f"💡 <b>Strategy:</b> Use the hint to narrow down the range, then guess smart!\n\n"
            message += f"🎮 <b>Type your first guess now!</b>"
            
            keyboard = [
                [InlineKeyboardButton("❓ How to Play", callback_data="nightmare_help")],
                [InlineKeyboardButton("📊 Stats", callback_data="nightmare_stats")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
        
        await safe_send(update.message.reply_text, message, parse_mode='HTML', reply_markup=reply_markup)
        
    except Exception as e:
        log_exception("nightmare_command", e, update.effective_user.id)
        await safe_send(update.message.reply_text, 
                       "❌ Error starting nightmare mode. Please try again.", 
                       parse_mode='HTML')

async def nightmare_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle nightmare mode button callbacks"""
    query = update.callback_query
    
    try:
        data = query.data
        user_id = query.from_user.id
        
        if data.startswith('nightmare_quit_') or data == 'nightmare_cancel':
            active_game = bot_instance.get_nightmare_game(user_id)
            if not active_game:
                await query.answer("⚠️ You don't have an active nightmare game!", show_alert=True)
                return
        
        if data == 'nightmare_retry':
            pass  # Allow retry for anyone
        
        await query.answer()
        
        if data.startswith('nightmare_decode_'):
            message = f"🎉 <b>Good News!</b>\n"
            message += f"━━━━━━━━━━━━━━━━━━\n\n"
            message += f"🎯 <b>Your hint is already clear and readable!</b>\n"
            message += f"✨ <b>No decoding needed anymore</b>\n\n"
            message += f"🧠 <b>Just use the mathematical clues to find the number</b>\n"
            message += f"🎮 <b>Type your guess when ready!</b>"
            
            await safe_edit_message(query, message, parse_mode='HTML')
            
        elif data == 'nightmare_stats':
            stats = bot_instance.get_nightmare_stats(user_id)
            
            if not stats or stats.get('nightmare_games_played', 0) == 0:
                message = f"💀 <b>NIGHTMARE MODE STATS</b>\n"
                message += f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                message += f"🎮 You haven't played Nightmare Mode yet!\n"
                message += f"🌟 Ready to face the ultimate challenge?\n\n"
                message += f"Use /nightmare to begin your first attempt!"
                win_rate = (stats['nightmare_games_won'] / stats['nightmare_games_played'] * 100) if stats['nightmare_games_played'] > 0 else 0
                avg_attempts = (stats['nightmare_total_attempts'] / stats['nightmare_games_won']) if stats['nightmare_games_won'] > 0 else 0
                
                message = f"💀 <b>YOUR NIGHTMARE STATS</b> 💀\n"
                message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                message += f"🎮 <b>Games Played:</b> {stats['nightmare_games_played']:,}\n"
                message += f"🏆 <b>Games Won:</b> {stats['nightmare_games_won']:,}\n"
                message += f"📈 <b>Win Rate:</b> {win_rate:.1f}%\n"
                message += f"🎯 <b>Best Performance:</b> {stats['nightmare_best_attempts']} attempts\n"
                message += f"📊 <b>Avg Attempts (Wins):</b> {avg_attempts:.1f}\n"
                
                if stats.get('has_shard_mastermind', False):
                    message += f"🧠⚡ <b>SHARD MASTERMIND ACHIEVED!</b> ⚡🧠\n"
                else:
                    message += f"💫 <b>Challenge:</b> Conquer nightmare for exclusive title!\n"
                
                if stats.get('nightmare_last_played'):
                    message += f"⏰ <b>Last Played:</b> {stats['nightmare_last_played'].strftime('%Y-%m-%d')}"
            
            await safe_edit_message(query, message, parse_mode='HTML')
            
        elif data == 'nightmare_leaderboard':
            leaderboard = bot_instance.get_nightmare_leaderboard(10)
            
            message = f"💀 <b>NIGHTMARE MODE LEGENDS</b> 💀\n"
            message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            if not leaderboard:
                message += f"👻 <b>No legends yet...</b>\n"
                message += f"🌟 Be the first to conquer Nightmare Mode!\n\n"
                message += f"🏆 Win to claim your place in history!"
            else:
                rank_emojis = {1: "🥇", 2: "🥈", 3: "🥉"}
                
                for i, player in enumerate(leaderboard, 1):
                    rank_display = rank_emojis.get(i, f"{i}.")
                    name = H(player['player_name'][:12])
                    won = player['nightmare_games_won']
                    played = player['nightmare_games_played']
                    best_attempts = player['nightmare_best_attempts'] or 'N/A'
                    win_pct = player['win_percentage']
                    mastermind = "🧠⚡" if player.get('has_shard_mastermind', False) else ""
                    
                    message += f"{rank_display} <b>{name}</b> {mastermind}\n"
                    message += f"   ┗ <b>Won:</b> {won}/{played} ({win_pct}%) | <b>Best:</b> {best_attempts}\n\n"
            
            message += f"💡 <b>Ranking by wins, then best attempts, then total games</b>"
            
            await safe_edit_message(query, message, parse_mode='HTML')
            
        elif data.startswith('nightmare_quit_'):
            game_id = int(data.split('_')[2])
            
            keyboard = [
                [InlineKeyboardButton("✅ Yes, Quit", callback_data=f"nightmare_quit_confirm_{game_id}")],
                [InlineKeyboardButton("❌ Cancel", callback_data="nightmare_cancel")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            message = f"⚠️ <b>QUIT NIGHTMARE GAME?</b>\n\n"
            message += f"🚨 This will count as a loss!\n"
            message += f"💀 Are you sure you want to quit?"
            
            await safe_edit_message(query, message, parse_mode='HTML', reply_markup=reply_markup)
            
        elif data.startswith('nightmare_quit_confirm_'):
            game_id = int(data.split('_')[3])
            
            conn = bot_instance.get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE nightmare_games 
                        SET is_completed = TRUE, is_won = FALSE, completed_at = CURRENT_TIMESTAMP
                        WHERE game_id = %s AND player_telegram_id = %s
                    """, (game_id, user_id))
                    
                    cursor.execute("""
                        UPDATE players 
                        SET nightmare_games_played = nightmare_games_played + 1,
                            nightmare_total_attempts = nightmare_total_attempts + (
                                SELECT attempts_used FROM nightmare_games WHERE game_id = %s
                            )
                        WHERE telegram_id = %s
                    """, (game_id, user_id))
                    
                    conn.commit()
                    
                    message = f"💀 <b>NIGHTMARE ABANDONED</b>\n\n"
                    message += f"😔 You have quit the nightmare challenge.\n"
                    message += f"🎮 Use /nightmare to start a new attempt when ready!"
                    
                    await safe_edit_message(query, message, parse_mode='HTML')
                    
                except Exception as e:
                    logger.error(f"Error quitting nightmare game: {e}")
                    try:
                        await safe_edit_message(query, "❌ Error quitting game. Please try /nightmare again.", parse_mode='HTML')
                    except:
                        pass
                finally:
                    bot_instance.return_db_connection(conn)
            
        elif data == 'nightmare_cancel':
            await safe_edit_message(query, "🎮 Nightmare mode continues! Send your guess.", parse_mode='HTML')
            
        elif data == 'nightmare_help':
            message = f"❓ <b>How to Play Nightmare Mode</b>\n"
            message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            message += f"🎯 <b>Goal:</b> Find the secret number (1 to 10,000)\n\n"
            message += f"🎮 <b>How it works:</b>\n"
            message += f"1️⃣ You get a clear mathematical hint\n"
            message += f"2️⃣ You have exactly 3 attempts\n"
            message += f"3️⃣ After each wrong guess, the number shifts slightly\n"
            message += f"4️⃣ You get a new hint after each shift\n\n"
            message += f"💡 <b>Strategy Tips:</b>\n"
            message += f"• Read your hint carefully - it gives real clues!\n"
            message += f"• Use logic, not random guessing\n"
            message += f"• Close guesses cause smaller shifts\n"
            message += f"• Far guesses cause bigger shifts\n\n"
            message += f"🏆 <b>Win:</b> 10,000 shards + special title!\n"
            message += f"🎮 <b>Ready? Just type a number!</b>"
            
            await safe_edit_message(query, message, parse_mode='HTML')
        
    except BadRequest as e:
        logger.warning(f"BadRequest in nightmare_callback: {e}")
    except Exception as e:
        logger.error(f"Error in nightmare_callback: {e}")
        try:
            await query.answer("❌ Error processing request.", show_alert=True)
        except:
            pass  # Fail silently if we can't even send error alert

async def handle_nightmare_guess(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Handle nightmare mode guesses. Returns True if message was a nightmare guess."""
    try:
        if update.edited_message:
            return False
        
        user_id = update.effective_user.id
        text = update.message.text.strip()
        
        active_game = bot_instance.get_nightmare_game(user_id)
        if not active_game:
            return False  # Not a nightmare guess
        
        try:
            guess = int(text)
            if guess < 1 or guess > 10000:
                await safe_send(update.message.reply_text,
                               "💀 <b>Invalid range!</b>\n"
                               "🎯 Please guess between 1 and 10,000", 
                               parse_mode='HTML')
                return True
        except ValueError:
            return False  # Not a number, let other handlers process
        
        result = bot_instance.make_nightmare_guess(active_game['game_id'], guess)
        
        if 'error' in result:
            await safe_send(update.message.reply_text, 
                           f"❌ {result['error']}", 
                           parse_mode='HTML')
            return True
        
        if result['is_correct']:
            message = f"🎉 <b>You Won Nightmare Mode!</b> 🎉\n"
            message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            message += f"🏆 <b>Incredible Achievement!</b>\n\n"
            message += f"✅ <b>Your guess {guess:,} was correct!</b>\n"
            message += f"💠 <b>Earned:</b> 10,000 shards\n"
            message += f"🏅 <b>New title:</b> Shard Mastermind\n"
            message += f"🎯 <b>Attempts used:</b> {active_game['max_attempts'] - result['attempts_remaining']}/3\n\n"
            message += f"🌟 <b>You've mastered the ultimate challenge!</b>"
            
            keyboard = [
                [InlineKeyboardButton("🎮 Play Again", callback_data="nightmare_retry")],
                [InlineKeyboardButton("📊 My Stats", callback_data="nightmare_stats")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await safe_send(update.message.reply_text, message, parse_mode='HTML', reply_markup=reply_markup)
            
            try:
                bot_instance.update_daily_leaderboard(user_id, 'nightmare', 10000, True, 3)  # Max score for nightmare
            except:
                pass  # Don't fail on daily leaderboard update
                
        elif result['game_over']:
            message = f"💀 <b>Game Over</b> 💀\n"
            message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            message += f"😔 <b>You ran out of attempts</b>\n\n"
            message += f"❌ <b>Your guess:</b> {guess:,}\n"
            message += f"🎯 <b>The final number was:</b> {result['current_number']:,}\n"
            message += f"❌ <b>Used all 3 attempts</b>\n\n"
            message += f"💡 <b>Tip:</b> The shifting made it extra challenging!\n"
            message += f"🎮 <b>Want to try again?</b>"
            
            keyboard = [
                [InlineKeyboardButton("📊 My Stats", callback_data="nightmare_stats")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await safe_send(update.message.reply_text, message, parse_mode='HTML', reply_markup=reply_markup)
            
        else:
            attempts_left = result['attempts_remaining']
            
            message = f"❌ <b>Wrong Guess</b>\n"
            message += f"━━━━━━━━━━━━━━━━━━━━\n\n"
            message += f"🎯 <b>Your guess:</b> {guess:,}\n"
            
            if result.get('shift_info'):
                message += f"{result['shift_info']}\n"
            else:
                message += f"🌪️ <b>Number has shifted!</b>\n"
                
            message += f"🎯 <b>Attempts left:</b> {attempts_left}\n\n"
            
            if result.get('new_hint'):
                message += f"💡 <b>New Hint:</b>\n{result['new_hint']}\n\n"
            
            if attempts_left == 2:
                message += f"🤔 <b>Think carefully about the hint...</b>\n"
            elif attempts_left == 1:
                message += f"⚠️ <b>Last chance! Make it count!</b>\n"
            
            message += f"✏️ <b>Type your next guess!</b>"
            
            keyboard = [
                [InlineKeyboardButton("❓ Need Help?", callback_data="nightmare_help")],
                [InlineKeyboardButton("❌ Quit", callback_data=f"nightmare_quit_{active_game['game_id']}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await safe_send(update.message.reply_text, message, parse_mode='HTML', reply_markup=reply_markup)
        
        return True  # Message was handled as nightmare guess
        
    except Exception as e:
        log_exception("handle_nightmare_guess", e, update.effective_user.id)
        return False

@check_banned
async def dailylb_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show daily leaderboard with toggle buttons for Chase/Guess"""
    try:
        game_type = 'chase'
        
        chase_lb = bot_instance.get_daily_leaderboard('chase', 20)
        guess_lb = bot_instance.get_daily_leaderboard('guess', 20)
        
        keyboard = [
            [
                InlineKeyboardButton("🏏 Chase", callback_data="dailylb_chase"),
                InlineKeyboardButton("🎲 Guess", callback_data="dailylb_guess")
            ],
            [InlineKeyboardButton("🔄 Refresh", callback_data="dailylb_refresh")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = generate_daily_leaderboard_message('chase', chase_lb, guess_lb)
        
        await safe_send(update.message.reply_text, message, parse_mode='HTML', reply_markup=reply_markup)
        
    except Exception as e:
        log_exception("dailylb_command", e, update.effective_user.id)
        await safe_send(update.message.reply_text, 
                       "❌ Error loading daily leaderboard. Please try again.", 
                       parse_mode='HTML')

async def dailylb_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle daily leaderboard button callbacks"""
    query = update.callback_query
    await query.answer()
    
    try:
        data = query.data
        
        if data.startswith('dailylb_'):
            action = data.split('_')[1]
            
            chase_lb = bot_instance.get_daily_leaderboard('chase', 10)
            guess_lb = bot_instance.get_daily_leaderboard('guess', 10)
            
            keyboard = [
                [
                    InlineKeyboardButton("🏏 Chase", callback_data="dailylb_chase"),
                    InlineKeyboardButton("🎲 Guess", callback_data="dailylb_guess")
                ],
                [InlineKeyboardButton("🔄 Refresh", callback_data="dailylb_refresh")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            if action == 'chase' or action == 'refresh':
                message = generate_daily_leaderboard_message('chase', chase_lb, guess_lb)
            elif action == 'guess':
                message = generate_daily_leaderboard_message('guess', chase_lb, guess_lb)
            else:
                return
            
            success = await safe_edit_message(query, message, parse_mode='HTML', reply_markup=reply_markup)
            if not success:
                await safe_send(query.message.reply_text, message, parse_mode='HTML', reply_markup=reply_markup)
        
    except BadRequest as e:
        logger.warning(f"BadRequest in dailylb_callback: {e}")
    except Exception as e:
        logger.error(f"Error in dailylb_callback: {e}")
        try:
            await query.answer("❌ Error processing request.", show_alert=True)
        except:
            pass

def generate_daily_leaderboard_message(game_type: str, chase_lb: list, guess_lb: list) -> str:
    """Generate daily leaderboard message for specified game type"""
    from datetime import datetime
    today = datetime.now().strftime("%B %d, %Y")
    
    if game_type == 'chase':
        leaderboard = chase_lb
        title = "🏏 DAILY CHASE LEADERBOARD"
        empty_msg = "🏏 No chase games played today!\n🎮 Be the first to play with /chase"
        
        rewards_msg = "🏆 <b>Daily Rewards (Manual Distribution):</b>\n"
        rewards_msg += "🥇 1st-3rd: <b>100 💠 each</b>\n"
        rewards_msg += "🏅 4th-6th: <b>80 💠 each</b>\n"
        rewards_msg += "🎖️ 7th-8th: <b>60 💠 each</b>\n"
        rewards_msg += "🏵️ 9th: <b>40 💠</b> • 🎗️ 10th: <b>20 💠</b>\n\n"
        
    else:  # guess
        leaderboard = guess_lb
        title = "🎲 DAILY GUESS LEADERBOARD"  
        empty_msg = "🎲 No guess games played today!\n🎮 Be the first to play with /guess"
        
        rewards_msg = "🏆 <b>Daily Rewards (Manual Distribution):</b>\n"
        rewards_msg += "🥇 1st-3rd: <b>100 💠 each</b>\n"
        rewards_msg += "🏅 4th-6th: <b>80 💠 each</b>\n"
        rewards_msg += "🎖️ 7th-8th: <b>60 💠 each</b>\n"
        rewards_msg += "🏵️ 9th: <b>40 💠</b> • 🎗️ 10th: <b>20 💠</b>\n\n"
    
    message = f"{title} 💠\n"
    message += f"━━━━━━━━━━━━━━━━━━━━\n"
    message += f"📅 <b>{today}</b>\n\n"
    
    if not leaderboard:
        message += empty_msg
    else:
        rank_emojis = {1: "🥇", 2: "🥈", 3: "🥉"}
        
        for i, player in enumerate(leaderboard, 1):
            rank_display = rank_emojis.get(i, f"{i}.")
            name = H(player['player_name'][:15])
            
            if game_type == 'chase':
                games = player.get('games_played', 0)
                best_score = player.get('best_score', 0)
                total_score = player.get('total_score', 0)
                best_level = player.get('level_completed', 1)
                
                message += f"{rank_display} <b>{name}</b>\n"
                message += f"   🏏 <b>Level {best_level}</b> • <b>Best:</b> {best_score:,} • <b>Games:</b> {games}\n\n"
                
            else:  # guess
                games = player.get('games_played', 0)
                total_score = player.get('total_score', 0)
                best_score = player.get('best_score', 0)
                won = player.get('games_won', 0)
                win_rate = round((won/games)*100, 1) if games > 0 else 0
                
                message += f"{rank_display} <b>{name}</b>\n"
                message += f"   🎯 <b>Score:</b> {total_score:,} • <b>Win Rate:</b> {win_rate}% ({won}/{games})\n\n"
    
    message += "━━━━━━━━━━━━━━━━━━━━\n"
    message += rewards_msg
    
    chase_count = len(chase_lb)
    guess_count = len(guess_lb) 
    message += f"📊 <b>Today's Activity:</b>\n"
    message += f"🏏 Chase Players: {chase_count} • 🎲 Guess Players: {guess_count}\n\n"
    if game_type == 'chase':
        message += f"� <b>Ranked by highest level, then best score!</b>\n"
    else:
        message += f"🎯 <b>Ranked by total daily score!</b>\n"
    message += f"⏰ <i>Rewards distributed manually by admins</i>"
    
    return message

async def distribute_daily_leaderboard_rewards():
    """Distribute rewards for daily leaderboards at 10 PM"""
    try:
        logger.info("Starting daily leaderboard reward distribution...")
        
        rewards = [100, 100, 100, 80, 80, 80, 60, 60, 40, 20]
        
        chase_lb = bot_instance.get_daily_leaderboard('chase', 10)
        guess_lb = bot_instance.get_daily_leaderboard('guess', 10)
        
        total_distributed = 0
        notifications = []
        
        for i, player in enumerate(chase_lb[:10]):  # Top 10 only
            reward_amount = rewards[i]
            player_id = player.get('player_telegram_id') or player.get('player_id')
            player_name = player.get('player_name', 'Unknown Player')
            rank = i + 1
            
            if not player_id:
                logger.error(f"No player ID found for chase rank {rank}: {player}")
                continue
            
            success = bot_instance.award_shards(
                player_id, 
                reward_amount, 
                f"Daily Chase Leaderboard - Rank #{rank}"
            )
            
            if success:
                total_distributed += reward_amount
                notifications.append({
                    'player_id': player_id,
                    'player_name': player_name,
                    'game_type': 'Chase 🏏',
                    'rank': rank,
                    'reward': reward_amount,
                    'score': player.get('total_score', 0)  # Fixed field name
                })
                logger.info(f"Awarded {reward_amount} shards to {player_name} (Chase Rank {rank})")
            else:
                logger.error(f"Failed to award {reward_amount} shards to {player_name} (Chase Rank {rank})")
        
        for i, player in enumerate(guess_lb[:10]):  # Top 10 only
            reward_amount = rewards[i]
            player_id = player.get('player_telegram_id') or player.get('player_id')
            player_name = player.get('player_name', 'Unknown Player')
            rank = i + 1
            
            if not player_id:
                logger.error(f"No player ID found for guess rank {rank}: {player}")
                continue
            
            success = bot_instance.award_shards(
                player_id,
                reward_amount,
                f"Daily Guess Leaderboard - Rank #{rank}"
            )
            
            if success:
                total_distributed += reward_amount
                notifications.append({
                    'player_id': player_id,
                    'player_name': player_name,
                    'game_type': 'Guess 🎲',
                    'rank': rank,
                    'reward': reward_amount,
                    'score': player.get('total_score', 0)  # Fixed field name
                })
                logger.info(f"Awarded {reward_amount} shards to {player_name} (Guess Rank {rank})")
            else:
                logger.error(f"Failed to award {reward_amount} shards to {player_name} (Guess Rank {rank})")
        
        from datetime import datetime
        today = datetime.now().strftime("%B %d, %Y")
        
        for notification in notifications:
            try:
                message = f"🏆 <b>DAILY LEADERBOARD REWARDS!</b> 💠\n\n"
                message += f"🎉 Congratulations! You ranked #{notification['rank']} in today's {notification['game_type']} leaderboard!\n\n"
                message += f"📅 <b>Date:</b> {today}\n"
                message += f"🏅 <b>Rank:</b> #{notification['rank']}\n"
                message += f"🎮 <b>Game:</b> {notification['game_type']}\n"
                message += f"📊 <b>Score:</b> {notification['score']:,}\n"
                message += f"💠 <b>Reward:</b> {notification['reward']} Shards\n\n"
                message += f"🎯 Keep playing to compete for tomorrow's rewards!\n"
                message += f"Use /dailylb to view current standings."
                
                await bot_instance.application.bot.send_message(
                    chat_id=notification['player_id'],
                    text=message,
                    parse_mode='HTML'
                )
                
            except Exception as e:
                logger.error(f"Failed to send reward notification to {notification['player_name']}: {e}")
        
        chase_winners = len([n for n in notifications if 'Chase' in n['game_type']])
        guess_winners = len([n for n in notifications if 'Guess' in n['game_type']])
        
        logger.info(f"Daily leaderboard rewards distributed successfully!")
        logger.info(f"Total shards distributed: {total_distributed}")
        logger.info(f"Chase winners: {chase_winners}, Guess winners: {guess_winners}")
        logger.info(f"Total notifications sent: {len(notifications)}")
        
        await bot_instance.send_admin_log(
            'leaderboard',
            f"Daily rewards distributed | Total: {total_distributed:,} shards | Chase winners: {chase_winners} | Guess winners: {guess_winners}",
            None,
            "System"
        )
        
    except Exception as e:
        logger.error(f"Error distributing daily leaderboard rewards: {e}")

async def schedule_daily_rewards():
    """Daily reward scheduler (DISABLED - Manual rewards only)"""
    logger.info("Automatic daily reward distribution is disabled")
    logger.info("Use /ddrlb command to manually distribute rewards to top 10 /dailylb users")
    
    import asyncio
    while True:
        await asyncio.sleep(86400)  # Sleep for 24 hours and do nothing

# ====================================
# ====================================

bot_instance = ArenaOfChampionsBot()

@check_banned
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    
    username = H(user.username or "")
    display_name = H(user.full_name or user.first_name or f"User{user.id}")
    raw_display_name = user.full_name or user.first_name or f"User{user.id}"  # For DB storage
    
    success, is_new_user = bot_instance.create_or_update_player(user.id, user.username or "", raw_display_name)
    
    if success:
        logger.info(f"User {'registered' if is_new_user else 'updated'}: {raw_display_name} (@{user.username or ''}) - ID: {user.id}")
    
    if is_new_user:
        welcome_message = f"""
╭─────────────────────╮
│   🏆 ARENA OF CHAMPIONS 🎮   │
╰─────────────────────╯

✨ <b>Welcome, {display_name}!</b> ✨

━━━━━━━━━━━━━━━━━━━━━━━━
🎉 <b>REGISTRATION COMPLETE</b> 🎊
━━━━━━━━━━━━━━━━━━━━━━━━

🎯 Ready to track your gaming achievements!
🏆 Earn rewards by playing amazing games
💠 Collect shards and climb leaderboards

━━━━━━━━━━━━━━━━━━━━━━━━
🎮 <b>FEATURED GAMES</b> 🎮
━━━━━━━━━━━━━━━━━━━━━━━━
🏏 <b>/chase</b> - Cricket Run Chase
🎯 <b>/guess</b> - Number Guessing  
🌙 <b>/nightmare</b> - Ultimate Challenge

━━━━━━━━━━━━━━━━━━━━━━━━
⚡ <b>QUICK START</b> ⚡
━━━━━━━━━━━━━━━━━━━━━━━━
🏆 <b>/achievements</b> - Your awards
👤 <b>/profile</b> - Complete stats
💠 <b>/balance</b> - Check shards
📚 <b>/help</b> - Full guide

🚀 <b>Ready to dominate the gaming arena?</b> 🚀
"""
    else:
        welcome_message = f"""
╭─────────────────────╮
│   🏆 ARENA OF CHAMPIONS 🎮   │
╰─────────────────────╯

🌟 <b>Welcome back, {display_name}!</b> 🎮

━━━━━━━━━━━━━━━━━━━━━━━━
🎉 <b>CHAMPION RETURNS</b> 🎉
━━━━━━━━━━━━━━━━━━━━━━━━

🏆 Your achievements are safe & ready
💠 Your shards are waiting to be spent
📈 New challenges await your skills

━━━━━━━━━━━━━━━━━━━━━━━━
🎮 <b>JUMP RIGHT IN</b> 🎮
━━━━━━━━━━━━━━━━━━━━━━━━
🏏 <b>/chase</b> - Cricket Action
🎯 <b>/guess</b> - Mind Games
💀 <b>/nightmare</b> - Ultimate Test

━━━━━━━━━━━━━━━━━━━━━━━━
⚡ <b>PLAYER HUB</b> ⚡
━━━━━━━━━━━━━━━━━━━━━━━━
👤 <b>/profile</b> - Your stats
🏆 <b>/achievements</b> - Your glory
💠 <b>/shardlb</b> - Rich list
📊 <b>/sleaderboard</b> - Top players

🔥 <b>Time to reclaim your throne!</b> 🔥
"""
    
    welcome_message += """
━━━━━━━━━━━━━━━━━━━━━━━━
❤️ <b>Crafted with passion for gaming champions</b> ❤️
━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    await update.message.reply_text(welcome_message, parse_mode='HTML')

@check_banned
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show paginated help information based on user role."""
    user_id = update.effective_user.id
    is_super_admin = bot_instance.is_super_admin(user_id)
    is_admin = bot_instance.is_admin(user_id)
    
    keyboard = [
        [InlineKeyboardButton("🎮 Games", callback_data="help_games"),
         InlineKeyboardButton("💠 Shards", callback_data="help_shards")],
        [InlineKeyboardButton("🏆 Achievements", callback_data="help_achievements"),
         InlineKeyboardButton("🎯 Auctions", callback_data="help_auctions")],
        [InlineKeyboardButton("📊 Stats", callback_data="help_stats")]
    ]
    
    if is_admin:
        keyboard.append([InlineKeyboardButton("👨‍💼 Admin Commands", callback_data="help_admin")])
    
    if is_super_admin:
        keyboard.append([InlineKeyboardButton("👑 Super Admin", callback_data="help_superadmin")])
    
    keyboard.append([InlineKeyboardButton("⚡ Quick Commands", callback_data="help_quick")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_message = f"""🏆 <b>Arena Of Champions Help</b> 🎮

Welcome to the ultimate gaming experience! 
Choose a category below to explore:

🎮 <b>Games:</b> Chase, Guess, Nightmare Mode
💠 <b>Shards:</b> Currency system & rewards
🏆 <b>Achievements:</b> Unlock titles & bonuses
🎯 <b>Auctions:</b> Team auctions & bidding
📊 <b>Stats:</b> Leaderboards & tracking
⚡ <b>Quick Commands:</b> Essential commands

👇 <b>Select a category to get started!</b>"""

    await safe_send(update.message.reply_text, welcome_message, parse_mode='HTML', reply_markup=reply_markup)

@check_banned
async def update_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show bot update information with user and admin sections"""
    user_id = update.effective_user.id
    is_admin = bot_instance.is_admin(user_id)
    
    if update.effective_chat.type != 'private':
        await safe_send(update.message.reply_text, 
                       "📱 <b>Please use this command in my DM!</b>\n\n"
                       "🔒 The /update command is only available in private chat for better readability.\n\n"
                       "👉 Click here to start: @ArenaOfChampionsBot", 
                       parse_mode='HTML')
        return
    
    keyboard = [
        [InlineKeyboardButton("👤 User Updates", callback_data="update_user")]
    ]
    
    if is_admin:
        keyboard.append([InlineKeyboardButton("👨‍💼 Admin Updates", callback_data="update_admin")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    
    welcome_message = f"""🎉 <b>ARENA OF CHAMPIONS - LATEST UPDATE!</b> 🚀

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 <b>Update Date:</b> {current_date}
🔄 <b>Version:</b> 3.1.0 - Enhanced Experience

👇 <b>Choose your update summary:</b>

👤 <b>User Updates:</b> New features, games, and rewards
{("👨‍💼 <b>Admin Updates:</b> Advanced admin panel & tools" if is_admin else "🔒 <b>Admin Features:</b> Contact admins for access")}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ <b>Ready to explore the new features?</b>"""

    await update.message.reply_text(welcome_message, parse_mode='HTML', reply_markup=reply_markup)

async def update_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle update information button callbacks"""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    is_admin = bot_instance.is_admin(user_id)
    data = query.data
    
    back_keyboard = [
        [InlineKeyboardButton("👤 User Updates", callback_data="update_user")]
    ]
    if is_admin:
        back_keyboard.append([InlineKeyboardButton("👨‍💼 Admin Updates", callback_data="update_admin")])
    back_keyboard.append([InlineKeyboardButton("🔙 Back to Menu", callback_data="update_main")])
    
    if data == "update_main":
        keyboard = [
            [InlineKeyboardButton("👤 User Updates", callback_data="update_user")]
        ]
        if is_admin:
            keyboard.append([InlineKeyboardButton("👨‍💼 Admin Updates", callback_data="update_admin")])
        
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")
        
        message = f"""🎉 <b>ARENA OF CHAMPIONS - LATEST UPDATE!</b> 🚀

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 <b>Update Date:</b> {current_date}
🔄 <b>Version:</b> 4.0.0 - Command Optimization Update

👇 <b>Choose your update summary:</b>

👤 <b>User Updates:</b> New features, games, and rewards
{("👨‍💼 <b>Admin Updates:</b> Advanced admin panel & tools" if is_admin else "🔒 <b>Admin Features:</b> Contact admins for access")}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ <b>Ready to explore the new features?</b>"""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "update_user":
        message = f"""👤 <b>USER UPDATE SUMMARY</b> 🎉

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

� <b>NEW: SHARD CURRENCY SYSTEM</b> 💎
<b>Your new digital wallet for the Arena Of Champions ecosystem!</b>

<b>💰 How to Earn Shards:</b>
• 🏏 <b>Chase Games:</b> 15-30 shards per game
• 🎯 <b>Guess Games:</b> 12-24 shards per game  
• 🌙 <b>Nightmare Mode:</b> 10,000 shards for victory!
• 🎁 <b>Daily Bonus:</b> 50 base + streak bonuses
• 🏆 <b>Achievements:</b> Various shard bonuses

<b>💸 Shard Commands:</b>
• <code>/balance</code> - Check your shard wallet
• <code>/dailyreward</code> - Claim daily bonus (50+ shards)
• <code>/shardlb</code> - View top shard holders
• <code>/shards @username</code> - Check others' balances

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎮 <b>ENHANCED GAMING EXPERIENCE</b>

<b>🏏 Chase Game Improvements:</b>
• Better rewards system with shard integration
• Enhanced scoring and level progression
• Real-time statistics tracking

<b>🎯 Guess Game Updates:</b>
• 5 difficulty levels to unlock progressively  
• Higher shard rewards for harder challenges
• Improved hint system and scoring

<b>🌙 NEW: Nightmare Mode</b>
• Ultimate cryptographic puzzle challenge
• Shifting numbers with mathematical hints
• Massive 10,000 shard reward for winners
• Elite status for nightmare victors

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 <b>DAILY LEADERBOARD SYSTEM</b>
<b>Compete every day for top prizes!</b>

<b>� Chase Daily Rewards:</b>
• 🥇🥈🥉 <b>Top 3:</b> 100 shards each
• 🏅 <b>4th-6th Place:</b> 80 shards each
• 🎖️ <b>7th-8th Place:</b> 60 shards each
• �️ <b>9th Place:</b> 40 shards
• �️ <b>10th Place:</b> 20 shards

<b>🎯 Guess Daily Rewards:</b>
• 🥇🥈🥉 <b>Top 3:</b> 100 shards each
• 🏅 <b>4th-6th Place:</b> 80 shards each
• 🎖️ <b>7th-8th Place:</b> 60 shards each
• �️ <b>9th Place:</b> 40 shards
• �️ <b>10th Place:</b> 20 shards

<b>📈 Track Your Progress:</b>
• <code>/dailylb</code> - View current daily rankings
• Separate leaderboards for Chase & Guess
• Rankings reset daily at midnight

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏆 <b>IMPROVED FEATURES</b>

<b>🎖️ Enhanced Achievements:</b>
• Better achievement tracking system
• Special titles for top performers
• Shard bonuses for new achievements
• Improved profile display with bold formatting

<b>📊 Better Statistics:</b>
• Comprehensive game history
• Win rate and performance tracking
• Personal bests and milestones
• Cross-game progress monitoring

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 <b>GET STARTED NOW!</b>

1. 🎁 <b>Claim Daily Bonus:</b> <code>/dailyreward</code>
2. 🎮 <b>Play Your First Game:</b> <code>/chase</code> or <code>/guess</code>
3. 🏆 <b>Check Your Stats:</b> <code>/profile</code>
4. 📊 <b>Join Daily Competition:</b> <code>/dailylb</code>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 <b>Start earning shards and dominating leaderboards today!</b>"""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(back_keyboard))
    
    elif data == "update_admin" and is_admin:
        message = f"""👨‍💼 <b>ADMIN UPDATE SUMMARY</b> 🛡️

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎛️ <b>NEW: COMPREHENSIVE ADMIN PANEL</b>
<b>Complete control center:</b> <code>/adminpanel</code>

<b>📊 8 Management Categories:</b>
• <b>Bot Statistics</b> - Real-time analytics
• <b>User Management</b> - Player oversight tools
• <b>Economy Control</b> - Shard system management
• <b>Game Management</b> - Game stats & cleanup
• <b>Admin Control</b> - Add/remove admin privileges  
• <b>Broadcasting</b> - Announcement system
• <b>Achievement System</b> - Bulk operations & titles
• <b>System Tools</b> - Maintenance & monitoring

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 <b>ENHANCED STATISTICS SYSTEM</b>

<b>📊 Real-Time Bot Analytics:</b>
• Total players and active user tracking
• Game statistics across all modes
• Shard economy circulation data
• Achievement distribution analysis
• Daily activity monitoring

<b>💎 Shard Economy Management:</b>
• <code>/giveshards @user amount [reason]</code>
• <code>/removeshards @user amount [reason]</code>
• <code>/transactions</code> - View transaction history
• Economy health monitoring
• Circulation and distribution stats

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👥 <b>ADVANCED USER MANAGEMENT</b>

<b>🔍 User Administration:</b>
• <code>/finduser @username</code> - Detailed user lookup
• Complete user profile analysis
• Activity and engagement tracking
• Achievement and shard history

<b>🛡️ Moderation Tools:</b>
• <code>/banuser @username</code> - User restriction system
• <code>/unbanuser @username</code> - Restore access
• Advanced user oversight capabilities

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📢 <b>BROADCASTING SYSTEM OVERHAUL</b>

<b>📻 Enhanced Communication:</b>
• <code>/draftbroadcast [message]</code> - Draft announcements
• <code>/testbroadcast</code> - Test with admins first
• <code>/broadcast [message]</code> - Send to all users
• Media support (photos, videos, documents)
• Delivery confirmation and statistics
• Targeted broadcasting capabilities

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎮 <b>GAME ADMINISTRATION</b>

<b>🏏 Game Management:</b>
• Real-time game statistics
• Active game monitoring
• Force cleanup stuck games
• Player performance analytics
• Leaderboard management tools

<b>📊 Advanced Analytics:</b>
• Chase game win rates and trends
• Guess game difficulty progression
• Nightmare mode completion rates
• Daily leaderboard performance tracking

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏆 <b>ACHIEVEMENT & TITLE SYSTEM</b>

<b>🎖️ Bulk Operations:</b>
• <code>/bulkward "Achievement" @user1 @user2</code> - Mass awards
• Advanced achievement management
• Title assignment and removal
• Achievement statistics and distribution
• Performance-based auto-awards

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔧 <b>SYSTEM ADMINISTRATION</b>

<b>⚙️ Maintenance Tools:</b>
• <code>/cleancache</code> - System performance optimization
• <code>/cachestatus</code> - View cache health & statistics
• <code>/threadsafety</code> - Check thread safety status
• Database health monitoring
• Connection pool management
• Error tracking and resolution
• System resource monitoring

<b>🛠️ Development Features:</b>
• Advanced logging and debugging
• Performance metrics tracking
• Database query optimization
• Real-time system health checks

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 <b>NEW ADMIN COMMANDS</b>

<b>💼 Essential Tools:</b>
• <code>/transactions [@user]</code> - Shard transaction logs
• <code>/cleancache</code> - Clear system cache
• <code>/cachestatus</code> - View cache health
• <code>/threadsafety</code> - Check thread safety status
• <code>/draftbroadcast</code> - Prepare announcements
• <code>/testbroadcast</code> - Admin-only test broadcasts
• <code>/finduser</code> - Advanced user search
• <code>/botstatus</code> - System status with shard circulation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 <b>QUICK ADMIN START GUIDE</b>

1. 🎛️ <b>Explore Panel:</b> <code>/adminpanel</code>
2. 📊 <b>Check Status:</b> <code>/botstatus</code>
3. 👥 <b>Review Users:</b> Click "User Management"  
4. 📢 <b>Test Broadcast:</b> Use draft system
5. 💎 <b>Monitor Economy:</b> Check shard circulation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🛡️ <b>Your admin powers have been significantly enhanced!</b>"""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(back_keyboard))

@check_banned
async def commands_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show all available commands with pagination based on user role"""
    user_id = update.effective_user.id
    is_super_admin = bot_instance.is_super_admin(user_id)
    is_admin = bot_instance.is_admin(user_id)
    
    if is_super_admin:
        categories = ["user", "games", "auctions", "admin", "superadmin", "system"]
        title = "👑 All Commands (Super Admin)"
    elif is_admin:
        categories = ["user", "games", "auctions", "admin"]
        title = "👨‍💼 All Commands (Admin)"
    else:
        categories = ["user", "games", "auctions"]
        title = "👤 Available Commands"
    
    keyboard = []
    row = []
    
    for i, category in enumerate(categories):
        category_names = {
            "user": "👤 Basic",
            "games": "🎮 Games", 
            "auctions": "🎯 Auctions",
            "admin": "👨‍💼 Admin",
            "superadmin": "👑 Super Admin",
            "system": "🔧 System"
        }
        
        row.append(InlineKeyboardButton(category_names[category], callback_data=f"cmd_{category}"))
        
        if len(row) == 2 or i == len(categories) - 1:
            keyboard.append(row)
            row = []
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_message = f"""🤖 <b>{title}</b>

📋 <b>Command Categories Available:</b>

👤 <b>Basic Commands:</b> Profile, balance, daily rewards
🎮 <b>Game Commands:</b> Chase, guess, nightmare mode
🎯 <b>Auction Commands:</b> Team auctions & bidding"""
    
    if is_admin:
        welcome_message += "\n👨‍💼 <b>Admin Commands:</b> User management & achievements"
    
    if is_super_admin:
        welcome_message += "\n👑 <b>Super Admin:</b> System control & database\n🔧 <b>System Commands:</b> Advanced maintenance"
    
    welcome_message += "\n\n👇 <b>Select a category to view commands:</b>"
    
    await update.message.reply_text(welcome_message, parse_mode='HTML', reply_markup=reply_markup)

async def commands_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle command category callbacks"""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    is_super_admin = bot_instance.is_super_admin(user_id)
    is_admin = bot_instance.is_admin(user_id)
    data = query.data
    
    back_button = [InlineKeyboardButton("🔙 Back to Categories", callback_data="cmd_back")]
    
    if data == "cmd_back":
        if is_super_admin:
            categories = ["user", "games", "auctions", "admin", "superadmin", "system"]
            title = "👑 All Commands (Super Admin)"
        elif is_admin:
            categories = ["user", "games", "auctions", "admin"]
            title = "👨‍💼 All Commands (Admin)"
        else:
            categories = ["user", "games", "auctions"]
            title = "👤 Available Commands"
        
        keyboard = []
        row = []
        
        for i, category in enumerate(categories):
            category_names = {
                "user": "👤 Basic",
                "games": "🎮 Games", 
                "auctions": "🎯 Auctions",
                "admin": "👨‍💼 Admin",
                "superadmin": "👑 Super Admin",
                "system": "🔧 System"
            }
            
            row.append(InlineKeyboardButton(category_names[category], callback_data=f"cmd_{category}"))
            
            if len(row) == 2 or i == len(categories) - 1:
                keyboard.append(row)
                row = []
        
        welcome_message = f"""🤖 <b>{title}</b>

📋 <b>Command Categories Available:</b>

👤 <b>Basic Commands:</b> Profile, balance, daily rewards
🎮 <b>Game Commands:</b> Chase, guess, nightmare mode  
🎯 <b>Auction Commands:</b> Team auctions & bidding"""
        
        if is_admin:
            welcome_message += "\n👨‍💼 <b>Admin Commands:</b> User management & achievements"
        
        if is_super_admin:
            welcome_message += "\n👑 <b>Super Admin:</b> System control & database\n🔧 <b>System Commands:</b> Advanced maintenance"
        
        welcome_message += "\n\n👇 <b>Select a category to view commands:</b>"
        
        await query.edit_message_text(welcome_message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
        return
    
    category = data.replace("cmd_", "")
    
    if category == "user":
        message = f"""👤 <b>Basic User Commands</b>
• <code>/start</code> - Welcome & register account
• <code>/help</code> - Interactive help menu
• <code>/commands</code> - This command list

<b>💰 ECONOMY & PROFILE:</b>  
• <code>/balance</code> - Check your shard balance
• <code>/daily</code> - Claim daily bonus (streak system)
• <code>/profile</code> - View complete gaming profile
• <code>/viewach</code> - View your achievements
• <code>/shards</code> - Shard leaderboard & stats

<b>📊 LEADERBOARDS:</b>
• <code>/leaderboard</code> - Global leaderboard menu
• <code>/dailylb</code> - Daily competition rankings
• <code>/update</code> - Bot updates & changelog

<b>🎯 UTILITY:</b>
• <code>/quit</code> - End all active games
• <code>/achievements</code> - Legacy achievements command"""
    
    elif category == "games":
        message = f"""🎮 <b>Game Commands</b>

<b>🏏 CHASE GAME:</b>
• <code>/chase</code> - Start cricket chase game
• <code>/chasestats</code> - Your chase statistics
• <code>/sleaderboard</code> - Top 10 chase players

<b>🎲 GUESS GAMES:</b>
• <code>/guess</code> - Number guessing (5 difficulties)
• <code>/guessstats</code> - Your guess statistics  
• <code>/guesslb</code> - Guess game leaderboards
• <code>/dailyguess</code> - Daily challenge
• <code>/dglb</code> - Daily guess leaderboard

<b>💀 ULTIMATE CHALLENGE:</b>
• <code>/nightmare</code> - 10,000 number challenge

<b>🎪 SPECIAL FEATURES:</b>
• <code>/goat</code> - Daily GOAT announcement
• <code>/myroast</code> - Your roast history

<b>📊 ADDITIONAL STATS:</b>
• <code>/shardlb</code> - Shard leaderboard
• <code>/shardlb</code> - Shard leaderboard"""
    
    elif category == "auctions":
        message = f"""🎯 <b>Auction System Commands</b>

<b>🏗️ AUCTION SETUP:</b>
• <code>/register</code> - Create auction proposal
• <code>/hostpanel [id]</code> - Host control panel

<b>👤 REGISTRATION:</b>
• <code>/regcap [id] [team]</code> - Register as captain
• <code>/regplay [id]</code> - Register as player

<b>💰 LIVE AUCTION:</b>
• Type amounts in chat: <code>1</code>, <code>2</code>, <code>5</code> (crores)
• <code>/myteam</code> - View your current team
• <code>/purse</code> - Check remaining budget
• <code>/out</code> - Exit current bidding

<b>📊 INFORMATION:</b>
• <code>/participants [id]</code> - View all participants
• <code>/auctionhelp</code> - Complete auction guide

<b>🎯 ADMIN CONTROLS:</b>
• <code>/auction [id]</code> - Start manual auction
• <code>/next [id]</code> - Move to next player
• <code>/setgc [id]</code> - Set group chat"""
    
    elif category == "admin" and is_admin:
        message = f"""👨‍💼 <b>Admin Commands</b>

<b>🏆 ACHIEVEMENT SYSTEM:</b>
• <code>/addach @user Achievement</code> - Award achievement
• <code>/remach @user Achievement</code> - Remove achievement
• <code>/confach [ID] [notes]</code> - Confirm achievement
• <code>/pending_conf</code> - Pending confirmations
• <code>/bulkaward "Achievement" @user1 @user2</code> - Bulk awards

<b>💰 SHARD MANAGEMENT:</b>
• <code>/giveshards @user amount [reason]</code> - Give shards
• <code>/removeshards @user amount [reason]</code> - Remove shards

<b>👥 USER MANAGEMENT:</b>
• <code>/finduser @user</code> - Find user information
• <code>/banuser @user [reason]</code> - Ban user
• <code>/unbanuser @user</code> - Unban user
• <code>/settitle @user "Title"</code> - Set player title
• <code>/removetitle @user</code> - Remove title

<b>🎯 AUCTION ADMIN:</b>
• <code>/pending</code> - View pending proposals
• <code>/listauc</code> - List all auctions
• <code>/delauc [id]</code> - Delete auction
• <code>/endauc [id]</code> - Force end auction
• <code>/addpauc [id]</code> - Add players (reply to usernames)
• <code>/removepauc [id]</code> - Remove players
• <code>/clearauc CONFIRM</code> - Clear all auctions"""
    
    elif category == "superadmin" and is_super_admin:
        message = f"""👑 <b>Super Admin Commands</b>

<b>👥 ADMIN MANAGEMENT:</b>
• <code>/addadmin @user</code> - Add new admin
• <code>/removeadmin @user</code> - Remove admin
• <code>/listadmins</code> - List all admins
• <code>/adminstatus</code> - Admin system status

<b>🗃️ DATABASE CONTROL:</b>
• <code>/resetall CONFIRM</code> - Reset entire database
• <code>/resetplayer @user</code> - Reset player data
• <code>/listplay</code> - View all registered players
• <code>/botstatus</code> - Bot & database status

<b>📢 BROADCAST SYSTEM:</b>
• <code>/broadcast message</code> - Send to all users
• <code>/draftbc message</code> - Draft broadcast
• <code>/testbc message</code> - Test broadcast

<b>🎮 GAME MANAGEMENT:</b>
• <code>/cleanupchase</code> - Force cleanup chase games
• <code>/cleanupguess</code> - Force cleanup guess games
• <code>/ddrlb</code> - Distribute daily rewards
• <code>/resetdlb CONFIRM</code> - Reset daily leaderboard
• <code>/dlbstats</code> - Daily leaderboard stats

<b>⚠️ CRITICAL OPERATIONS:</b>
• <code>/restart</code> - Restart bot system"""
    
    elif category == "system" and is_super_admin:
        message = f"""🔧 <b>System Commands</b>

<b>💾 SYSTEM MAINTENANCE:</b>
• <code>/backup</code> - Create system backup
• <code>/cleancache</code> - Clear system cache
• <code>/cstatus</code> - Cache status information
• <code>/tstatus</code> - Thread safety status
• <code>/transactions</code> - Recent transactions

<b>📊 MONITORING:</b>
• <code>/adminpanel</code> - Admin control panel
• <code>/testlog</code> - Test logging system
• <code>/emojis</code> - Achievement emoji guide

<b>🔧 TECHNICAL INFO:</b>
• System performance monitoring
• Database connection status
• Memory and thread management
• Error tracking and logging

<b>⚙️ ADVANCED FEATURES:</b>
• Automated cleanup systems
• Performance optimization tools
• Debug and diagnostic commands"""
    
    else:
        message = "❌ Access denied! You don't have permission for this category."
    
    await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup([back_button]))

async def help_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle help menu button callbacks"""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    is_super_admin = bot_instance.is_super_admin(user_id)
    is_admin = bot_instance.is_admin(user_id)
    
    data = query.data
    
    back_button = [InlineKeyboardButton("🔙 Back to Menu", callback_data="help_main")]
    
    if data == "help_main":
        keyboard = [
            [InlineKeyboardButton("🎮 Games", callback_data="help_games"),
             InlineKeyboardButton("💠 Shards", callback_data="help_shards")],
            [InlineKeyboardButton("🏆 Achievements", callback_data="help_achievements"),
             InlineKeyboardButton("📊 Stats", callback_data="help_stats")]
        ]
        
        if is_admin:
            keyboard.append([InlineKeyboardButton("👨‍💼 Admin Commands", callback_data="help_admin")])
        if is_super_admin:
            keyboard.append([InlineKeyboardButton("👑 Super Admin", callback_data="help_superadmin")])
            
        keyboard.append([InlineKeyboardButton("⚡ Quick Commands", callback_data="help_quick")])
        
        message = f"""🏆 <b>Arena Of Champions Help</b> 🎮

Welcome to the ultimate gaming experience! 
Choose a category below to explore:

🎮 <b>Games:</b> Chase, Guess, Nightmare Mode
💠 <b>Shards:</b> Currency system & rewards
🏆 <b>Achievements:</b> Unlock titles & bonuses
📊 <b>Stats:</b> Leaderboards & tracking
⚡ <b>Quick Commands:</b> Essential commands

👇 <b>Select a category to get started!</b>"""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "help_games":
        message = f"""🎮 <b>Game Commands</b> 🏏

<b>🏏 CHASE GAME:</b>
• /chase - Play Run Chase cricket game
• /sleaderboard - View top 10 chase players
• Hand cricket rules: same number = wicket!
• 🍀 15% luck factor to escape wickets
• ⚡ 10 second cooldown between games
• 💠 Earn 30-90 shards per game

<b>🎲 GUESS GAME:</b>
• /guess - Play Guess the Number game
• /guesslb - View guess game rankings
• /dailyguess - Play daily challenge
• 5 difficulty levels: Beginner to Expert
• 🔓 Win games to unlock harder levels
• 💠 Earn 25-65 shards per game

<b>💀 NIGHTMARE MODE:</b>
• /nightmare - Ultimate challenge game
• 🎯 Guess number from 1-10,000 in 3 attempts
• ⚡ Secret number shifts after wrong guesses
• 💡 Clear mathematical hints provided
• 🏆 Win 10,000 💠 + exclusive title

<b>📊 DAILY COMPETITIONS:</b>
• /dailylb - View daily leaderboards
• Daily competitions for Chase & Guess
• 🥇🥈🥉 Top 3 earn bonus shards at midnight"""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup([back_button]))
    
    elif data == "help_shards":
        message = f"""💠 <b>Shards System</b> 💰

<b>💠 BASIC COMMANDS:</b>
• /balance - Check your shard balance
• /daily - Claim daily shard bonus (50-250)
• /shards - View shard leaderboard & stats

<b>💰 EARNING SHARDS:</b>
• 🎮 Games: 25-90 shards per game
• 🏆 Achievements: 100-200 shards each
• 📅 Daily bonus: 50-250 with streak bonus
• 🐐 GOAT winner: 300 bonus shards
• 🥇 Daily leaderboard: Bonus rewards

<b>🔥 DAILY BONUS SYSTEM:</b>
• Claim every day to build your streak
• Higher streaks = bigger bonuses
• Maximum bonus increases with consistency
• Never miss a day to maximize rewards!

<b>🏆 ACHIEVEMENT BONUSES:</b>
• Winner: 150 shards
• Orange Cap: 200 shards
• Purple Cap: 200 shards
• MVP: 100 shards
• Special achievements: Up to 500 shards"""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup([back_button]))
    
    elif data == "help_achievements":
        message = f"""🏆 <b>Achievements System</b> 🏅

<b>🏆 BASIC COMMANDS:</b>
• /viewach - View your achievements
• /profile - View comprehensive profile

<b>🏅 AVAILABLE ACHIEVEMENTS:</b>
• 🏆 Winner - Tournament victory
• 🟧 Orange Cap - Top scorer
• 🟪 Purple Cap - Best bowler
• 🏅 MVP - Most Valuable Player
• 🎖️ Captain - Leadership achievement
• ⭐ Special tournament achievements

<b>🎯 HOW TO EARN:</b>
• Participate in tournaments
• Achieve excellence in games
• Consistent high performance
• Admin recognition for special plays
• Community contributions

<b>💠 ACHIEVEMENT REWARDS:</b>
• Instant shard bonus (100-500)
• Exclusive player titles
• Leaderboard recognition
• Special status in community

<b>📈 PROGRESSION:</b>
• Achievements unlock new titles
• Build your gaming reputation
• Track your journey in /profile"""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup([back_button]))
    
    elif data == "help_auctions":
        message = f"""🎯 <b>Auction System</b> 🏆

<b>📝 AUCTION SETUP:</b>
• /register - Create auction proposal (ANY USER)
• /hostpanel [id] - Host control panel
• /pending - View pending proposals (Admin)
• /listauc - List all auctions (Admin)

<b>👑 REGISTRATION:</b>
• /regcap [id] [team] - Register as captain
• /regplay [id] - Register as player
• Both require host/admin approval

<b>🎯 LIVE BIDDING (Manual System):</b>
• Type amounts directly in chat: 1, 2, 5, 10 (crores)
• ".." - Admin control (next player, sell, etc.)
• /myteam - View your current team
• /purse - Check remaining budget
• /out - Exit bidding for current player

<b>🔧 ADMIN COMMANDS:</b>
• /auction [id] - Start manual auction
• /next [id] - Move to next player
• /setgc [id] - Set group chat for auction
• /participants [id] - View all participants
• /addpauc [id] - Add players (reply to usernames)
• /removepauc [id] - Remove players (reply to usernames)
• /delauc [id] - Delete auction
• /endauc [id] - Force end auction
• /clearauc CONFIRM - Clear all auction data

<b>💰 CRORE-BASED SYSTEM:</b>
• All amounts in crores (1cr, 2.5cr, 10cr)
• Type simple numbers to bid: 1, 2, 5
• Real-time purse tracking
• Manual progression by admin

<b>📊 FEATURES:</b>
• Manual auction progression
• Crore-based amounts for easier bidding
• Admin "." controls for quick actions
• Group chat integration
• Complete participant management"""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup([back_button]))
    
    elif data == "help_stats":
        message = f"""📊 <b>Statistics & Leaderboards</b> 📈

<b>🏏 CHASE STATISTICS:</b>
• /sleaderboard - Top 10 chase players
• /chasestats - Your personal & global chase stats
• Track your best scores and levels

<b>🎲 GUESS STATISTICS:</b>
• /guesslb - Guess game rankings
• /guessstats - Your personal & global guess stats
• Monitor your win rates and progression

<b>📅 DAILY LEADERBOARDS:</b>
• /dailylb - Daily chase/guess leaderboards
• Reset every day at midnight
• Compete for daily champion status
• Top 3 earn bonus shard rewards

<b>💀 NIGHTMARE STATISTICS:</b>
• Track your nightmare victories
• Elite leaderboard for masters
• Exclusive statistics for winners

<b>💠 SHARD TRACKING:</b>
• /shards - Shard leaderboard & stats
• Monitor your earning patterns
• Compare with other players

<b>📊 PERSONAL STATS:</b>
• /profile - Complete gaming history
• Track achievements and titles
• Monitor your progression across all games"""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup([back_button]))
    
    elif data == "help_quick":
        message = f"""⚡ <b>Quick Command Reference</b> 🚀

<b>🎮 ESSENTIAL GAMES:</b>
• /chase - Cricket chase game
• /guess - Number guessing
• /nightmare - Ultimate challenge

<b>🎮 GAME MANAGEMENT:</b>
• /quit - End all active games
• /help - Navigate with buttons

<b>💠 SHARDS & REWARDS:</b>
• /balance - Check shards
• /daily - Daily bonus
• /shards - Shard leaderboard

<b>🏆 PROFILE & ACHIEVEMENTS:</b>
• /viewach - Your achievements
• /profile - Complete profile
• /start - Register/update profile

<b>📊 LEADERBOARDS:</b>
• /sleaderboard - Chase top 10
• /guesslb - Guess rankings
• /dailylb - Daily competitions

<b>🎯 AUCTIONS:</b>
• /register - Create auction
• /regcap [id] [team] - Register as captain
• /regplay [id] - Register as player
• Type amounts (1, 2, 5) to bid in chat

<b>🎯 OTHER FEATURES:</b>
• /goat - Daily GOAT announcement
• /myroast - Your roast history

<b>💡 PRO TIPS:</b>
• Use /quit to easily switch between games
• Play games daily for better rewards
• Build streaks for bonus shards
• All progress is saved when you quit games"""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup([back_button]))
    
    elif data == "help_admin" and is_admin:
        message = f"""👨‍💼 <b>Admin Commands</b> 🛡️

<b>🏆 ACHIEVEMENT MANAGEMENT:</b>
• /addach @user Achievement
• /remach @user Achievement
• /bulkward "Achievement" @user1 @user2
• /settitle @user "Title"
• /emojis - Achievement emoji guide

<b>💠 SHARD MANAGEMENT:</b>
• /giveshards @user amount [reason]
• /removeshards @user amount [reason]

<b>🎮 GAME ADMINISTRATION:</b>
• /cleanupchase - Force cleanup chase games
• /cleanupguess - Force cleanup guess games

<b>🎯 AUCTION ADMINISTRATION:</b>
• /listauc - View all auctions
• /delauc <id> - Delete auction
• /endauc <id> - Force end auction
• /pending - View pending proposals
• /clearauc CONFIRM - Clear all auctions

<b>📊 DAILY LEADERBOARDS:</b>
• /ddrlb - Trigger daily rewards
• /resetdailylb CONFIRM - Reset daily leaderboard
• /dlbstats - Daily leaderboard statistics

<b>🎯 SPECIAL FEATURES:</b>
• /confach [ID] [notes] - Confirm achievement
• /pending_conf - Pending confirmations
• /broadcast message - Broadcast to all

<b>📝 EXAMPLES:</b>
• /addach @john Winner
• /giveshards @player 500 Tournament winner
• /settitle @mike "Captain\""""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup([back_button]))
    
    elif data == "help_superadmin" and is_super_admin:
        message = f"""👑 <b>Super Admin Commands</b> ⚡

<b>👥 ADMIN MANAGEMENT:</b>
• /addadmin @user - Add new admin
• /removeadmin @user - Remove admin
• /listadmins - List all admins
• /adminstatus - Show admin hierarchy & conflicts

<b>🗜️ DATABASE MANAGEMENT:</b>
• /resetall CONFIRM - Reset entire database
• /resetplayer @user - Reset player data
• /listplay - View all players
• /botstatus - Bot and database status

<b>📢 BROADCAST SYSTEM:</b>
• /broadcast message - Send to all users/groups
• /groups - View active groups

<b>⚠️ CRITICAL OPERATIONS:</b>
• All admin commands available
• Full database access
• System-level operations
• Emergency cleanup tools

<b>🔒 SECURITY FEATURES:</b>
• Confirmation required for destructive operations
• Comprehensive logging
• Backup and recovery tools

<b>📊 SYSTEM MONITORING:</b>
• Monitor bot performance
• Track database health
• Manage system resources

<b>💡 SUPER ADMIN TIPS:</b>
• Use CONFIRM for destructive operations
• Regular system health checks
• Monitor user activity patterns"""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup([back_button]))

GOAT_ROAST_LINES = [

    "🖕 {name} pitch pe utarta hai gaand marwane, runs lene nahi!",
    "🤬 {name} ka batting dekh ke lagta hai bench pe aaya chutiyapa karne!",
    "💩 {name} ki batting potty jaisi hai – smell hi smell, runs zero!",
    "🍌 {name} shots maarne me la*de ka bhi istemal nahi karta!",
    "🔥 {name} wicket donate kar deta hai jaise randi apna maal!",
    "🧻 {name} team ke liye tissue paper hai – ek use me fenk diya!",
    "🚮 {name} ka naam hi dustbin me likhna chahiye!",
    "👙 {name} ka game bra jaisa hai – support de nahi paata!",
    "🍆 {name} ke shots condom bina ka sex – risk hi risk!",
    "🍼 {name} batting me baby jaise hai – har over me susu karta hai!",
    "🤡 {name} ka cricket IQ condom ke expiry date jaisa zero hai!",
    "⚰️ {name} ka batting innings kabristan se bhi zyada dead hai!",
    "🪣 {name} batting me balti jaisi awaaz karta hai, run zero!",
    "🍺 {name} ke shots daaru ki bottle jaise – toot ke bikhar jaate hain!",
    "🥵 {name} wicket pe utarta hai aur 2 minute me nikal jaata hai – honeymoon ka record tod diya!",
    "🪠 {name} ka bowling dekh ke lagta hai ball nikalne ke liye plumber bulaana padega gaand se!",
    "🚬 {name} fielding karte waqt ball se jyada sutta dhundhta hai!",
    "🪳 {name} ke shots cockroach ki tarah bhag jaate hai – idhar-udhar ulti gati!",
    "👙 {name} ka batting form utna loose hai jitna Nehru Nagar ki randi ka blouse!",
    "🛶 {name} ki batting Titanic jaisi hai – shuru hote hi doob jaati hai!",
    "🎪 {name} ka fielding dekh ke circus wale bhi bolte hain, 'isko humare pass bhejo!'",
    "🥴 {name} ki shot selection utni hi bakwaas hai jitni uske chhapri doston ki advice!",
    "🥗 {name} bowler se salad banwata hai, ekdum chopped!",
    "🥷 {name} toss ke baad gaayab ho jaata hai jaise girlfriend ka reply!",
    "🧅 {name} ka batting dekh ke aankh me paani aa jaata hai – pure onion cutting feels!",
    "🧨 {name} ka six attempt Diwali ke patakhe jaisa hota hai – ya to footega ya chootega!",
    "🧹 {name} sweep shot me ball se jyada mitti udaata hai!",
    "🥶 {name} ko ball dekh ke hi thand lag jaati hai gaand tak!",
    "🧟 {name} ka footwork zombie jaisa hai – dheere dheere ghisakna bas!",
    "🥊 {name} ball se jyada teammates ki gaali khata hai!",
    "🧃 {name} ka batting juice machine jaisa hai – dabaate hi nikla wicket!",
    "🍌 {name} ki straight drive seedha slip me jaati hai, jaise apna career seedha gutter me!",
    "🧩 {name} team ke liye woh missing condom hai jo hamesha faata hua milta hai!",
    "🧻 {name} ekdum tissue paper player hai – ek baar use karo aur seedha dustbin!",
    "🕳️ {name} fielding me itne holes chhodta hai jaise maa-behen ke gaaliyon ka dictionary!",
    "🍼 {name} fielding karte waqt itna rota hai jaise maa ne doodh band kar diya ho!",
    "🛑 {name} ke dot balls dekh ke lagta hai iski gaand pe signal laga hua hai – hamesha RED!",
    "🦍 {name} ball ko maarne aata hai, lagta hai bandar ko bat de diya ho!",
    "🦆 {name} itne ducks kha gaya hai ki ab ande dena shuru karega!",
    "💀 {name} apna khud ka funeral khelta hai har match me!",
    "🧨 {name} pressure me foot jaata hai jaise Diwali ka 2 rupaye ka phatka!",
    "🪓 {name} ka shot utna hi dangerous hai jitna uske gaon ka tutta hua hathoda!",
    "🥚 {name} ne utne ducks banaye hain jitne ande murgi farm deta hai!",
    "🪞 {name} batting practice sirf mirror me karta hai – ground pe bawasir!",
    "📡 {name} ke shots utna upar jaate hai ki NASA wale bhi confuse ho jaate hai!",
    "🧨 {name} ka form ab lund pe latak gaya hai – runs zero, attitude 100!",
    "🧩 {name} ke batting me wo missing piece hai – talent!",
    "🧻 {name} ke batting records toilet paper jaisa – bas use karke fek do!",
    "🚪 {name} pitch pe bas exit marne aata hai, entrance ka matlab hi nahi!",
    "💩 {name} ka cricket career tatti jaise – flush karne ka mann karta hai!",
    "🪠 {name} bowling se jyada gaand unblock karne ka kaam kar raha hai!",
    "🍷 {name} pitch pe utna hi high hota hai jitna daaru pe!",
    "🛏️ {name} ka batting innings sleeping pill hai – sabko ground pe sula deta hai!",
    "📦 {name} apni wicket gift pack karke free home delivery deta hai!",
    "🥵 {name} ka form utna garam hai jitna December me thandi chai!",
    "🧻 {name} ke game ka naam 'Ek Baar Use Karo Aur Feko' hona chahiye!",
    "🎯 {name} ka cricket career Tinder jaisa hai – swipe left, match kabhi nahi!",
    "🥊 {name} wicket girte hi apni hi gaand maar leta hai!",
    "🪩 {name} ka batting disco ball jaisa hai – shine jyada, kaam zero!",
    "🍵 {name} pitch pe khelne se jyada chai break leta hai!",
    "🛋️ {name} ka dugout hi permanent address hai!",
    "🚬 {name} ball se jyada umpire se lighter maangta hai!",
    "🧠 {name} ka cricket IQ randi ke client list jaisa – zero sorted!",
    "🧻 {name} ek baar crease pe aata hai aur poore team ka kabada kar deta hai!",
    "🧟 {name} ka batting dekh ke lagta hai zombie ko bat pakda diya ho!",
    "🧨 {name} bowling me Diwali ka rassi bomb hai – awaaz badi, kaam zero!",
    "👙 {name} ke shots randi ke blouse se bhi jyada loose hai!",
    "🪠 {name} ke yorker dekh ke lagta hai condom faat gaya ho!",
    "🚮 {name} ka naam hi kachra gadi me hona chahiye!",
    "🩸 {name} ka batting dekh ke lagta hai period pain se bhi zyada dard hota hai!",
    "🧟‍♂️ {name} ka stamina murde jaisa hai – chal nahi pata!",
    "🍼 {name} ka game baby diapers jaisa hai – har over leak!",
    "🧨 {name} batting karte hi phat jaata hai – bas awaaz karta hai!",
    "🧻 {name} bowling me bhi tissue paper – bas waste hi waste!",
    "🕳️ {name} ke gloves se ball guzar jaata hai jaise condom ke hole se!",
    "🔥 {name} shots itne bakwas maarta hai ki gaand pe bhi boundary nahi lagti!",
    "🛠️ {name} ka bat hammer jaisa lagta hai – ball todta nahi, game tod deta hai!",
    "🚾 {name} ka batting dekh ke lagta hai toilet training bhi adhuri chhodi hai!",
    "🧠 {name} ka shot selection toilet me chipke tissue jaisa hai – bekaar aur chipka hua!",
    "🧨 {name} ka form pura bawasir ho gaya hai – har ball pe dard hi dard!",
    "🪣 {name} ka bowling bucket shot jaisa hai – bas leak hi leak!",
    "🤬 {name} team ka asli chut*ya hai – kaam zero, bakchodi 100!",
]

# ---- RUN CHASE SIMULATOR (Hand Cricket) ----

LEVELS = {
    1: {"target": 20,  "balls": 10, "wickets": 1},
    2: {"target": 40,  "balls": 15, "wickets": 1},
    3: {"target": 50,  "balls": 18, "wickets": 1},
    4: {"target": 75,  "balls": 25, "wickets": 2},
    5: {"target": 80,  "balls": 25, "wickets": 2},
    6: {"target": 90,  "balls": 30, "wickets": 2},
    7: {"target": 100, "balls": 50, "wickets": 2},
    8: {"target": 100, "balls": 40, "wickets": 3},
    9: {"target": 125, "balls": 50, "wickets": 3},
    10: {"target": 150, "balls": 50, "wickets": 3},
}

ACTIVE_CHASE_GAMES = ThreadSafeDict()  # user_id -> game_data
MAX_CONCURRENT_GAMES = 3  # Max games per user
GAME_TIMEOUT = 1800  # 30 minutes timeout

CHASE_KEYBOARD = InlineKeyboardMarkup([
    [InlineKeyboardButton("1️⃣", callback_data="chase:1"),
    InlineKeyboardButton("2️⃣", callback_data="chase:2"),
    InlineKeyboardButton("3️⃣", callback_data="chase:3")],
    [InlineKeyboardButton("4️⃣", callback_data="chase:4"),
    InlineKeyboardButton("5️⃣", callback_data="chase:5"),
    InlineKeyboardButton("6️⃣", callback_data="chase:6")],
])

def _chase_keyboard():
    return CHASE_KEYBOARD

def _format_chase_card(state: dict, last_event: str | None = None) -> str:
    level = state["level"]
    target = state["target"]
    balls_left = state["balls_left"]
    score = state["score"]
    balls_used = state["balls_used"]
    owner_name = state.get("player_name", "Player")
    wickets_left = state.get("wickets_left", 1)

    wickets_fallen = LEVELS[level].get("wickets", 1) - wickets_left
    
    message = (
        f"🏏 <b>CHASE GAME - Level {level}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"👤 <b>Player:</b> {H(owner_name)}\n"
        f"🎯 <b>Target:</b> {target} runs\n"
        f"🏏 <b>Score:</b> {score}/{wickets_fallen}\n"
        f"📊 <b>Need:</b> {target - score} in {balls_left} balls\n"
        f"💪 <b>Wickets:</b> {wickets_left} left\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
    )
    
    progress = (score / target) if target > 0 else 0
    filled = int(progress * 10)
    bar = "▓" * filled + "░" * (10 - filled)
    message += f"📈 <b>Progress:</b> {bar} {int(progress * 100)}%\n"
    
    if last_event:
        message += f"\n⚡ <b>Last ball:</b> {last_event}\n"
    
    message += f"\n🎯 <b>Choose:</b> 1 • 2 • 3 • 4 • 6 • OUT"
    
    return message

def _reset_level_state(user_id: int, user_name: str, level: int) -> dict:
    wickets = LEVELS[level]["wickets"]
    
    return {
        "player_id": user_id,
        "player_name": user_name,
        "level": level,
        "target": LEVELS[level]["target"],
        "balls_left": LEVELS[level]["balls"],
        "balls_used": 0,
        "score": 0,
        "wickets_left": wickets,
        "message_id": None,   # filled after first send
        "chat_id": None,      # filled after first send
        "active": True,
        "last_action_time": 0,  # For rate limiting
        "start_time": time.time(),  # Track when game started
    }

def initialize_roast_rotation(conn):
    """Initialize the roast rotation system with all roast lines."""
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS roast_rotation (
                id SERIAL PRIMARY KEY,
                roast_line TEXT NOT NULL UNIQUE,
                last_used_date DATE,
                usage_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS roast_usage (
                id SERIAL PRIMARY KEY,
                roast_line TEXT NOT NULL,
                player_id INTEGER REFERENCES players(id),
                used_date DATE NOT NULL DEFAULT CURRENT_DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        for roast_line in GOAT_ROAST_LINES:
            cursor.execute("""
                INSERT INTO roast_rotation (roast_line, usage_count) 
                VALUES (%s, 0) 
                ON CONFLICT (roast_line) DO NOTHING
            """, (roast_line,))
        
        conn.commit()
        logger.info("Roast rotation system initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing roast rotation: {e}")
        conn.rollback()

def get_next_roast_line(conn):
    """
    Get the next roast line using rotation system.
    Rules:
    1. Use lines that haven't been used yet
    2. If all lines used once, randomly pick from least used
    3. If a line has been used, it can only repeat after all others have been used once
    """
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT roast_line FROM roast_rotation 
            WHERE usage_count = 0 
            ORDER BY RANDOM() 
            LIMIT 1
        """)
        
        result = cursor.fetchone()
        if result:
            return result[0]
        
        cursor.execute("""
            SELECT roast_line FROM roast_rotation 
            WHERE usage_count = (SELECT MIN(usage_count) FROM roast_rotation)
            ORDER BY RANDOM() 
            LIMIT 1
        """)
        
        result = cursor.fetchone()
        if result:
            return result[0]
        
        return random.choice(GOAT_ROAST_LINES)
        
    except Exception as e:
        logger.error(f"Error getting next roast line: {e}")
        return random.choice(GOAT_ROAST_LINES)

def update_roast_usage(conn, roast_line, player_id):
    """Update roast usage statistics."""
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE roast_rotation 
            SET usage_count = usage_count + 1, last_used_date = CURRENT_DATE 
            WHERE roast_line = %s
        """, (roast_line,))
        
        cursor.execute("""
            INSERT INTO roast_usage (roast_line, player_id, used_date) 
            VALUES (%s, %s, CURRENT_DATE)
        """, (roast_line, player_id))
        
        conn.commit()
        
    except Exception as e:
        logger.error(f"Error updating roast usage: {e}")
        conn.rollback()

@check_banned
async def quit_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Quit all active games and show current status"""
    user = update.effective_user
    user_id = user.id
    
    active_games = bot_instance.get_all_active_games(user_id)
    
    total_active = 0
    if active_games['guess']:
        total_active += 1
    if active_games['nightmare']:
        total_active += 1
    total_active += len(active_games['chase'])
    
    if total_active == 0:
        await update.message.reply_text(
            "🎮 <b>No Active Games</b>\n\n"
            "You don't have any active games to quit.\n\n"
            "🎯 Start a new game:\n"
            "• /chase - Cricket Run Chase\n"
            "• /guess - Number Guessing Game\n"
            "• /nightmare - Nightmare Mode Challenge",
            parse_mode='HTML'
        )
        return
    
    message = f"🚪 <b>Quitting {total_active} Active Game{'s' if total_active > 1 else ''}</b>\n\n"
    
    games_quit = []
    
    if active_games['guess']:
        bot_instance.end_guess_game(user_id, 'quit')
        g = active_games['guess']
        games_quit.append(f"🎯 <b>Guess Game</b> ({g['difficulty']}) - {g['attempts_left']} attempts left")
    
    if active_games['nightmare']:
        conn = bot_instance.get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE nightmare_games 
                    SET is_completed = TRUE, completed_at = CURRENT_TIMESTAMP, result = 'quit'
                    WHERE player_telegram_id = %s AND NOT is_completed
                """, (user_id,))
                conn.commit()
            except Exception as e:
                logger.error(f"Error ending nightmare game: {e}")
            finally:
                bot_instance.return_db_connection(conn)
        
        n = active_games['nightmare']
        games_quit.append(f"💀 <b>Nightmare Mode</b> - {n['attempts_left']} attempts left")
    
    for i, chase in enumerate(active_games['chase']):
        for key, game_state in list(ACTIVE_CHASE_GAMES.items()):
            if game_state.get('player_id') == user_id:
                ACTIVE_CHASE_GAMES.pop(key, None)
                break
        
        games_quit.append(f"🏏 <b>Chase Game #{i+1}</b> - Level {chase['level']}, {chase['score']}")
    
    message += "✅ <b>Games Ended:</b>\n\n"
    for game in games_quit:
        message += f"• {game}\n"
    
    message += f"\n🎯 <b>All games have been ended!</b>\n"
    message += f"💠 <b>Note:</b> Any progress or shard rewards are saved.\n\n"
    message += f"🎮 <b>Start a new game anytime:</b>\n"
    message += f"• /chase - Cricket Run Chase\n"
    message += f"• /guess - Number Guessing Game\n" 
    message += f"• /nightmare - Nightmare Mode Challenge"
    
    await update.message.reply_text(message, parse_mode='HTML')

@check_banned
async def goat_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Select and announce the GOAT (Greatest Of All Time) player of the day."""
    today = date.today()
    
    conn = bot_instance.get_db_connection()
    if not conn:
        await update.message.reply_text("❌ <b>Database error!</b>\n\nPlease try again later.", parse_mode='HTML')
        return
    
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_goat (
                id SERIAL PRIMARY KEY,
                date DATE UNIQUE NOT NULL,
                player_id INTEGER REFERENCES players(id) ON DELETE CASCADE,
                roast_line TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'daily_goat' AND column_name = 'roast_line'
        """)
        
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE daily_goat ADD COLUMN roast_line TEXT")
            logger.info("Added missing roast_line column to daily_goat table")
        
        cursor.execute("SELECT player_id, roast_line FROM daily_goat WHERE date = %s", (today,))
        result = cursor.fetchone()
        
        if result:
            player_id, stored_roast = result
            if not stored_roast:
                stored_roast = bot_instance.get_cached_roast_line()
                cursor.execute(
                    "UPDATE daily_goat SET roast_line = %s WHERE date = %s",
                    (stored_roast, today)
                )
        else:
            cursor.execute("""
                SELECT id FROM players 
                WHERE id NOT IN (
                    SELECT player_id FROM daily_goat 
                    WHERE date = %s - INTERVAL '1 day'
                )
                ORDER BY RANDOM() LIMIT 1
            """, (today,))
            
            player_result = cursor.fetchone()
            
            if not player_result:
                cursor.execute("SELECT id FROM players ORDER BY RANDOM() LIMIT 1")
                player_result = cursor.fetchone()
                
            if not player_result:
                await update.message.reply_text("❌ <b>No players found!</b>\n\nPlease register some players first.", parse_mode='HTML')
                return
                
            player_id = player_result[0]
            
            initialize_roast_rotation(conn)
            
            stored_roast = bot_instance.get_cached_roast_line()
            
            bot_instance.update_roast_usage_async(stored_roast)
            
            update_roast_usage(conn, stored_roast, player_id)
            
            cursor.execute(
                "INSERT INTO daily_goat (date, player_id, roast_line) VALUES (%s, %s, %s)",
                (today, player_id, stored_roast)
            )
            conn.commit()
            
            try:
                player_telegram_result = cursor.execute(
                    "SELECT telegram_id FROM players WHERE id = %s", (player_id,)
                )
                player_data = cursor.fetchone()
                
                if player_data:
                    telegram_id = player_data[0]
                    goat_bonus = 300  # Special bonus for being daily GOAT
                    
                    shard_success = bot_instance.award_shards(
                        telegram_id,
                        goat_bonus,
                        'goat_bonus',
                        f'Daily GOAT winner for {today}'
                    )
                    
                    if shard_success:
                        logger.info(f"Awarded {goat_bonus} GOAT bonus shards to player {player_id}")
                
            except Exception as e:
                logger.error(f"Error awarding GOAT bonus shards: {e}")
        
        cursor.execute("SELECT id, telegram_id, username, display_name, title FROM players WHERE id = %s", (player_id,))
        player = cursor.fetchone()
        
        if not player:
            await update.message.reply_text("❌ <b>Player not found!</b>\n\nPlease try again.", parse_mode='HTML')
            return
        
        player_dict = {
            "id": player[0],
            "telegram_id": player[1], 
            "username": player[2],
            "display_name": player[3],
            "title": player[4]
        }
        
        achievements = bot_instance.get_player_achievements(player_id)
        
        date_str = today.strftime("%B %d, %Y")
        
        player_mention = H(player_dict['display_name'])
        if player_dict.get('telegram_id'):
            try:
                player_mention = f'<a href="tg://user?id={player_dict["telegram_id"]}">{H(player_dict["display_name"])}</a>'
            except:
                pass
        
        if achievements and len(achievements) > 0:
            total_awards = 0
            message = (
                "🐐🔥 <b>✨ TODAY'S GOAT ✨</b> 🔥🐐\n"
                "━━━━━━━━━━━━━━━\n"
                f"📅 <b>{date_str}</b>\n"
                f"👑 <b>{player_mention}</b>\n"
            )

            if player_dict['title']:
                message += f"🏆 <i>{player_dict['title']}</i>\n"

            message += "━━━━━━━━━━━━━━━\n\n"
            message += "🎖️ <b>ACHIEVEMENTS</b> 🎖️\n"
            message += "━━━━━━━━━━━━━━━\n"

            for ach, count in achievements:
                emoji = bot_instance.get_achievement_emoji(ach)
                count_display = f" ×{count}" if count > 1 else ""
                message += f"{emoji} <b>{ach.upper()}</b>{count_display}\n"
                total_awards += count

            message += (
                f"\n💎 <b>Total Awards:</b> {total_awards} 💎\n"
                "━━━━━━━━━━━━━━━\n"
                "🙌 <b>ALL HAIL THE GOAT!</b> 🙌"
            )

            
        elif player_dict['title']:
            message = (
                "🐐🔥 <b>✨ TODAY'S GOAT ✨</b> 🔥🐐\n"
                "━━━━━━━━━━━━━━━\n"
                f"📅 <b>{date_str}</b>\n"
                f"👑 <b>{player_mention}</b>\n"
                "━━━━━━━━━━━━━━━\n\n"
                f"🏆 <b>Title:</b> <i>{H(player_dict['title'])}</i> 🏆\n"
                "━━━━━━━━━━━━━━━\n\n"
                "💪 <i>Keep grinding to turn that title into legendary status!</i> 🚀\n"
                "🌟 <i>Titles are given, legends are made on the field!</i> 🌟"
            )

            
        else:
            if not stored_roast:
                stored_roast = "{name}, even the scoreboard feels sorry for you! 🏏😅"
            
            roast = stored_roast.format(name=H(player_dict['display_name']))
            message = (
                "🏏💥 <b>🔥 TODAY'S GOAT 🔥</b> 💥🏏\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"📅 <b>{date_str}</b>\n"
                f"👑 <b>{player_mention}</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                "🐐 <b>Roast of the Day</b> 🐐\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"😂 {H(roast)}\n\n"
                "📢 <i>Step up champ… the team’s watching! 🏆</i>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "⚡ <i>Earn those achievements and silence the haters!</i> 💪\n"
            )

        
        await safe_send(update.message.reply_text, message, parse_mode='HTML')
        logger.info(f"GOAT announcement sent for {player_dict['display_name']}")
    
    except Exception as e:
        log_exception("goat_command", e, update.effective_user.id)
        await safe_send(update.message.reply_text, 
                       ERROR_MESSAGES['generic'], 
                       parse_mode='HTML')
    finally:
        bot_instance.return_db_connection(conn)

@check_banned
async def my_roast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show all roast lines used for the calling player with dates in a table format."""
    user_id = update.effective_user.id
    
    conn = bot_instance.get_db_connection()
    if not conn:
        await update.message.reply_text("❌ <b>Database error!</b>\n\nPlease try again later.", parse_mode='HTML')
        return
    
    try:
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, display_name FROM players WHERE telegram_id = %s", (user_id,))
        player = cursor.fetchone()
        
        if not player:
            await update.message.reply_text(
                "❌ <b>Player not found!</b>\n\n"
                "You need to be registered to use this command.\n"
                "Ask an admin to register you first!", 
                parse_mode='HTML'
            )
            return
        
        player_id, display_name = player
        
        cursor.execute("""
            SELECT roast_line, used_date 
            FROM roast_usage 
            WHERE player_id = %s 
            ORDER BY used_date DESC
        """, (player_id,))
        
        roast_history = cursor.fetchall()
        
        if not roast_history:
            await update.message.reply_text(
                f"🎭 <b>{display_name}'s Roast History</b> 🎭\n\n"
                "🍀 <b>Lucky you!</b> No roasts yet!\n"
                "🏆 <i>Keep it up and maybe you'll never see one!</i>", 
                parse_mode='HTML'
            )
            return
        
        message = f"🎭 <b>{display_name}'s Roast Collection</b> 🎭\n\n"
        message += "📋 <b>Your Complete Roast History:</b>\n"
        message += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for idx, (roast_line, used_date) in enumerate(roast_history, 1):
            formatted_roast = roast_line.format(name=display_name)
            date_str = used_date.strftime("%d/%m/%Y") if used_date else "Unknown"
            
            message += f"<b>#{idx}</b> 📅 <code>{date_str}</code>\n"
            message += f"🎯 {formatted_roast}\n\n"
        
        message += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        message += f"📊 <b>Statistics:</b>\n"
        message += f"🎭 Total Roasts: <b>{len(roast_history)}</b>\n"
        
        if roast_history:
            latest_date = roast_history[0][1].strftime("%d/%m/%Y") if roast_history[0][1] else "Unknown"
            message += f"📅 Latest Roast: <b>{latest_date}</b>\n"
        
        MAX_MESSAGE_LENGTH = 4090  # Leave buffer for safety
        
        if len(message) > MAX_MESSAGE_LENGTH:
            messages = split_message_safely(message, MAX_MESSAGE_LENGTH)
            
            for i, msg in enumerate(messages):
                if i == 0:
                    await update.message.reply_text(msg, parse_mode='HTML')
                else:
                    await asyncio.sleep(0.5)  # Prevent rate limiting
                    await update.message.reply_text(msg, parse_mode='HTML')
        else:
            await update.message.reply_text(message, parse_mode='HTML')
        
        logger.info(f"Roast history sent for {display_name}")
    
    except Exception as e:
        logger.error(f"Error in /myroast command: {e}")
        await update.message.reply_text(
            "❌ <b>Something went wrong!</b>\n\nPlease try again later.", 
            parse_mode='HTML'
        )
    finally:
        bot_instance.return_db_connection(conn)

# ---- RUN CHASE SIMULATOR (Hand Cricket) ----

GAME_EXPIRY_TIME = 1800  # 30 minutes

def cleanup_expired_games() -> int:
    """Clean up expired/inactive chase games to prevent memory leaks."""
    now = time.time()
    expired = []
    
    for key, state in list(ACTIVE_CHASE_GAMES.items()):
        if not state.get("active", True):
            expired.append(key)
            continue
            
        last_action = state.get("last_action_time", 0)
        if now - last_action > GAME_TIMEOUT:
            expired.append(key)
    
    for key in expired:
        ACTIVE_CHASE_GAMES.pop(key, None)
        logger.info(f"Cleaned up expired chase game: {key} (inactive for {GAME_TIMEOUT//60} minutes)")
    
    return len(expired)

def get_user_active_games(user_id: int) -> int:
    """Get count of active games for a user with cleanup"""
    return sum(1 for state in ACTIVE_CHASE_GAMES.values() 
               if state.get("player_id") == user_id and state.get("active", False))

@check_banned
async def chase_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start the Run Chase Simulator with enhanced performance"""
    user = update.effective_user
    name = H(user.full_name or user.first_name or f"User{user.id}")  # HTML-safe display name
    raw_name = user.full_name or user.first_name or f"User{user.id}"  # For DB storage
    current_time = time.time()

    active_games = get_user_active_games(user.id)
    if active_games >= MAX_CONCURRENT_GAMES:
        transition_msg, _ = bot_instance.create_game_switch_message(user.id, "Chase Game")
        enhanced_msg = f"⚠️ <b>Maximum Games Reached!</b>\n\n{transition_msg}\n\n⏰ Games auto-expire after 10 minutes of inactivity."
        await update.message.reply_text(enhanced_msg, parse_mode="HTML")
        return

    game_key = f"{user.id}_{update.effective_chat.id}"
    
    existing_global_game = ACTIVE_CHASE_GAMES.get(game_key)
    existing_user_game = context.user_data.get("chase")
    
    if existing_global_game or existing_user_game:
        existing_state = existing_global_game or existing_user_game
        if existing_state and existing_state.get('active'):
            logger.info(f"Ending existing active game for user {user.id} before starting new one")
            end_chase_game(existing_state, context, 'abandoned')
        else:
            ACTIVE_CHASE_GAMES.pop(game_key, None)
            context.user_data.pop("chase", None)
            logger.info(f"Force cleaned up inactive game data for user {user.id}")

    state = _reset_level_state(user.id, raw_name, level=1)
    state["last_action_time"] = current_time
    context.user_data["chase"] = state
    
    ACTIVE_CHASE_GAMES[game_key] = state

    try:
        text = _format_chase_card(state, last_event=None)
        sent = await update.message.reply_text(
            text, parse_mode="HTML", reply_markup=_chase_keyboard()
        )

        state["message_id"] = sent.message_id
        state["chat_id"] = sent.chat_id
        
        logger.info(f"Started chase game for user {user.id} at level 1")
        
    except Exception as e:
        ACTIVE_CHASE_GAMES.pop(game_key, None)
        context.user_data.pop("chase", None)
        logger.error(f"Failed to start chase game: {e}")
        await update.message.reply_text("❌ Failed to start game. Please try again.")

def end_chase_game(state: dict, context: ContextTypes.DEFAULT_TYPE, game_outcome: str = 'completed', message: str = None) -> bool:
    """
    Unified function to properly end chase games and ensure cleanup.
    Always call this to end any chase game to prevent ghost sessions.
    
    Args:
        state: The game state dictionary
        context: Telegram context
        game_outcome: 'won', 'lost', 'quit', 'timeout', 'error'
        message: Optional final message
    
    Returns:
        bool: True if cleanup successful
    """
    try:
        logger.info(f"end_chase_game called with outcome: {game_outcome}")
        
        if not state:
            logger.warning("end_chase_game called with empty state")
            return False
            
        if state.get('player_id') and state.get('player_name'):
            logger.info(f"Recording chase game: player_id={state.get('player_id')}, player_name={state.get('player_name')}")
            balls_faced = state.get('balls_used', 0)  # Use actual cricket balls faced
            success = bot_instance.record_chase_game(
                telegram_id=state['player_id'],
                player_name=state['player_name'],
                chat_id=state.get('chat_id', 0),
                final_score=state.get('score', 0),
                max_level=state.get('level', 1),
                game_outcome=game_outcome,
                game_duration=balls_faced  # Now storing balls faced instead of time
            )
            
            if not success:
                logger.warning(f"Failed to record chase game for player {state['player_id']}")
            else:
                logger.info(f"Successfully recorded chase game for player {state['player_id']}")
                
                try:
                    score = state.get('score', 0)
                    level = state.get('level', 1)
                    won = game_outcome in ['completed', 'won']
                    
                    shard_reward = bot_instance.calculate_game_shard_reward('chase_game', score, level, won)
                    
                    if shard_reward > 0:
                        shard_success = bot_instance.award_shards(
                            state['player_id'],
                            shard_reward,
                            'chase_game',
                            f'Score: {score}, Level: {level}, Outcome: {game_outcome}'
                        )
                        
                        if shard_success:
                            logger.info(f"Awarded {shard_reward} shards to player {state['player_id']} for chase game")
                            state['shard_reward'] = shard_reward
                        else:
                            logger.warning(f"Failed to award shards to player {state['player_id']}")
                
                except Exception as e:
                    logger.error(f"Error awarding chase game shards: {e}")
                
                try:
                    daily_success = bot_instance.update_daily_leaderboard(
                        state['player_id'],
                        state['player_name'],
                        'chase',
                        state.get('score', 0),
                        state.get('level', 1),
                        won
                    )
                    
                    if daily_success:
                        logger.info(f"Updated daily chase leaderboard for player {state['player_id']}")
                    else:
                        logger.warning(f"Failed to update daily chase leaderboard for player {state['player_id']}")
                
                except Exception as e:
                    logger.error(f"Error updating daily chase leaderboard: {e}")
        else:
            logger.warning(f"Cannot record chase game - missing player data: player_id={state.get('player_id')}, player_name={state.get('player_name')}")
        
        game_key = f"{state['player_id']}_{state.get('chat_id', 0)}"
        ACTIVE_CHASE_GAMES.pop(game_key, None)
        
        context.user_data.pop("chase", None)
        
        state["active"] = False
        
        logger.info(f"Chase game ended: {game_outcome} for player {state.get('player_id')}")
        return True
        
    except Exception as e:
        logger.error(f"Error ending chase game: {e}")
        try:
            game_key = f"{state.get('player_id', 'unknown')}_{state.get('chat_id', 0)}"
            ACTIVE_CHASE_GAMES.pop(game_key, None)
            context.user_data.pop("chase", None)
            if state:
                state["active"] = False
        except:
            pass
        return False

async def chase_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle number taps for Run Chase Simulator with enhanced performance"""
    query = update.callback_query
    current_time = time.time()
    
    await query.answer()
    
    state = context.user_data.get("chase")
    if not state or not state.get("active"):
        game_key = f"{query.from_user.id}_{query.message.chat_id}"
        stale_game = ACTIVE_CHASE_GAMES.get(game_key)
        if stale_game:
            logger.info(f"Found stale game for user {query.from_user.id}, ending it properly")
            end_chase_game(stale_game, context, 'timeout')
        
        await query.edit_message_text(
            "❌ <b>Game Expired</b>\n\n"
            "🎮 This game session has ended.\n"
            "🆕 Start a new game with /chase",
            parse_mode="HTML"
        )
        return

    if query.message.message_id != state.get("message_id"):
        return

    if query.from_user.id != state["player_id"]:
        await query.answer("❌ This is not your game!", show_alert=True)
        return

    last_action = state.get("last_action_time", 0)
    time_diff = current_time - last_action
    
    if time_diff < 1.5:  # Increased from 0.5 to 1.5 seconds
        return  # Silent ignore rapid clicks
    
    state["last_action_time"] = current_time

    try:
        _, num_str = query.data.split(":")
        player_num = int(num_str)
        if player_num not in {1, 2, 3, 4, 5, 6}:
            logger.warning(f"Invalid player number: {player_num}")
            return
    except Exception as e:
        logger.warning(f"Failed to parse callback data: {query.data}, error: {e}")
        return

    bot_num = random.randint(1, 6)
    
    luck_roll = random.randint(1, 100)
    is_lucky = luck_roll <= 15  # 15% luck chance
    
    if player_num == bot_num:
        if is_lucky:
            last = (
                f"⚫ <b>Ball {state['balls_used']}:</b> "
                f"You <b>{player_num}</b> | Bot <b>{bot_num}</b>\n"
                f"🍀 <b>LUCKY ESCAPE!</b> +{player_num} runs (saved by luck!)"
            )
            state["score"] += player_num
        else:
            state["wickets_left"] -= 1
            last = (
                f"⚫ <b>Ball {state['balls_used']}:</b> "
                f"You <b>{player_num}</b> | Bot <b>{bot_num}</b>\n"
                f"💀 <b>WICKET!</b> Wickets left: {state['wickets_left']}"
            )
    else:
        state["score"] += player_num
        last = (
            f"⚫ <b>Ball {state['balls_used']}:</b> "
            f"You <b>{player_num}</b> | Bot <b>{bot_num}</b>\n"
            f"✅ <b>+{player_num} runs</b>"
        )

    state["balls_left"] -= 1
    state["balls_used"] += 1

    if player_num == bot_num and not is_lucky and state["wickets_left"] <= 0:
        state["active"] = False
        
        score = state.get('score', 0)
        level = state.get('level', 1)
        won = False  # Game lost
        shard_reward = bot_instance.calculate_game_shard_reward('chase_game', score, level, won)
        
        end = (
            f"\n\n❌ <b>ALL OUT!</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Final Score:</b> {state['score']}\n"
            f"🎯 <b>Target was:</b> {state['target']}\n"
            f"🏏 <b>Balls Used:</b> {state['balls_used']}\n"
            f"📈 <b>Level Reached:</b> {state['level']}"
        )
        
        if shard_reward > 0:
            end += f"\n💠 <b>Shards Earned:</b> +{shard_reward}"
        
        end += f"\n━━━━━━━━━━━━━━━━━━━━\n💪 <i>Try again with /chase to improve!</i>"
        
        text = f"{_format_chase_card(state, last)}{end}"
        try:
            await query.edit_message_text(text, parse_mode="HTML")
        except Exception as e:
            logger.warning(f"Failed to edit message on game end: {e}")
        
        end_chase_game(state, context, 'lost')
        return
    
    if player_num == bot_num and not is_lucky and state["wickets_left"] > 0:
        try:
            text = _format_chase_card(state, last)
            await query.edit_message_text(text, parse_mode="HTML", reply_markup=_chase_keyboard())
        except telegram_error.BadRequest as e:
            if "message is not modified" not in str(e).lower():
                logger.warning(f"Chase callback edit error: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error in chase callback: {e}")
        return

    if state["score"] >= state["target"]:
        if state["level"] == 10:
            score = state.get('score', 0)
            level = state.get('level', 1)
            won = True
            shard_reward = bot_instance.calculate_game_shard_reward('chase_game', score, level, won)
            
            end_chase_game(state, context, 'won')
            
            shard_info = ""
            if shard_reward > 0:
                shard_info = f"\n💠 <b>Shards Earned:</b> +{shard_reward}"
            
            text = (
                f"{_format_chase_card(state, last)}"
                f"\n\n🌟🏆 <b>ALL LEVELS COMPLETE!</b> 🏆🌟\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🔥 <b>ULTIMATE CHASE MASTER!</b> 🔥\n"
                f"📊 <b>Final Score:</b> {state['score']}\n"
                f"🏏 <b>Balls Used:</b> {state['balls_used']}\n"
                f"📈 <b>Levels Completed:</b> 10/10"
                f"{shard_info}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"👑 <i>You are a cricket legend!</i>"
            )
            try:
                await query.edit_message_text(text, parse_mode="HTML")
            except Exception as e:
                logger.warning(f"Failed to edit message on victory: {e}")
            return

        next_level = state["level"] + 1
        
        score = state.get('score', 0)
        level = state.get('level', 1)
        won = True
        shard_reward = bot_instance.calculate_game_shard_reward('chase_game', score, level, won)
        
        shard_info = ""
        if shard_reward > 0:
            shard_info = f"\n💠 <b>Shards Earned:</b> +{shard_reward}"
        
        text = (
            f"{_format_chase_card(state, last)}"
            f"\n\n🏆 <b>LEVEL {state['level']} COMPLETE!</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Score:</b> {state['score']} in {state['balls_used']} balls"
            f"{shard_info}\n\n"
            f"➡️ <b>Next Level {next_level}:</b> "
            f"{LEVELS[next_level]['target']} runs in {LEVELS[next_level]['balls']} balls"
        )

        new_state = _reset_level_state(state["player_id"], state["player_name"], next_level)
        new_state["message_id"] = query.message.message_id
        new_state["chat_id"] = query.message.chat_id
        new_state["last_action_time"] = current_time
        new_state["start_time"] = state.get("start_time", time.time())  # Preserve original start time
        
        context.user_data["chase"] = new_state
        game_key = f"{state['player_id']}_{query.message.chat_id}"
        ACTIVE_CHASE_GAMES[game_key] = new_state

        try:
            await query.edit_message_text(text, parse_mode="HTML", reply_markup=_chase_keyboard())
        except Exception as e:
            logger.warning(f"Failed to edit message on level advance: {e}")
        return

    if state["balls_left"] == 0 and state["score"] < state["target"]:
        score = state.get('score', 0)
        level = state.get('level', 1)
        won = False  # Target not reached
        shard_reward = bot_instance.calculate_game_shard_reward('chase_game', score, level, won)
        
        end_chase_game(state, context, 'lost')
        
        end = (
            f"\n\n💀 <b>BALLS FINISHED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Final Score:</b> {state['score']}/{state['target']}\n"
            f"🏏 <b>Balls Used:</b> {state['balls_used']}\n"
            f"🎯 <b>Target:</b> {state['target']}\n"
            f"📈 <b>Level Reached:</b> {state['level']}"
        )
        
        if shard_reward > 0:
            end += f"\n💠 <b>Shards Earned:</b> +{shard_reward}"
        
        end += f"\n━━━━━━━━━━━━━━━━━━━━\n💪 <i>Try again with /chase!</i>"
        
        try:
            await query.edit_message_text(f"{_format_chase_card(state, last)}{end}", parse_mode="HTML")
        except Exception as e:
            logger.warning(f"Failed to edit message on game over: {e}")
        return

    try:
        text = _format_chase_card(state, last)
        
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=_chase_keyboard())
        
        logger.debug(f"Chase game updated - Score: {state['score']}, Balls left: {state['balls_left']}")
        
    except telegram_error.BadRequest as e:
        error_msg = str(e).lower()
        if "message is not modified" in error_msg:
            logger.warning("Message content identical - forcing update with timestamp")
            try:
                fresh_text = _format_chase_card(state, last)
                await query.edit_message_text(fresh_text, parse_mode="HTML", reply_markup=_chase_keyboard())
            except Exception as retry_error:
                logger.error(f"Retry failed: {retry_error}")
        else:
            logger.warning(f"Chase callback edit error: {e}")
    except Exception as e:
        logger.error(f"Critical error updating chase game: {e}")
        end_chase_game(state, context, 'error')
        try:
            await query.edit_message_text(
                f"{_format_chase_card(state, last)}\n\n"
                f"❌ <b>Game Error</b>\n"
                f"Game ended due to technical issue. Start new game with /chase",
                parse_mode="HTML"
            )
        except:
            pass  # Final fallback

@check_banned
async def chase_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show chase game statistics - Public command with personal stats"""
    user = update.effective_user
    
    cleanup_expired_games()
    
    bot_instance.create_or_update_player(user.id, user.username or "", user.full_name or user.first_name or f"User{user.id}")
    
    personal_stats = {'games_played': 0, 'wins': 0, 'avg_score': 0, 'high_score': 0}
    
    overall_stats = bot_instance.get_chase_game_stats()
    
    message = (
        f"🏏 <b>ARENA OF CHAMPIONS - CHASE STATISTICS</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"👤 <b>YOUR STATS</b>\n"
        f"🎮 <b>Games Played:</b> {personal_stats.get('total_games', 0)}\n"
        f"🏆 <b>Games Won:</b> {personal_stats.get('games_won', 0)}\n"
        f"📈 <b>Win Rate:</b> {personal_stats.get('win_rate', 0):.1f}%\n"
        f"🥇 <b>Best Score:</b> {personal_stats.get('high_score', 0)}\n"
        f"⚡ <b>Average Score:</b> {personal_stats.get('avg_score', 0):.1f}\n\n"
        f"🌍 <b>GLOBAL STATS</b>\n"
        f"📊 <b>Total Games:</b> {overall_stats.get('total_games', 0)}\n"
        f"👥 <b>Total Players:</b> {overall_stats.get('total_players', 0)}\n"
        f"🏆 <b>High Score:</b> {overall_stats.get('high_score', 0)}\n"
        f"📈 <b>Average Score:</b> {overall_stats.get('avg_score', 0):.1f}\n"
        f"🕐 <b>Games (24h):</b> {overall_stats.get('games_24h', 0)}\n"
        f"📅 <b>Games (7d):</b> {overall_stats.get('games_7d', 0)}\n\n"
        f"🎯 <b>Want to play?</b> Use /chase to start!"
    )
    
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    keyboard = [
        [InlineKeyboardButton("🏏 Play Chase", callback_data="start_chase"),
         InlineKeyboardButton("🏆 Leaderboard", callback_data="chase_leaderboard")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(message, parse_mode="HTML", reply_markup=reply_markup)

@check_banned
async def leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display the Chase Game Leaderboard - Modern Block Style"""
    try:
        leaderboard = bot_instance.get_chase_leaderboard(10)

        if not leaderboard:
            await update.message.reply_text(
                "🏏 <b>CHASE LEADERBOARD</b> 🏏\n\n"
                "📊 No chase games played yet!\n"
                "🎮 Be the first to start with /chase",
                parse_mode='HTML'
            )
            return

        message = "🏏 <b>ARENA OF CHAMPIONS - CHASE LEADERBOARD</b> 🏏\n"
        message += "━━━━━━━━━━━━━━━━━━━━\n\n"

        rank_emojis = {1: "🥇", 2: "🥈", 3: "🥉"}

        for i, player in enumerate(leaderboard, 1):
            if i in rank_emojis:
                rank_display = rank_emojis[i]
            else:
                rank_display = f"{i}."  # Simple number format
            
            name = H(player.get('display_name', 'Unknown'))
            level = player.get('highest_level', 0) or 0
            runs = player.get('runs_scored', 0) or 0
            balls = player.get('balls_faced', 0) or 0  # Ensure not None
            strike_rate = player.get('strike_rate', 0.0) or 0.0

            sr_display = f" | <b>SR:</b> {strike_rate:.0f}" if strike_rate > 0 and balls > 0 else ""

            message += (
                f"{rank_display} <b>{name}</b>\n"
                f"   ┗ <b>Level:</b> {level} | "
                f"<b>Runs:</b> {runs}({balls}){sr_display}\n\n"
            )

        message += "━━━━━━━━━━━━━━━━━━━━━━━\n"
        message += "📋 <b>Ranking Rules:</b>\n"
        message += "1️⃣ Highest Level\n"
        message += "2️⃣ Highest Runs at that Level\n"
        message += "3️⃣ Fewer Balls Faced\n"
        message += "━━━━━━━━━━━━━━━━━━━━━━━\n"
        message += "🎯 Play <b>/chase</b> & climb the ranks!"

        await safe_send(update.message.reply_text, message, parse_mode='HTML')

    except Exception as e:
        logger.error(f"Error in leaderboard command: {e}")
        await update.message.reply_text(
            "❌ <b>Error loading leaderboard</b>\n\n"
            "Please try again later.",
            parse_mode='HTML'
        )

async def cleanup_chase_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin command to force cleanup all chase games"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ Admin only command!")
        return
    
    before_count = len(ACTIVE_CHASE_GAMES)
    games_recorded = 0
    
    games_to_end = list(ACTIVE_CHASE_GAMES.items())  # Create a copy to iterate over
    for game_key, game_state in games_to_end:
        try:
            if game_state and game_state.get('active'):
                class MockContext:
                    def __init__(self):
                        self.user_data = {}
                
                mock_context = MockContext()
                logger.info(f"Admin cleanup: ending game for key {game_key}")
                success = end_chase_game(game_state, mock_context, 'admin_cleanup')
                if success:
                    games_recorded += 1
        except Exception as e:
            logger.error(f"Error ending game {game_key} during cleanup: {e}")
    
    ACTIVE_CHASE_GAMES.clear()
    
    await update.message.reply_text(
        f"🧹 <b>FORCE CLEANUP COMPLETE!</b>\n\n"
        f"✅ Found {before_count} active chase games\n"
        f"✅ Properly recorded {games_recorded} games in database\n"
        f"✅ Cleared all global game states\n\n"
        f"⚠️ <b>IMPORTANT:</b> Users with 'game already active' errors should:\n"
        f"• Wait 30 seconds and try /chase again\n"
        f"• Contact admin if issue persists\n\n"
        f"💡 <b>All games have been properly ended and recorded!</b>",
        parse_mode="HTML"
    )

async def admin_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show comprehensive admin status with hierarchy and potential conflicts"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can view admin status!", parse_mode='HTML')
        return
    
    user_level = bot_instance.get_admin_level(update.effective_user.id)
    
    message = "🔍 <b>ADMIN STATUS & HIERARCHY</b> 🛡️\n\n"
    message += "━━━━━━━━━━━━━━━━━━━━\n"
    
    if user_level == "SUPER_ADMIN":
        message += f"👑 <b>Your Level:</b> SUPER ADMIN (Creator)\n"
        message += f"🆔 <b>Your ID:</b> {update.effective_user.id}\n\n"
    elif user_level == "ENV_ADMIN":
        message += f"🌟 <b>Your Level:</b> Environment Admin\n"
        message += f"🆔 <b>Your ID:</b> {update.effective_user.id}\n\n"
    elif user_level == "DB_ADMIN":
        message += f"🗄️ <b>Your Level:</b> Database Admin\n"
        message += f"🆔 <b>Your ID:</b> {update.effective_user.id}\n\n"
    
    message += "📊 <b>ADMIN HIERARCHY</b>\n\n"
    
    message += f"1️⃣ <b>Super Admin (Creator)</b>\n"
    message += f"   🆔 ID: {bot_instance.super_admin_id}\n"
    message += f"   🔑 Source: Environment (SUPER_ADMIN_ID)\n"
    message += f"   ⚡ Authority: Highest\n\n"
    
    env_admins = bot_instance.admin_ids
    if env_admins:
        message += f"2️⃣ <b>Environment Admins ({len(env_admins)})</b>\n"
        for admin_id in env_admins[:3]:  # Show first 3
            level = bot_instance.get_admin_level(admin_id)
            message += f"   🆔 {admin_id} ({level})\n"
        if len(env_admins) > 3:
            message += f"   ... and {len(env_admins) - 3} more\n"
        message += f"   🔑 Source: Environment (ADMIN_IDS)\n"
        message += f"   ⚡ Authority: Permanent\n\n"
    else:
        message += f"2️⃣ <b>Environment Admins (0)</b>\n"
        message += f"   📝 None configured in ADMIN_IDS\n\n"
    
    try:
        db_admins = bot_instance.get_all_admins()
        if db_admins:
            message += f"3️⃣ <b>Database Admins ({len(db_admins)})</b>\n"
            for admin in db_admins[:3]:  # Show first 3
                username = f"@{admin['username']}" if admin['username'] else "N/A"
                level = bot_instance.get_admin_level(admin['telegram_id'])
                message += f"   🆔 {admin['telegram_id']} ({level})\n"
                message += f"      👤 {admin['display_name']} ({username})\n"
            if len(db_admins) > 3:
                message += f"   ... and {len(db_admins) - 3} more\n"
            message += f"   🔑 Source: Database (admins table)\n"
            message += f"   ⚡ Authority: Dynamic\n\n"
        else:
            message += f"3️⃣ <b>Database Admins (0)</b>\n"
            message += f"   📝 None added to database\n\n"
    except Exception as e:
        message += f"3️⃣ <b>Database Admins (?)</b>\n"
        message += f"   ❌ Database error: {str(e)[:50]}...\n\n"
    
    message += "🔍 <b>CONFLICT DETECTION</b>\n"
    
    conflicts_found = False
    total_unique_admins = set()
    total_unique_admins.add(bot_instance.super_admin_id)
    total_unique_admins.update(env_admins)
    
    try:
        db_admin_ids = {admin['telegram_id'] for admin in db_admins} if db_admins else set()
        total_unique_admins.update(db_admin_ids)
        
        db_only = db_admin_ids - set(env_admins) - {bot_instance.super_admin_id}
        if db_only:
            message += f"⚠️ <b>DB-Only Admins:</b> {len(db_only)} users\n"
            message += f"   (Admins in database but not in environment)\n"
            conflicts_found = True
        
        env_not_in_db = set(env_admins) - db_admin_ids
        if env_not_in_db:
            message += f"🔄 <b>Env Not in DB:</b> {len(env_not_in_db)} users\n"
            message += f"   (Environment admins not yet in database)\n"
            conflicts_found = True
        
    except Exception as e:
        message += f"❌ <b>Conflict Check Failed:</b> DB error\n"
        conflicts_found = True
    
    if not conflicts_found:
        message += "✅ <b>No conflicts detected!</b>\n"
        message += "   All admin sources are synchronized\n"
    
    message += "\n━━━━━━━━━━━━━━━━━━━━\n"
    message += f"📊 <b>Total Unique Admins:</b> {len(total_unique_admins)}\n"
    message += f"🔧 <b>Auto-Sync:</b> {'✅ Enabled' if env_admins else '⚠️ No env admins'}"
    
    await update.message.reply_text(message, parse_mode='HTML')

async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin command to broadcast messages to all registered users"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can broadcast messages!", parse_mode='HTML')
        return
    
    if update.message.reply_to_message:
        reply_msg = update.message.reply_to_message
        
        if reply_msg.text:
            broadcast_text = reply_msg.text
        elif reply_msg.caption:
            broadcast_text = reply_msg.caption
        else:
            await update.message.reply_text("❌ Can only broadcast text messages or media with captions!")
            return
            
        broadcast_media = None
        if reply_msg.photo:
            broadcast_media = {'type': 'photo', 'media': reply_msg.photo[-1].file_id}
        elif reply_msg.video:
            broadcast_media = {'type': 'video', 'media': reply_msg.video.file_id}
        elif reply_msg.document:
            broadcast_media = {'type': 'document', 'media': reply_msg.document.file_id}
            
    else:
        if not context.args:
            await update.message.reply_text(
                "📢 <b>BROADCAST COMMAND</b>\n\n"
                "<b>Usage:</b>\n"
                "• /broadcast &lt;message&gt; - Broadcast text\n"
                "• Reply to any message with /broadcast - Forward that message\n\n"
                "<b>Examples:</b>\n"
                "• /broadcast Server maintenance in 1 hour\n"
                "• /broadcast 🎉 New features added!\n"
                "• Reply to photo/video with /broadcast",
                parse_mode='HTML'
            )
            return
            
        broadcast_text = ' '.join(context.args)
        broadcast_media = None
    
    try:
        with bot_instance.get_db_cursor() as cursor_result:
            if cursor_result is None:
                await update.message.reply_text("❌ Database error occurred!")
                return
            cursor, conn = cursor_result
            
            cursor.execute("SELECT telegram_id FROM players")
            users = cursor.fetchall()
            
            cursor.execute("""
                SELECT chat_id FROM chats 
                WHERE chat_type IN ('group', 'supergroup') 
                AND is_active = TRUE
            """)
            groups = cursor.fetchall()
            
            all_recipients = [(row[0], 'user') for row in users] + [(row[0], 'group') for row in groups]
            
            if not all_recipients:
                await update.message.reply_text("❌ No recipients found! (No users or groups)")
                return
            
    except Exception as e:
        logger.error(f"Database error in broadcast: {e}")
        await update.message.reply_text("❌ Database error occurred!")
        return
    
    user_count = len(users)
    group_count = len(groups)
    total_recipients = len(all_recipients)
    preview_text = InputValidator.truncate_with_ellipsis(broadcast_text, 100)
    
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    keyboard = [
        [
            InlineKeyboardButton("✅ CONFIRM BROADCAST", callback_data=f"broadcast_confirm_{total_recipients}"),
            InlineKeyboardButton("❌ CANCEL", callback_data="broadcast_cancel")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    media_text = "📎 <b>Media:</b> Included\n" if broadcast_media else ""
    
    confirm_msg = await update.message.reply_text(
        f"⚠️ <b>BROADCAST CONFIRMATION REQUIRED</b>\n\n"
        f"👥 <b>Users:</b> {user_count}\n"
        f"👨‍👩‍👧‍👦 <b>Groups:</b> {group_count}\n"
        f"📊 <b>Total Recipients:</b> {total_recipients}\n"
        f"📝 <b>Message Preview:</b>\n<code>{preview_text}</code>\n"
        f"{media_text}\n"
        f"🚨 <b>This will send to ALL recipients!</b>\n"
        f"⚠️ <i>Click CONFIRM to proceed or CANCEL to abort.</i>",
        parse_mode='HTML',
        reply_markup=reply_markup
    )
    
    context.user_data['pending_broadcast'] = {
        'text': broadcast_text,
        'media': broadcast_media,
        'recipients': all_recipients,
        'user_count': user_count,
        'group_count': group_count,
        'confirm_msg_id': confirm_msg.message_id
    }
    
    return  # Wait for confirmation

async def broadcast_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle broadcast confirmation callbacks"""
    query = update.callback_query
    await query.answer()
    
    if not bot_instance.is_admin(query.from_user.id):
        await query.edit_message_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Admin access required.", parse_mode='HTML')
        return
    
    if query.data == "broadcast_cancel":
        await query.edit_message_text(
            "❌ <b>Broadcast Cancelled</b>\n\n"
            "📢 No messages were sent.",
            parse_mode='HTML'
        )
        context.user_data.pop('pending_broadcast', None)
        return
    
    if query.data.startswith("broadcast_confirm_"):
        broadcast_data = context.user_data.get('pending_broadcast')
        if not broadcast_data:
            await query.edit_message_text("❌ Broadcast data not found. Please try again.")
            return
        
        broadcast_text = broadcast_data['text']
        broadcast_media = broadcast_data['media']
        all_recipients = broadcast_data['recipients']
        user_count = broadcast_data['user_count']
        group_count = broadcast_data['group_count']
        
        await query.edit_message_text(
            f"📢 <b>BROADCASTING...</b>\n\n"
            f"👥 Users: {user_count} | 👨‍👩‍👧‍👦 Groups: {group_count}\n"
            f"📊 Total: {len(all_recipients)} recipients\n\n"
            f"🔄 <i>Sending messages...</i>",
            parse_mode='HTML'
        )
    
    success_count = 0
    failed_count = 0
    blocked_count = 0
    user_success = 0
    group_success = 0
    error_details = {}
    
    for chat_id, chat_type in all_recipients:
        try:
            if broadcast_media:
                if broadcast_media['type'] == 'photo':
                    await context.bot.send_photo(
                        chat_id=chat_id,
                        photo=broadcast_media['media'],
                        caption=f"📢 <b>ANNOUNCEMENT</b>\n\n{broadcast_text}",
                        parse_mode='HTML'
                    )
                elif broadcast_media['type'] == 'video':
                    await context.bot.send_video(
                        chat_id=chat_id,
                        video=broadcast_media['media'],
                        caption=f"📢 <b>ANNOUNCEMENT</b>\n\n{broadcast_text}",
                        parse_mode='HTML'
                    )
                elif broadcast_media['type'] == 'document':
                    await context.bot.send_document(
                        chat_id=chat_id,
                        document=broadcast_media['media'],
                        caption=f"📢 <b>ANNOUNCEMENT</b>\n\n{broadcast_text}",
                        parse_mode='HTML'
                    )
            else:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"📢 <b>ANNOUNCEMENT</b>\n\n{broadcast_text}",
                    parse_mode='HTML'
                )
            success_count += 1
            if chat_type == 'user':
                user_success += 1
            else:
                group_success += 1
            
        except telegram_error.Forbidden as e:
            if "blocked" in str(e).lower() or "forbidden" in str(e).lower():
                blocked_count += 1
                logger.info(f"Chat {chat_id} blocked the bot")
            else:
                failed_count += 1
                error_type = "Forbidden Access"
                error_details[error_type] = error_details.get(error_type, 0) + 1
                
        except telegram_error.BadRequest as e:
            failed_count += 1
            if "chat not found" in str(e).lower():
                error_type = "Chat Not Found"
            elif "user is deactivated" in str(e).lower():
                error_type = "User Deactivated"
            else:
                error_type = "Bad Request"
            error_details[error_type] = error_details.get(error_type, 0) + 1
            logger.warning(f"BadRequest for chat {chat_id}: {e}")
            
        except telegram_error.RetryAfter as e:
            failed_count += 1
            error_type = "Rate Limited"
            error_details[error_type] = error_details.get(error_type, 0) + 1
            logger.warning(f"Rate limited, retry after {e.retry_after}s")
            
            await asyncio.sleep(e.retry_after + 1)
            
        except Exception as e:
            failed_count += 1
            error_type = f"Other: {type(e).__name__}"
            error_details[error_type] = error_details.get(error_type, 0) + 1
            logger.warning(f"Failed to broadcast to chat {chat_id}: {e}")
            
        batch_size = 20  # Messages per batch
        if (success_count + failed_count) % batch_size == 0:
            await asyncio.sleep(1.0)
        else:
            await asyncio.sleep(0.1)
    
    total_recipients = len(all_recipients)
    preview_text = InputValidator.truncate_with_ellipsis(broadcast_text, 50)
    summary_text = (
        f"📢 <b>BROADCAST COMPLETE</b>\n\n"
        f"✅ <b>Total Sent:</b> {success_count} recipients\n"
        f"👥 <b>Users:</b> {user_success}/{user_count}\n"
        f"👨‍👩‍👧‍👦 <b>Groups:</b> {group_success}/{group_count}\n"
        f"❌ <b>Failed:</b> {failed_count}\n"
        f"🚫 <b>Blocked:</b> {blocked_count}\n"
        f"📊 <b>Success Rate:</b> {(success_count/total_recipients*100):.1f}%\n"
    )
    
    if error_details:
        summary_text += "\n<b>🔍 Error Types:</b>\n"
        for error_type, count in error_details.items():
            summary_text += f"• {error_type}: {count}\n"
    
    summary_text += f"\n📝 <b>Message:</b> {preview_text}"
    
    await query.edit_message_text(summary_text, parse_mode='HTML')
    
    context.user_data.pop('pending_broadcast', None)
    
    logger.info(f"Broadcast completed: {success_count}/{total_recipients} successful")

async def emojis_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show emoji guide for achievements"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can view emoji guide!", parse_mode='HTML')
        return
    
    emoji_guide = """
🎯 <b>ACHIEVEMENT EMOJI GUIDE</b> 🏆

<b>4 Main Achievement Categories:</b>

🏆 <b>WINNER</b> - Contains "winner"
• S7 Winner → 🏆
• Tournament Winner → 🏆
• Best Winner → 🏆

🟧 <b>ORANGE CAP</b> - Contains "orange"
• Orange Cap → 🟧
• Orange Cap S7 → 🟧
• Best Orange → 🟧

🟪 <b>PURPLE CAP</b> - Contains "purple" 
• Purple Cap → 🟪
• Purple Cap T20 → 🟪
• Best Purple → 🟪

🏅 <b>MVP</b> - Contains "mvp"
• MVP S7 → 🏅
• MVP Award → 🏅
• Best MVP → 🏅

📝 <b>USAGE EXAMPLES:</b>
• /addachievement @player "S7 Winner"
• /bulkward "Orange Cap" @player1 @player2
• /settitle @player "MVP Champion 🏅"

<b>💡 TIP:</b> Use keywords "winner", "orange", "purple", "mvp" to get the right emojis automatically!

<b>💡 TIP:</b> Use keywords like "MVP", "Orange", "Purple", "Strike Rate", "Six", "Economy" to get the right emojis!
"""
    await update.message.reply_text(emoji_guide, parse_mode='HTML')

async def add_admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add new admin (Super Admin only)"""
    if not bot_instance.is_super_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n👑 Only Super Admin can add admins!", parse_mode='HTML')
        return
    
    if len(context.args) < 1:
        await update.message.reply_text(
            "❌ <b>INCORRECT USAGE!</b>\n\n"
            "📝 <b>Format:</b> /addadmin &lt;@username or user_id&gt;\n\n"
            "💡 <b>Examples:</b>\n"
            "• /addadmin @john_cricket\n"
            "• /addadmin 123456789",
            parse_mode='HTML'
        )
        return
    
    target_identifier = context.args[0]
    
    player = bot_instance.find_player_by_identifier(target_identifier)
    
    if not player:
        await update.message.reply_text(
            f"❌ <b>USER NOT FOUND!</b>\n\n"
            f"🔍 Could not find: {target_identifier}\n\n"
            f"💡 User must have interacted with the bot first!",
            parse_mode='HTML'
        )
        return
    
    if bot_instance.add_admin(player['telegram_id'], player['username'], player['display_name'], update.effective_user.id):
        await update.message.reply_text(
            f"✅ <b>ADMIN ADDED!</b> 🛡️\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"👤 <b>New Admin:</b> {player['display_name']}\n"
            f"📱 <b>Username:</b> @{player['username'] or 'N/A'}\n"
            f"🆔 <b>User ID:</b> {player['telegram_id']}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🎉 <b>Admin privileges granted!</b> 🔐",
            parse_mode='HTML'
        )
    else:
        await update.message.reply_text("❌ <b>FAILED!</b> User might already be an admin.", parse_mode='HTML')

async def remove_admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove admin (Super Admin only)"""
    if not bot_instance.is_super_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n👑 Only Super Admin can remove admins!", parse_mode='HTML')
        return
    
    if len(context.args) < 1:
        await update.message.reply_text(
            "❌ <b>INCORRECT USAGE!</b>\n\n"
            "📝 <b>Format:</b> /rmadmin &lt;@username or user_id&gt;",
            parse_mode='HTML'
        )
        return
    
    target_identifier = context.args[0]
    
    player = bot_instance.find_player_by_identifier(target_identifier)
    
    if not player:
        await update.message.reply_text(f"❌ <b>USER NOT FOUND!</b> {target_identifier}", parse_mode='HTML')
        return
    
    if player['telegram_id'] == bot_instance.super_admin_id:
        await update.message.reply_text("❌ <b>CANNOT REMOVE SUPER ADMIN!</b> 👑", parse_mode='HTML')
        return
    
    if bot_instance.remove_admin(player['telegram_id']):
        await update.message.reply_text(
            f"✅ <b>ADMIN REMOVED!</b> 🗑️\n\n"
            f"👤 <b>Removed:</b> {player['display_name']}\n"
            f"🔓 <b>Admin privileges revoked!</b>",
            parse_mode='HTML'
        )
    else:
        await update.message.reply_text("❌ <b>FAILED!</b> User might not be an admin.", parse_mode='HTML')

async def list_admins_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all admins"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can view admin list!", parse_mode='HTML')
        return
    
    admins = bot_instance.get_all_admins()
    
    message = "🛡️ <b>ARENA OF CHAMPIONS ADMIN LIST</b> 👑\n\n"
    message += "━━━━━━━━━━━━━━━━━━━━\n"
    
    message += f"👑 <b>SUPER ADMIN (Creator)</b>\n"
    message += f"🆔 ID: {bot_instance.super_admin_id}\n\n"
    
    if admins:
        message += f"🛡️ <b>ADMINS ({len(admins)}):</b>\n\n"
        for i, admin in enumerate(admins, 1):
            username = f"@{H(admin['username'])}" if admin['username'] else "N/A"
            message += f"<b>{i}.</b> {H(admin['display_name'])}\n"
            message += f"   📱 {username}\n"
            message += f"   🆔 {admin['telegram_id']}\n\n"
    else:
        message += "📝 <b>No additional admins added yet.</b>\n\n"
    
    message += "━━━━━━━━━━━━━━━━━━━━\n"
    message += f"📊 <b>Total Admins:</b> {len(admins) + 1}"
    
    await update.message.reply_text(message, parse_mode='HTML')

async def bulk_award_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Bulk award achievement to multiple players"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can use bulk award!", parse_mode='HTML')
        return
    
    if len(context.args) < 2:
        await update.message.reply_text(
            "❌ <b>INCORRECT USAGE!</b>\n\n"
            "📝 <b>Format:</b> /bulkward \"&lt;achievement&gt;\" &lt;player1&gt; &lt;player2&gt; ...\n\n"
            "💡 <b>Examples:</b>\n"
            "• /bulkward \"S7 Winner\" 123456789 987654321\n"
            "• /bulkward MVP @john @jane\n"
            "• /bulkward \"Orange Cap\" @player1 123456789",
            parse_mode='HTML'
        )
        return
    
    message_text = update.message.text
    command_parts = message_text.split(maxsplit=1)  # Split only once to get command and rest
    
    if len(command_parts) < 2:
        await update.message.reply_text("❌ <b>No arguments provided!</b>", parse_mode='HTML')
        return
    
    args_text = command_parts[1]  # Everything after /bulkward
    
    if args_text.startswith('"'):
        end_quote = args_text.find('"', 1)
        if end_quote == -1:
            await update.message.reply_text(
                "❌ <b>MISSING CLOSING QUOTE!</b>\n\n"
                "📝 Use: /bulkward \"Achievement Name\" player1 player2",
                parse_mode='HTML'
            )
            return
        
        achievement = args_text[1:end_quote]  # Extract quoted achievement
        remaining_args = args_text[end_quote+1:].strip()  # Get players after quote
        player_identifiers = remaining_args.split() if remaining_args else []
        
    else:
        parts = args_text.split()
        achievement = parts[0]
        player_identifiers = parts[1:]
    
    if not player_identifiers:
        await update.message.reply_text("❌ <b>NO PLAYERS SPECIFIED!</b>", parse_mode='HTML')
        return
    
    successful = []
    not_registered = []
    failed = []
    
    for identifier in player_identifiers:
        player = bot_instance.find_player_by_identifier(identifier)
        if player and bot_instance.add_achievement(player['id'], achievement, update.effective_user.id, player.get('username')):
            successful.append(player['display_name'])
        elif not player:
            not_registered.append(identifier)
        else:
            failed.append(identifier)
    
    message = f"🏆 <b>BULK AWARD RESULTS</b>\n\n"
    message += f"🎖️ <b>Achievement:</b> {achievement}\n\n"
    
    if successful:
        message += f"✅ <b>AWARDED SUCCESSFULLY ({len(successful)}):</b>\n"
        for name in successful:
            message += f"• {name}\n"
        message += "\n"
    
    if not_registered:
        message += f"🚫 <b>NOT REGISTERED ({len(not_registered)}):</b>\n"
        for identifier in not_registered:
            message += f"• {identifier}\n"
        message += "💡 <i>These users need to start the bot first with /start!</i>\n\n"
    
    if failed:
        message += f"❌ <b>FAILED ({len(failed)}):</b>\n"
        for identifier in failed:
            message += f"• {identifier}\n"
        message += "\n"
    
    success_rate = len(successful)
    total_users = len(player_identifiers)
    message += f"📊 <b>Success Rate:</b> {success_rate}/{total_users}"
    
    if not_registered:
        message += f"\n\n📢 <b>Tip:</b> Ask unregistered users to use /start first!"
    
    await update.message.reply_text(message, parse_mode='HTML')

async def reset_player_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset player data (Admin and Super Admin only)"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only Admins can reset players!", parse_mode='HTML')
        return
    
    if len(context.args) < 1:
        await update.message.reply_text(
            "❌ <b>INCORRECT USAGE!</b>\n\n"
            "📝 <b>Format:</b> /resetplayer &lt;player&gt;\n"
            "⚠️ <b>WARNING:</b> This will delete ALL achievements and title!",
            parse_mode='HTML'
        )
        return
    
    player_identifier = context.args[0]
    player = bot_instance.find_player_by_identifier(player_identifier)
    
    if not player:
        await update.message.reply_text(f"❌ <b>PLAYER NOT FOUND!</b> {player_identifier}", parse_mode='HTML')
        return
    
    if bot_instance.reset_player_data(player['id'], update.effective_user.id):
        await update.message.reply_text(
            f"✅ <b>PLAYER RESET COMPLETE!</b> 🔄\n\n"
            f"👤 <b>Player:</b> {player['display_name']}\n"
            f"🗑️ <b>All achievements deleted</b>\n"
            f"👑 <b>Title removed</b>\n\n"
            f"📝 <b>Data backed up for recovery</b> ✅",
            parse_mode='HTML'
        )
    else:
        await update.message.reply_text("❌ <b>RESET FAILED!</b> Please try again.", parse_mode='HTML')

async def add_achievement_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add achievement to player (Admin only)"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can add achievements!", parse_mode='HTML')
        return
    
    if len(context.args) < 2:
        await update.message.reply_text(
            "❌ <b>INCORRECT USAGE!</b>\n\n"
            "📝 <b>Correct Format:</b>\n"
            "/addachievement &lt;player&gt; &lt;achievement&gt;\n\n"
            "💡 <b>Examples:</b>\n"
            "• /addachievement @user1 MVP\n"
            "• /addachievement @player123 Purple Cap\n"
            "• /addachievement 123456789 Most Sixes\n\n"
            "ℹ️ Use /help for detailed guide!",
            parse_mode='HTML'
        )
        return
    
    player_identifier = context.args[0]
    achievement = ' '.join(context.args[1:])
    
    player = bot_instance.find_player_by_identifier(player_identifier)
    
    if not player:
        await update.message.reply_text(
            f"❌ <b>USER NOT REGISTERED!</b>\n\n"
            f"👤 <b>Player:</b> {player_identifier}\n"
            f"📋 <b>Status:</b> Not registered with the bot\n\n"
            f"💡 <b>SOLUTION:</b>\n"
            f"Ask the user to start the bot first:\n"
            f"• They need to send /start to the bot\n"
            f"• After registration, you can award achievements\n\n"
            f"🎯 <b>Note:</b> Only registered users can receive achievements!",
            parse_mode='HTML'
        )
        return
    
    if bot_instance.add_achievement(player['id'], achievement, update.effective_user.id, player.get('username')):
        emoji = bot_instance.get_achievement_emoji(achievement)
        await update.message.reply_text(
            f"✅ <b>ACHIEVEMENT AWARDED!</b> 🎉\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🏆 <b>Award:</b> {achievement.title()}\n"
            f"👤 <b>Player:</b> {player['display_name']}\n"
            f"🎖️ <b>Badge:</b> {emoji}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🎯 <b>Achievement successfully recorded!</b> 📝",
            parse_mode='HTML'
        )
    else:
        await update.message.reply_text("❌ <b>FAILED!</b> Unable to add achievement. Please try again.", parse_mode='HTML')

async def remove_achievement_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove achievement from player (Admin only)"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can remove achievements!", parse_mode='HTML')
        return
    
    if len(context.args) < 2:
        await update.message.reply_text(
            "❌ <b>INCORRECT USAGE!</b>\n\n"
            "📝 <b>Correct Format:</b>\n"
            "/removeachievement &lt;player&gt; &lt;achievement&gt;\n\n"
            "💡 <b>Examples:</b>\n"
            "• /removeachievement @user1 MVP\n"
            "• /removeachievement @player123 Purple Cap\n\n"
            "ℹ️ Use /help for detailed guide!",
            parse_mode='HTML'
        )
        return
    
    player_identifier = context.args[0]
    achievement = ' '.join(context.args[1:])
    
    player = bot_instance.find_player_by_identifier(player_identifier)
    
    if not player:
        await update.message.reply_text(
            f"❌ <b>PLAYER NOT FOUND!</b>\n\n"
            f"🔍 Could not find: {player_identifier}",
            parse_mode='HTML'
        )
        return
    
    achievements = bot_instance.get_player_achievements(player['id'])
    current_count = 0
    for ach_name, count in achievements:
        if ach_name.lower() == achievement.lower():
            current_count = count
            break
    
    if current_count == 0:
        await update.message.reply_text(
            f"❌ <b>NOT FOUND!</b>\n\n"
            f"🔍 Achievement {achievement} not found for this player!",
            parse_mode='HTML'
        )
        return
    
    if bot_instance.remove_achievement(player['id'], achievement):
        emoji = bot_instance.get_achievement_emoji(achievement)
        await update.message.reply_text(
            f"✅ <b>ACHIEVEMENT REMOVED!</b> 🗑️\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🏆 <b>Award:</b> {achievement.title()}\n"
            f"👤 <b>Player:</b> {player['display_name']}\n"
            f"🎖️ <b>Badge:</b> {emoji}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📝 <b>Achievement successfully removed!</b>",
            parse_mode='HTML'
        )
    else:
        await update.message.reply_text(
            f"❌ <b>NOT FOUND!</b>\n\n"
            f"🔍 Achievement {achievement} not found for this player!",
            parse_mode='HTML'
        )

async def set_title_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set player title (Admin only)"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can set titles!", parse_mode='HTML')
        return
    
    if len(context.args) < 2:
        await update.message.reply_text(
            "❌ **INCORRECT USAGE!**\n\n"
            "📝 **Correct Format:**\n"
            "`/settitle <player> <title>`\n\n"
            "💡 **Examples:**\n"
            "• `/settitle @user1 Best Batsman 🏏`\n"
            "• `/settitle @player123 Star Bowler ⭐`\n\n"
            "ℹ️ Use `/help` for detailed guide!",
            parse_mode='Markdown'
        )
        return
    
    player_identifier = context.args[0]
    title = ' '.join(context.args[1:])
    
    player = bot_instance.find_player_by_identifier(player_identifier)
    
    if not player:
        await update.message.reply_text(
            f"❌ **PLAYER NOT FOUND!**\n\n"
            f"🔍 Could not find: `{player_identifier}`",
            parse_mode='Markdown'
        )
        return
    
    if bot_instance.set_player_title(player['id'], title):
        await update.message.reply_text(
            f"✅ <b>TITLE ASSIGNED!</b> 👑\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"👤 <b>Player:</b> {player['display_name']}\n"
            f"👑 <b>New Title:</b> <b>{title}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🎉 <b>Title successfully assigned!</b> ⭐",
            parse_mode='HTML'
        )
    else:
        await update.message.reply_text("❌ <b>FAILED!</b> Unable to set title. Please try again.", parse_mode='HTML')

async def remove_title_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove player title (Admin only)"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can remove titles!", parse_mode='HTML')
        return

    if len(context.args) < 1:
        await update.message.reply_text(
            "📝 <b>REMOVE TITLE COMMAND</b>\n\n"
            "<b>Usage:</b> /rmtitle @username or UserID\n\n"
            "<b>Examples:</b>\n"
            "• /rmtitle @player123\n"
            "• /rmtitle 123456789\n\n"
            "💡 This will remove the player's custom title.",
            parse_mode='HTML'
        )
        return

    player_identifier = context.args[0]
    
    player = bot_instance.find_player_by_identifier(player_identifier)
    
    if not player:
        await update.message.reply_text(
            f"❌ <b>Player not found!</b>\n\n"
            f"🔍 Could not find player: <code>{player_identifier}</code>\n\n"
            f"💡 Make sure they have started the bot first!",
            parse_mode='HTML'
        )
        return

    if bot_instance.set_player_title(player['id'], ""):
        await update.message.reply_text(
            f"✅ <b>TITLE REMOVED!</b>\n\n"
            f"👤 <b>Player:</b> {player['display_name']}\n"
            f"🏷️ <b>Previous Title:</b> {player.get('title', 'None')}\n"
            f"🗑️ <b>New Status:</b> No title\n\n"
            f"🎯 Title successfully removed!",
            parse_mode='HTML'
        )
    else:
        await update.message.reply_text("❌ Error removing title. Please try again.")

async def finduser_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Find user information (Admin only)"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can search user information!", parse_mode='HTML')
        return

    if len(context.args) < 1:
        await update.message.reply_text(
            "🔍 <b>FIND USER COMMAND</b>\n\n"
            "<b>Usage:</b> /finduser @username or UserID\n\n"
            "<b>Examples:</b>\n"
            "• /finduser @player123\n"
            "• /finduser 123456789\n\n"
            "💡 This will show detailed user information.",
            parse_mode='HTML'
        )
        return

    user_identifier = context.args[0]
    
    player = bot_instance.find_player_by_identifier(user_identifier)
    
    if not player:
        await update.message.reply_text(
            f"❌ <b>User not found!</b>\n\n"
            f"🔍 Could not find user: <code>{user_identifier}</code>\n\n"
            f"💡 Make sure they have started the bot first!",
            parse_mode='HTML'
        )
        return

    achievements = bot_instance.get_player_achievements(player['id'])
    current_shards, total_earned = bot_instance.get_player_shards(player['telegram_id'])
    
    message = f"👤 <b>USER INFORMATION</b>\n\n"
    message += f"🆔 <b>ID:</b> {player['telegram_id']}\n"
    message += f"👤 <b>Display Name:</b> {player['display_name']}\n"
    if player.get('username'):
        message += f"📧 <b>Username:</b> @{player['username']}\n"
    if player.get('title'):
        message += f"🏷️ <b>Title:</b> {player['title']}\n"
    message += f"📅 <b>Joined:</b> {player['created_at'].strftime('%Y-%m-%d')}\n"
    message += f"🕐 <b>Last Active:</b> {player['updated_at'].strftime('%Y-%m-%d')}\n\n"
    message += f"💎 <b>Shards:</b> {current_shards:,} (Total Earned: {total_earned:,})\n"
    message += f"🏆 <b>Achievements:</b> {len(achievements)} unique\n\n"
    
    if achievements:
        message += f"🎖️ <b>Achievement Summary:</b>\n"
        for ach_name, count in achievements[:5]:  # Show top 5
            message += f"• {ach_name}: {count}\n"
        if len(achievements) > 5:
            message += f"• ... and {len(achievements) - 5} more\n"

    await update.message.reply_text(message, parse_mode='HTML')

async def banuser_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ban user (Admin only)"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can ban users!", parse_mode='HTML')
        return

    if len(context.args) < 1:
        await update.message.reply_text(
            "🚫 <b>BAN USER</b>\n\n"
            "<b>Usage:</b> /banuser @username [reason]\n"
            "<b>Usage:</b> /banuser UserID [reason]\n\n"
            "<b>Examples:</b>\n"
            "• /banuser @spammer Spamming the chat\n"
            "• /banuser 123456789 Violating rules\n\n"
            "⚠️ <b>Note:</b> Banned users cannot use any bot commands.",
            parse_mode='HTML'
        )
        return

    target_input = context.args[0]
    reason = " ".join(context.args[1:]) if len(context.args) > 1 else "No reason provided"
    
    target_player = bot_instance.find_player_by_identifier(target_input)

    if not target_player:
        await update.message.reply_text("❌ <b>User not found!</b> Make sure they're registered with the bot.", parse_mode='HTML')
        return

    if bot_instance.is_admin(target_player['telegram_id']):
        await update.message.reply_text("❌ <b>Cannot ban admin users!</b>", parse_mode='HTML')
        return

    if bot_instance.is_banned(target_player['telegram_id']):
        await update.message.reply_text("⚠️ <b>User is already banned!</b>", parse_mode='HTML')
        return

    admin = update.effective_user
    success = bot_instance.ban_user(
        target_player['telegram_id'],
        target_player.get('username', ''),
        target_player.get('display_name', ''),
        admin.id,
        admin.full_name or admin.username or str(admin.id),
        reason
    )

    if success:
        await update.message.reply_text(
            f"🚫 <b>USER BANNED</b>\n\n"
            f"� <b>User:</b> {target_player.get('display_name', 'Unknown')}\n"
            f"🆔 <b>ID:</b> {target_player['telegram_id']}\n"
            f"📝 <b>Reason:</b> {reason}\n"
            f"👮 <b>Banned by:</b> {admin.full_name or admin.username}\n\n"
            f"✅ <b>User is now banned from using the bot.</b>",
            parse_mode='HTML'
        )
    else:
        await update.message.reply_text("❌ <b>Failed to ban user!</b> Database error.", parse_mode='HTML')

async def unbanuser_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Unban user (Admin only)"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can unban users!", parse_mode='HTML')
        return

    if len(context.args) < 1:
        await update.message.reply_text(
            "✅ <b>UNBAN USER</b>\n\n"
            "<b>Usage:</b> /unbanuser @username\n"
            "<b>Usage:</b> /unbanuser UserID\n\n"
            "<b>Examples:</b>\n"
            "• /unbanuser @player123\n"
            "• /unbanuser 123456789\n\n"
            "💡 <b>Note:</b> This will restore user's access to the bot.",
            parse_mode='HTML'
        )
        return

    target_input = context.args[0]
    
    target_player = bot_instance.find_player_by_identifier(target_input)

    if not target_player:
        await update.message.reply_text("❌ <b>User not found!</b> Make sure they're registered with the bot.", parse_mode='HTML')
        return

    if not bot_instance.is_banned(target_player['telegram_id']):
        await update.message.reply_text("⚠️ <b>User is not currently banned!</b>", parse_mode='HTML')
        return

    success = bot_instance.unban_user(target_player['telegram_id'])

    if success:
        admin = update.effective_user
        await update.message.reply_text(
            f"✅ <b>USER UNBANNED</b>\n\n"
            f"👤 <b>User:</b> {target_player.get('display_name', 'Unknown')}\n"
            f"🆔 <b>ID:</b> {target_player['telegram_id']}\n"
            f"👮 <b>Unbanned by:</b> {admin.full_name or admin.username}\n\n"
            f"✅ <b>User can now use the bot again.</b>",
            parse_mode='HTML'
        )
    else:
        await update.message.reply_text("❌ <b>Failed to unban user!</b> Database error or user was not banned.", parse_mode='HTML')

async def restart_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Restart bot (Super Admin only) - Placeholder implementation"""
    if not bot_instance.is_super_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n👑 Only Super Admin can restart the bot!", parse_mode='HTML')
        return

    await update.message.reply_text(
        "🔄 <b>BOT RESTART SYSTEM</b>\n\n"
        "⚠️ <b>Feature Coming Soon!</b>\n\n"
        "This feature will allow safe bot restart with:\n"
        "• Graceful connection closure\n"
        "• State preservation\n"
        "• Automatic recovery\n\n"
        "💡 Use hosting platform controls for now.",
        parse_mode='HTML'
    )

async def backup_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Create database backup (Super Admin only) - Placeholder implementation"""
    if not bot_instance.is_super_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n👑 Only Super Admin can create backups!", parse_mode='HTML')
        return

    await update.message.reply_text(
        "💾 <b>DATABASE BACKUP SYSTEM</b>\n\n"
        "⚠️ <b>Feature Coming Soon!</b>\n\n"
        "This feature will provide:\n"
        "• Complete database backup\n"
        "• Automated scheduling\n"
        "• Secure storage options\n"
        "• Easy restoration\n\n"
        "💡 Use database provider tools for now.",
        parse_mode='HTML'
    )

async def cleancache_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clean bot cache (Admin only)"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can clean cache!", parse_mode='HTML')
        return

    try:
        cache_cleared = 0
        
        if hasattr(bot_instance, 'profile_cache') and bot_instance.profile_cache['data']:
            profile_count = len(bot_instance.profile_cache['data'])
            bot_instance.profile_cache['data'].clear()
            bot_instance.profile_cache['last_updated'].clear()
            bot_instance.profile_cache['data_version'].clear()
            bot_instance.profile_cache['global_invalidated_at'] = time.time()
            cache_cleared += profile_count
        
        if hasattr(bot_instance, 'roast_cache') and bot_instance.roast_cache:
            roast_count = len(bot_instance.roast_cache)
            bot_instance.roast_cache.clear()
            bot_instance.roast_rotation_index = 0
            cache_cleared += roast_count
        
        if hasattr(bot_instance, 'goat_cache'):
            bot_instance.goat_cache = None
            cache_cleared += 1
        
        await update.message.reply_text(
            f"🧹 <b>CACHE CLEANUP COMPLETE!</b>\n\n"
            f"✅ <b>Items Cleared:</b> {cache_cleared}\n"
            f"🗑️ <b>Caches Cleaned:</b>\n"
            f"• Profile cache\n"
            f"• Roast cache\n"
            f"• GOAT cache\n"
            f"• General cache\n\n"
            f"🚀 Bot performance may improve!",
            parse_mode='HTML'
        )
        
    except Exception as e:
        logger.error(f"Error cleaning cache: {e}")
        await update.message.reply_text("❌ Error cleaning cache. Please try again.")

async def cache_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show comprehensive cache status and health (Admin only)"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can view cache status!", parse_mode='HTML')
        return

    try:
        current_time = time.time()
        message = "📊 <b>CACHE STATUS & HEALTH</b> 🧠\n\n"
        
        if hasattr(bot_instance, 'profile_cache'):
            cache = bot_instance.profile_cache
            cached_users = len(cache['data'])
            total_entries = len(cache['last_updated'])
            
            if cache['last_updated']:
                ages = [current_time - timestamp for timestamp in cache['last_updated'].values()]
                avg_age = sum(ages) / len(ages) if ages else 0
                oldest_age = max(ages) if ages else 0
            else:
                avg_age = oldest_age = 0
            
            stale_count = 0
            for user_id, timestamp in cache['last_updated'].items():
                if current_time - timestamp >= cache['cache_duration']:
                    stale_count += 1
            
            message += f"👤 <b>Profile Cache</b>\n"
            message += f"   📦 <b>Cached Users:</b> {cached_users}\n"
            message += f"   ⏱️ <b>Cache Duration:</b> {cache['cache_duration']}s\n"
            message += f"   📊 <b>Average Age:</b> {avg_age:.1f}s\n"
            message += f"   🕒 <b>Oldest Entry:</b> {oldest_age:.1f}s\n"
            message += f"   🔄 <b>Stale Entries:</b> {stale_count}\n"
            
            if cache['global_invalidated_at'] > 0:
                global_age = current_time - cache['global_invalidated_at']
                message += f"   🌐 <b>Global Invalidated:</b> {global_age:.1f}s ago\n"
            
            message += "\n"
        
        if hasattr(bot_instance, 'leaderboard_cache'):
            cache = bot_instance.leaderboard_cache
            entries = len(cache['data']) if cache['data'] else 0
            age = current_time - cache['last_updated'] if cache['last_updated'] > 0 else 0
            is_stale = age >= cache['cache_duration']
            
            message += f"🏆 <b>Leaderboard Cache</b>\n"
            message += f"   📦 <b>Entries:</b> {entries}\n"
            message += f"   ⏱️ <b>Cache Duration:</b> {cache['cache_duration']}s\n"
            message += f"   📊 <b>Age:</b> {age:.1f}s\n"
            message += f"   ⚠️ <b>Status:</b> {'🔴 Stale' if is_stale else '🟢 Fresh'}\n\n"
        
        if hasattr(bot_instance, 'goat_cache'):
            cache = bot_instance.goat_cache
            has_data = cache['data'] is not None
            age = current_time - cache['last_updated'] if cache['last_updated'] > 0 else 0
            is_stale = age >= cache['cache_duration']
            
            message += f"🐐 <b>GOAT Cache</b>\n"
            message += f"   📦 <b>Has Data:</b> {'✅ Yes' if has_data else '❌ No'}\n"
            message += f"   ⏱️ <b>Cache Duration:</b> {cache['cache_duration']}s\n"
            message += f"   📊 <b>Age:</b> {age:.1f}s\n"
            message += f"   ⚠️ <b>Status:</b> {'🔴 Stale' if is_stale else '🟢 Fresh'}\n\n"
        
        message += "🔍 <b>CACHE HEALTH SUMMARY</b>\n"
        
        total_cached_items = 0
        total_stale_items = 0
        
        if hasattr(bot_instance, 'profile_cache'):
            total_cached_items += len(bot_instance.profile_cache['data'])
            for user_id, timestamp in bot_instance.profile_cache['last_updated'].items():
                if current_time - timestamp >= bot_instance.profile_cache['cache_duration']:
                    total_stale_items += 1
        
        hit_rate = ((total_cached_items - total_stale_items) / total_cached_items * 100) if total_cached_items > 0 else 100
        
        message += f"   📊 <b>Total Items:</b> {total_cached_items}\n"
        message += f"   🔄 <b>Stale Items:</b> {total_stale_items}\n"
        message += f"   ⭐ <b>Hit Rate:</b> {hit_rate:.1f}%\n"
        
        if hit_rate < 70:
            message += f"   ⚠️ <b>Recommendation:</b> Consider running /cleancache\n"
        elif hit_rate > 90:
            message += f"   ✅ <b>Status:</b> Cache performing well!\n"
        else:
            message += f"   ℹ️ <b>Status:</b> Cache performance is acceptable\n"
        
        message += "\n💡 <b>Commands:</b> /cleancache to clear all caches"
        
        await update.message.reply_text(message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Error getting cache status: {e}")
        await update.message.reply_text("❌ Error retrieving cache status. Please try again.")

async def thread_safety_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show thread safety status of shared data structures (Admin only)"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can view thread safety status!", parse_mode='HTML')
        return

    try:
        message = "🔒 <b>THREAD SAFETY STATUS</b> 🛡️\n\n"
        
        games_type = type(ACTIVE_CHASE_GAMES).__name__
        games_count = len(ACTIVE_CHASE_GAMES)
        games_safe = isinstance(ACTIVE_CHASE_GAMES, ThreadSafeDict)
        
        message += f"🎮 <b>Active Chase Games</b>\n"
        message += f"   📦 <b>Type:</b> {games_type}\n"
        message += f"   📊 <b>Count:</b> {games_count}\n"
        message += f"   🔒 <b>Thread Safe:</b> {'✅ Yes' if games_safe else '❌ No'}\n\n"
        
        if hasattr(bot_instance, 'profile_cache'):
            cache = bot_instance.profile_cache
            data_type = type(cache['data']).__name__
            updated_type = type(cache['last_updated']).__name__
            version_type = type(cache['data_version']).__name__
            
            data_safe = isinstance(cache['data'], ThreadSafeDict)
            updated_safe = isinstance(cache['last_updated'], ThreadSafeDict)
            version_safe = isinstance(cache['data_version'], ThreadSafeDict)
            
            message += f"👤 <b>Profile Cache</b>\n"
            message += f"   📦 <b>Data Type:</b> {data_type}\n"
            message += f"   🔒 <b>Data Safe:</b> {'✅ Yes' if data_safe else '❌ No'}\n"
            message += f"   ⏱️ <b>Timestamps Type:</b> {updated_type}\n"
            message += f"   🔒 <b>Timestamps Safe:</b> {'✅ Yes' if updated_safe else '❌ No'}\n"
            message += f"   🏷️ <b>Versions Type:</b> {version_type}\n"
            message += f"   🔒 <b>Versions Safe:</b> {'✅ Yes' if version_safe else '❌ No'}\n\n"
        
        if hasattr(bot_instance, 'roast_cache'):
            usage_counts = bot_instance.roast_cache['usage_counts']
            usage_type = type(usage_counts).__name__
            usage_safe = isinstance(usage_counts, (ThreadSafeDict, ThreadSafeCounter))
            usage_count = len(usage_counts) if usage_counts else 0
            
            message += f"🎭 <b>Roast Cache</b>\n"
            message += f"   📦 <b>Usage Counts Type:</b> {usage_type}\n"
            message += f"   🔒 <b>Thread Safe:</b> {'✅ Yes' if usage_safe else '❌ No'}\n"
            message += f"   📊 <b>Tracked Lines:</b> {usage_count}\n\n"
        
        all_structures = [
            ("Active Games", isinstance(ACTIVE_CHASE_GAMES, ThreadSafeDict)),
        ]
        
        if hasattr(bot_instance, 'profile_cache'):
            cache = bot_instance.profile_cache
            all_structures.extend([
                ("Profile Data", isinstance(cache['data'], ThreadSafeDict)),
                ("Profile Timestamps", isinstance(cache['last_updated'], ThreadSafeDict)),
                ("Profile Versions", isinstance(cache['data_version'], ThreadSafeDict)),
            ])
        
        if hasattr(bot_instance, 'roast_cache'):
            all_structures.append(
                ("Roast Usage", isinstance(bot_instance.roast_cache['usage_counts'], (ThreadSafeDict, ThreadSafeCounter)))
            )
        
        safe_count = sum(1 for _, is_safe in all_structures if is_safe)
        total_count = len(all_structures)
        safety_percentage = (safe_count / total_count * 100) if total_count > 0 else 0
        
        message += f"📊 <b>THREAD SAFETY SUMMARY</b>\n"
        message += f"   ✅ <b>Safe Structures:</b> {safe_count}/{total_count}\n"
        message += f"   📈 <b>Safety Rate:</b> {safety_percentage:.1f}%\n"
        
        if safety_percentage == 100:
            message += f"   🛡️ <b>Status:</b> Fully protected from race conditions!\n"
        elif safety_percentage >= 75:
            message += f"   ⚠️ <b>Status:</b> Mostly safe, some improvements needed\n"
        else:
            message += f"   🚨 <b>Status:</b> High risk of race conditions!\n"
        
        active_threads = threading.active_count()
        current_thread = threading.current_thread().name
        
        message += f"\n🧵 <b>THREADING INFO</b>\n"
        message += f"   📊 <b>Active Threads:</b> {active_threads}\n"
        message += f"   🎯 <b>Current Thread:</b> {current_thread}\n"
        
        if safety_percentage < 100:
            message += f"\n💡 <b>RECOMMENDATIONS</b>\n"
            message += f"   • Restart bot to activate thread-safe structures\n"
            message += f"   • Monitor for race condition errors in logs\n"
            message += f"   • Use /cleancache if experiencing data corruption\n"
        
        await update.message.reply_text(message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Error getting thread safety status: {e}")
        await update.message.reply_text("❌ Error retrieving thread safety status. Please try again.")

async def transactions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """View recent shard transactions (Admin only)"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can view transaction logs!", parse_mode='HTML')
        return

    try:
        conn = bot_instance.get_db_connection()
        if not conn:
            await update.message.reply_text("❌ Database error occurred!")
            return
        
        cursor = conn.cursor()
        
        if context.args and len(context.args) > 0:
            user_identifier = context.args[0]
            player = bot_instance.find_player_by_identifier(user_identifier)
            
            if not player:
                await update.message.reply_text(f"❌ User not found: {user_identifier}")
                return
            
            cursor.execute("""
                SELECT transaction_type, amount, source, source_details, performed_at
                FROM shard_transactions 
                WHERE player_telegram_id = %s
                ORDER BY performed_at DESC
                LIMIT 20
            """, (player['telegram_id'],))
            
            transactions = cursor.fetchall()
            
            if not transactions:
                await update.message.reply_text(f"💸 No transactions found for {player['display_name']}")
                return
            
            message = f"💸 <b>TRANSACTION LOG</b>\n"
            message += f"👤 <b>User:</b> {player['display_name']}\n\n"
            
        else:
            cursor.execute("""
                SELECT player_name, transaction_type, amount, source, source_details, performed_at
                FROM shard_transactions 
                ORDER BY performed_at DESC
                LIMIT 20
            """)
            
            transactions = cursor.fetchall()
            
            if not transactions:
                await update.message.reply_text("💸 No transactions found in the system")
                return
            
            message = f"💸 <b>RECENT TRANSACTIONS</b>\n\n"
        
        cursor.close()
        bot_instance.return_db_connection(conn)
        
        for i, tx in enumerate(transactions, 1):
            if context.args:  # User-specific
                tx_type, amount, source, details, performed_at = tx
                player_name = ""
            else:  # Global
                player_name, tx_type, amount, source, details, performed_at = tx
                player_name = f"👤 {player_name}\n"
            
            emoji = "💰" if tx_type == "earned" else "💸"
            sign = "+" if tx_type == "earned" else "-"
            
            message += f"{i}. {emoji} <b>{sign}{amount}</b> shards\n"
            if not context.args:
                message += f"   {player_name}"
            message += f"   📊 <b>Source:</b> {source}\n"
            if details:
                message += f"   📝 <b>Details:</b> {details}\n"
            message += f"   🕐 {performed_at.strftime('%Y-%m-%d %H:%M')}\n\n"
        
        message += "━━━━━━━━━━━━━━━━━━━━\n"
        message += f"📊 Showing last {len(transactions)} transactions"
        
        await update.message.reply_text(message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Error viewing transactions: {e}")
        await update.message.reply_text("❌ Error retrieving transaction log.")

@check_banned
async def achievements_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """View your own achievements"""
    user = update.effective_user
    
    if not bot_instance.create_or_update_player(
        user.id, 
        user.username or "", 
        user.full_name or user.username or f"User {user.id}"
    ):
        await update.message.reply_text("❌ **ERROR!** Failed to load your profile. Please try again.")
        return
    
    player = bot_instance.find_player_by_identifier(str(user.id))
    
    if not player:
        await update.message.reply_text("❌ **ERROR!** Failed to load your profile. Please contact admin.")
        return
    
    achievements = bot_instance.get_player_achievements(player['id'])
    
    message = bot_instance.format_achievements_message(player, achievements)
    await update.message.reply_text(message, parse_mode='HTML')

@check_banned
async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """View your comprehensive player profile with caching"""
    user = update.effective_user
    
    try:
        if not bot_instance.create_or_update_player(
            user.id, 
            user.username or "", 
            user.full_name or user.username or f"User {user.id}"
        ):
            await update.message.reply_text(
                "❌ <b>ERROR!</b> Failed to load your profile. Please try again later.",
                parse_mode='HTML'
            )
            return
        
        profile_data = bot_instance.get_cached_profile_data(user.id)
        
        if not profile_data:
            await update.message.reply_text(
                "❌ <b>ERROR!</b> Failed to load your profile. Please contact admin.",
                parse_mode='HTML'
            )
            return
        
        player = profile_data['player']
        achievements = profile_data['achievements']
        chase_stats = profile_data['chase_stats']
        
        message = f"👑 <b>CHAMPION PROFILE</b> 🏆\n"
        message += f"━━━━━━━━━━━━━━━━━━━━\n\n"
        
        if player.get('title') and player['title'].strip():
            message += f"⚡ <b>ARENA STATUS</b> ⚡\n"
            message += f"👑 <b>{H(player['title'])}</b>\n\n"
        
        message += f"━━━━━━━━━━━━━━━━━━━━\n"
        message += f"🏆 <b>CHAMPION:</b> {H(player['display_name'])}\n"
        
        if player.get('username'):
            message += f"🆔 <b>Username:</b> @{player['username']}\n"
        
        if player.get('created_at'):
            try:
                from datetime import datetime
                if isinstance(player['created_at'], str):
                    join_date = datetime.fromisoformat(player['created_at'].replace('Z', '+00:00'))
                else:
                    join_date = player['created_at']
                formatted_date = join_date.strftime("%B %d, %Y")
                message += f"📅 <b>Joined:</b> {formatted_date}\n"
            except Exception as e:
                logger.error(f"Error formatting date: {e}")
                message += f"📅 <b>Joined:</b> Recently\n"
        else:
            message += f"📅 <b>Joined:</b> Recently\n"
            
        message += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        message += "🏏 <b>CHASE GAME STATS</b>\n"
        message += "━━━━━━━━━━━━━━━━━━━━\n"
        message += f"🎮 <b>Games Played:</b> {chase_stats.get('games_played', 0)}\n"
        message += f"📈 <b>Highest Level:</b> {chase_stats.get('highest_level', 0)}\n"
        message += f"🏆 <b>Highest Score:</b> {chase_stats.get('highest_score', 0)}\n"
        message += f"⚡ <b>Best Strike Rate:</b> {chase_stats.get('best_sr', 0.0)}\n"
        
        if chase_stats.get('rank'):
            message += f"🏅 <b>Overall Rank:</b> #{chase_stats['rank']}\n"
        else:
            message += f"🏅 <b>Overall Rank:</b> Unranked\n"
        
        message += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        guess_stats = bot_instance.get_guess_game_stats(user.id)
        
        conn = bot_instance.get_db_connection()
        player_scores = {'highest_score': 0, 'total_score': 0, 'unlocked_levels': ['beginner']}
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT highest_score, total_score, unlocked_levels 
                    FROM players WHERE telegram_id = %s
                """, (user.id,))
                result = cursor.fetchone()
                if result:
                    player_scores = {
                        'highest_score': result[0] or 0,
                        'total_score': result[1] or 0,
                        'unlocked_levels': result[2] or ['beginner']
                    }
            finally:
                bot_instance.return_db_connection(conn)
        
        message += "🎲 <b>GUESS GAME STATS</b>\n"
        message += "━━━━━━━━━━━━━━━━━━━━\n"
        
        if guess_stats and guess_stats.get('games_played', 0) > 0:
            message += f"🎮 <b>Games Played:</b> {guess_stats.get('games_played', 0)}\n"
            message += f"🏆 <b>Games Won:</b> {guess_stats.get('games_won', 0)}\n"
            message += f"📈 <b>Win Rate:</b> {guess_stats.get('win_rate', 0)}%\n"
            message += f"💎 <b>Highest Score:</b> {player_scores['highest_score']}\n"
            message += f"🎯 <b>Total Score:</b> {player_scores['total_score']}\n"
            message += f"⚡ <b>Perfect Guesses:</b> {guess_stats.get('perfect_guesses', 0)}\n"
            message += f"🔓 <b>Unlocked Levels:</b> {len(player_scores['unlocked_levels'])}/5\n"
            message += f"📊 <b>Available:</b> {', '.join(player_scores['unlocked_levels'])}\n"
        else:
            message += "🚫 <b>No guess games played</b>\n"
            message += "🎯 <i>Use /guess to start!</i>\n"
            message += f"🔓 <b>Unlocked:</b> {len(player_scores['unlocked_levels'])}/5 levels\n"
            
        message += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        message += f"💡 <b>QUICK TIPS</b>\n"
        message += "━━━━━━━━━━━━━━━━━━━━\n"
        message += "🎮 <b>Commands:</b> /chase, /guess, /dailyguess\n"
        message += "🔍 <b>View:</b> /achievements, /shards, /sleaderboard\n"
        message += "📊 <b>Stats:</b> /chasestats, /guessstats\n"
        message += "━━━━━━━━━━━━━━━━━━━━"
        
        await update.message.reply_text(message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Profile command error for user {user.id}: {e}")
        await update.message.reply_text(
            "❌ <b>ERROR!</b> Failed to load your profile. Please try again later.",
            parse_mode='HTML'
        )

async def shardlb_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show top 10 shard leaderboard"""
    try:
        shard_stats = bot_instance.get_shard_economy_stats()
        top_holders = shard_stats.get('top_holders', [])
        
        if not top_holders:
            await update.message.reply_text(
                "📊 <b>Shard Leaderboard</b>\n\n❌ No shard holders found!",
                parse_mode='HTML'
            )
            return
        
        message = f"""💠 <b>SHARD LEADERBOARD</b> 👑

━━━━━━━━━━━━━━━━━━━━━━━
🏆 <b>TOP 10 SHARD MILLIONAIRES</b>
━━━━━━━━━━━━━━━━━━━━━━━

"""
        
        rank_emojis = ["🥇", "🥈", "🥉"]
        
        for i, (name, current, total) in enumerate(top_holders[:10]):
            if i < 3:
                rank = rank_emojis[i]
            else:
                rank = f"{i+1}."
            
            if current >= 1000:
                indicator = " 🤑"
            elif current >= 500:
                indicator = " 💰"
            elif current >= 200:
                indicator = " 💸"
            else:
                indicator = ""
                
            message += f"{rank} <b>{H(str(name))}</b>{indicator}\n"
            message += f"    💠 <b>Balance:</b> {current:,} shards\n"
            message += f"    📈 <b>Total Earned:</b> {total:,} shards\n\n"
        
        total_circulation = shard_stats.get('total_circulation', 0)
        total_earned = shard_stats.get('total_earned', 0)
        
        message += f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        message += f"📊 <b>ECONOMY SUMMARY</b>\n"
        message += f"• <b>Total Circulation:</b> {total_circulation:,} 💠\n"
        message += f"• <b>Total Ever Earned:</b> {total_earned:,} 💠\n\n"
        message += f"💡 <b>Earn shards by playing games!</b>"
        
        await update.message.reply_text(message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Error in shardlb command: {e}")
        await update.message.reply_text("❌ Error getting shard leaderboard!", parse_mode='HTML')

async def admin_panel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle admin panel button callbacks"""
    query = update.callback_query
    await query.answer()
    
    if not bot_instance.is_super_admin(query.from_user.id):
        await query.edit_message_text("❌ <b>ACCESS DENIED!</b> Only Super Admin can use this panel.", parse_mode='HTML')
        return
    
    data = query.data
    back_button = [InlineKeyboardButton("🔙 Back to Panel", callback_data="panel_main")]
    
    if data == "panel_main":
        keyboard = [
            [InlineKeyboardButton("📊 Bot Statistics", callback_data="panel_stats"),
             InlineKeyboardButton("👥 User Management", callback_data="panel_users")],
            [InlineKeyboardButton("💠 Economy Control", callback_data="panel_economy"),
             InlineKeyboardButton("🎮 Game Management", callback_data="panel_games")],
            [InlineKeyboardButton("🛡️ Admin Control", callback_data="panel_admins"),
             InlineKeyboardButton("📢 Broadcasting", callback_data="panel_broadcast")],
            [InlineKeyboardButton("🏆 Achievement System", callback_data="panel_achievements"),
             InlineKeyboardButton("🔧 System Tools", callback_data="panel_system")]
        ]
        
        message = f"""👑 <b>SUPER ADMIN CONTROL PANEL</b> 🚀

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 <b>Arena Of Champions Management</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>Welcome back, Super Admin!</b> 
Choose a management category below:

📊 <b>Bot Statistics</b> - View comprehensive bot stats
👥 <b>User Management</b> - Manage players and profiles  
💠 <b>Economy Control</b> - Shard system management
🎮 <b>Game Management</b> - Game stats and cleanup
🛡️ <b>Admin Control</b> - Add/remove admin privileges
📢 <b>Broadcasting</b> - Send messages to all users
🏆 <b>Achievement System</b> - Bulk awards and titles
🔧 <b>System Tools</b> - Database and system maintenance

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ <b>All bot functions at your fingertips!</b>"""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
        
    elif data == "panel_stats":
        keyboard = [
            [InlineKeyboardButton(" Shard Economy", callback_data="action_shardstats")],
            [InlineKeyboardButton("🏏 Chase Stats", callback_data="action_chasestats"),
             InlineKeyboardButton("🎯 Guess Stats", callback_data="action_guessstats")],
            [InlineKeyboardButton("📈 Daily Activity", callback_data="action_dailystats")],
            back_button
        ]
        
        message = f"""📊 <b>BOT STATISTICS PANEL</b> 📈

Choose which statistics to view:

📊 <b>Full Bot Stats</b> - Complete overview
💠 <b>Shard Economy</b> - Currency circulation
🏏 <b>Chase Stats</b> - Cricket game metrics  
🎯 <b>Guess Stats</b> - Number game metrics
📈 <b>Daily Activity</b> - Recent user activity

All statistics are real-time and comprehensive."""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
        
    elif data == "panel_economy":
        keyboard = [
            [InlineKeyboardButton("💰 Give Shards", callback_data="action_giveshards"),
             InlineKeyboardButton("🗑️ Remove Shards", callback_data="action_removeshards")],
            [InlineKeyboardButton("📊 Shard Leaderboard", callback_data="action_shardlb"),
             InlineKeyboardButton("💸 Transaction Log", callback_data="action_transactions")],
            [InlineKeyboardButton("🔄 Daily Rewards", callback_data="action_dailyrewards")],
            back_button
        ]
        
        message = f"""💠 <b>ECONOMY CONTROL PANEL</b> 💰

Manage the bot's shard economy:

💰 <b>Give Shards</b> - Award shards to players
🗑️ <b>Remove Shards</b> - Deduct shards from players
📊 <b>Shard Leaderboard</b> - Top 10 richest players
💸 <b>Transaction Log</b> - View recent transactions
🔄 <b>Daily Rewards</b> - Distribute daily bonuses

⚠️ <b>Use economy controls responsibly!</b>"""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
        
    elif data == "panel_games":
        keyboard = [
            [InlineKeyboardButton("🏏 Chase Cleanup", callback_data="action_chasecleanup"),
             InlineKeyboardButton("🎯 Guess Cleanup", callback_data="action_guesscleanup")],
            [InlineKeyboardButton("📊 Game Overview", callback_data="action_gameoverview"),
             InlineKeyboardButton("🔄 Reset Daily LB", callback_data="action_resetdaily")],
            back_button
        ]
        
        message = f"""🎮 <b>GAME MANAGEMENT PANEL</b> 🕹️

Manage all bot games and leaderboards:

🏏 <b>Chase Cleanup</b> - Force end all chase games
🎯 <b>Guess Cleanup</b> - Clear expired guess games  
📊 <b>Game Overview</b> - Active games across all modes
🔄 <b>Reset Daily LB</b> - Reset daily leaderboards

⚡ <b>Keep games running smoothly!</b>"""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "panel_admins":
        keyboard = [
            [InlineKeyboardButton("➕ Add Admin", callback_data="action_addadmin"),
             InlineKeyboardButton("➖ Remove Admin", callback_data="action_removeadmin")],
            [InlineKeyboardButton("📋 List Admins", callback_data="action_listadmins")],
            back_button
        ]
        
        message = f"""🛡️ <b>ADMIN CONTROL PANEL</b> 👑

Manage administrator privileges:

➕ <b>Add Admin</b> - Grant admin privileges
➖ <b>Remove Admin</b> - Revoke admin privileges
📋 <b>List Admins</b> - View all current admins

<b>Commands needed:</b>
• Add: /addadmin @username
• Remove: /rmadmin @username
• List: /aadmins
• Status: /adminstatus - Show hierarchy & conflicts"""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "panel_users":
        keyboard = [
            [InlineKeyboardButton("👥 List All Users", callback_data="action_listusers"),
             InlineKeyboardButton("🔍 Search User", callback_data="action_searchuser")],
            [InlineKeyboardButton("🚫 Ban User", callback_data="action_banuser"),
             InlineKeyboardButton("✅ Unban User", callback_data="action_unbanuser")],
            [InlineKeyboardButton("📊 User Stats", callback_data="action_userstats")],
            back_button
        ]
        
        message = f"""👥 <b>USER MANAGEMENT PANEL</b> 👤

Manage registered users and their accounts:

👥 <b>List All Users</b> - View all registered players
🔍 <b>Search User</b> - Find specific user details
🚫 <b>Ban User</b> - Restrict user access
✅ <b>Unban User</b> - Remove user restrictions
📊 <b>User Stats</b> - View user activity statistics

⚠️ <b>Handle user management responsibly!</b>"""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "panel_broadcast":
        keyboard = [
            [InlineKeyboardButton("📢 Send Announcement", callback_data="action_broadcast"),
             InlineKeyboardButton("📝 Draft Message", callback_data="action_draftmessage")],
            [InlineKeyboardButton("👥 Broadcast Stats", callback_data="action_broadcaststats")],
            back_button
        ]
        
        message = f"""📢 <b>BROADCASTING PANEL</b> 📻

Send messages to all registered users:

📢 <b>Send Announcement</b> - Broadcast to all users
📝 <b>Draft Message</b> - Prepare announcement text
👥 <b>Broadcast Stats</b> - View broadcast history

<b>Commands needed:</b>
• Broadcast: /broadcast [message]
• Test broadcast: /testbroadcast [message]"""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "panel_achievements":
        keyboard = [
            [InlineKeyboardButton("🏆 Award Achievement", callback_data="action_awardachievement"),
             InlineKeyboardButton("🗑️ Remove Achievement", callback_data="action_removeachievement")],
            [InlineKeyboardButton("📊 Achievement Stats", callback_data="action_achievementstats"),
             InlineKeyboardButton("👑 Manage Titles", callback_data="action_managetitles")],
            [InlineKeyboardButton("🔄 Bulk Operations", callback_data="action_bulkachievements")],
            back_button
        ]
        
        message = f"""🏆 <b>ACHIEVEMENT SYSTEM PANEL</b> 🎖️

Manage player achievements and titles:

🏆 <b>Award Achievement</b> - Grant achievement to player
🗑️ <b>Remove Achievement</b> - Remove player achievement
📊 <b>Achievement Stats</b> - View achievement distribution
👑 <b>Manage Titles</b> - Award/remove player titles
🔄 <b>Bulk Operations</b> - Mass achievement operations

<b>Commands needed:</b>
• Award: /giveachievement @username achievement_name
• Remove: /rmachievement @username achievement_name
• Title: /givetitle @username "title" """
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "panel_auctions":
        keyboard = [
            [InlineKeyboardButton("📋 List All Auctions", callback_data="action_listauctions"),
             InlineKeyboardButton("🗑️ Delete Auction", callback_data="action_deleteauction")],
            [InlineKeyboardButton("⏹️ Force End Auction", callback_data="action_forceendauction"),
             InlineKeyboardButton("📝 Pending Proposals", callback_data="action_pendingauctions")],
            [InlineKeyboardButton("📊 Auction Statistics", callback_data="action_auctionstats")],
            back_button
        ]
        
        total_proposals = len(bot_instance.auction_proposals)
        total_approved = len(bot_instance.approved_auctions)
        active_auctions = len([a for a in bot_instance.approved_auctions.values() if a.status == "active"])
        
        message = f"""🏆 <b>AUCTION MANAGEMENT PANEL</b> 🎯

Current System Status:

📝 <b>Pending Proposals:</b> {total_proposals}
✅ <b>Approved Auctions:</b> {total_approved}
🔥 <b>Active Auctions:</b> {active_auctions}

🛠️ <b>Available Actions:</b>

📋 <b>List All Auctions</b> - View all auctions with IDs
🗑️ <b>Delete Auction</b> - Remove any auction permanently
⏹️ <b>Force End Auction</b> - Forcefully end active auctions
📝 <b>Pending Proposals</b> - Review awaiting approval
📊 <b>Auction Statistics</b> - System usage stats

⚠️ <b>Admin Commands:</b>
• /delete_auction [id] - Delete specific auction
• /force_end_auction [id] - Force end active auction
• /list_auctions - Show all auctions with management options"""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "panel_system":
        keyboard = [
            [InlineKeyboardButton("🔄 Restart Bot", callback_data="action_restartbot"),
             InlineKeyboardButton("💾 Backup Database", callback_data="action_backupdb")],
            [InlineKeyboardButton("🧹 Clean Cache", callback_data="action_cleancache"),
             InlineKeyboardButton("📊 System Health", callback_data="action_systemhealth")],
            [InlineKeyboardButton("⚠️ Reset All Data", callback_data="action_resetalldata")],
            back_button
        ]
        
        message = f"""🔧 <b>SYSTEM TOOLS PANEL</b> ⚙️

Advanced system maintenance and operations:

🔄 <b>Restart Bot</b> - Restart the bot process
💾 <b>Backup Database</b> - Create database backup
🧹 <b>Clean Cache</b> - Clear all cached data
📊 <b>System Health</b> - Check system status
⚠️ <b>Reset All Data</b> - Nuclear option - delete everything

⚠️ <b>DANGER ZONE - Use with extreme caution!</b>"""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_shardstats":
        try:
            players = bot_instance.get_all_players_with_shards()
            if players:
                total_shards = sum(p.get('shards', 0) for p in players)
                avg_balance = total_shards / len(players) if players else 0
                richest = max(players, key=lambda x: x.get('shards', 0))
                
                message = f"""💰 <b>SHARD ECONOMY OVERVIEW</b>

💎 <b>Total Circulation:</b> {total_shards:,} shards
👥 <b>Players with Shards:</b> {len([p for p in players if p.get('shards', 0) > 0])}
📊 <b>Average Balance:</b> {avg_balance:.1f} shards
🤑 <b>Richest Player:</b> {richest.get('display_name', 'Unknown')} ({richest.get('shards', 0):,} shards)

💸 <b>Distribution:</b>
• 0 shards: {len([p for p in players if p.get('shards', 0) == 0])} players
• 1-100: {len([p for p in players if 1 <= p.get('shards', 0) <= 100])} players
• 101-1000: {len([p for p in players if 101 <= p.get('shards', 0) <= 1000])} players
• 1000+: {len([p for p in players if p.get('shards', 0) > 1000])} players"""
            else:
                message = "💰 No shard data available."
        except Exception as e:
            logger.error(f"Error getting shard stats: {e}")
            message = "❌ Error retrieving shard statistics."
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_shardlb":
        try:
            top_players = bot_instance.get_shard_leaderboard(limit=10)
            if top_players:
                message = "🏆 <b>TOP 10 SHARD LEADERBOARD</b>\n\n"
                for i, player in enumerate(top_players, 1):
                    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}️⃣"
                    message += f"{emoji} <b>{player['display_name']}</b> - {player['shards']:,} shards\n"
            else:
                message = "🏆 No players with shards yet!"
        except Exception as e:
            logger.error(f"Error getting shard leaderboard: {e}")
            message = "❌ Error retrieving leaderboard."
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_giveshards":
        message = """💰 <b>GIVE SHARDS TO PLAYER</b>

<b>Usage:</b> /giveshards @username amount
<b>Example:</b> /giveshards @player123 500

This will award shards directly to a player's balance.
Use this command in the main chat to execute."""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_removeshards":
        message = """🗑️ <b>REMOVE SHARDS FROM PLAYER</b>

<b>Usage:</b> /removeshards @username amount
<b>Example:</b> /removeshards @player123 100

This will deduct shards from a player's balance.
Use this command in the main chat to execute."""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_listadmins":
        try:
            admins = bot_instance.get_all_admins()
            if admins:
                message = "🛡️ <b>CURRENT ADMINISTRATORS</b>\n\n"
                for admin in admins:
                    message += f"• <b>{admin.get('display_name', admin.get('username', 'Unknown'))}</b>"
                    if admin.get('username'):
                        message += f" (@{admin['username']})"
                    message += f"\n  ID: {admin['telegram_id']}\n\n"
            else:
                message = "🛡️ No administrators found."
        except Exception as e:
            logger.error(f"Error getting admin list: {e}")
            message = "❌ Error retrieving admin list."
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_addadmin":
        message = """➕ <b>ADD ADMINISTRATOR</b>

<b>Usage:</b> /addadmin @username
<b>Example:</b> /addadmin @newadmin

This will grant administrator privileges to a user.
Use this command in the main chat to execute."""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_removeadmin":
        message = """➖ <b>REMOVE ADMINISTRATOR</b>

<b>Usage:</b> /rmadmin @username  
<b>Example:</b> /rmadmin @oldadmin

This will revoke administrator privileges from a user.
Use this command in the main chat to execute."""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_transactions":
        message = """💸 <b>TRANSACTION LOG</b>

<b>Usage:</b> /transactions or /transactions @username
<b>Example:</b> /transactions @player123

View recent shard transactions for debugging.
Use this command in the main chat to execute."""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_dailyrewards":
        message = """🔄 <b>DAILY REWARDS MANAGEMENT</b>

Daily rewards are automatically distributed.
Check /dailyreward status or manually trigger:

<b>Available commands:</b>
• /dailyreward - Player claims daily reward
• Manual distribution coming soon!"""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_chasestats":
        try:
            conn = bot_instance.get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) as total, COUNT(CASE WHEN game_outcome = 'won' THEN 1 END) as won FROM chase_games")
                stats = cursor.fetchone()
                cursor.close()
                bot_instance.return_db_connection(conn)
                
                if stats:
                    total_games = stats[0]
                    games_won = stats[1]
                    win_rate = (games_won / total_games * 100) if total_games > 0 else 0
                    
                    message = f"""🏏 <b>CHASE GAME STATISTICS</b>

📊 <b>Total Games:</b> {total_games}
🏆 <b>Games Won:</b> {games_won}
📈 <b>Win Rate:</b> {win_rate:.1f}%
💔 <b>Games Lost:</b> {total_games - games_won}"""
                else:
                    message = "🏏 No chase game data available."
            else:
                message = "❌ Database connection error."
        except Exception as e:
            logger.error(f"Error getting chase stats: {e}")
            message = "❌ Error retrieving chase statistics."
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_guessstats":
        try:
            conn = bot_instance.get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) as total, COUNT(CASE WHEN game_outcome = 'won' THEN 1 END) as won FROM guess_games")
                stats = cursor.fetchone()
                cursor.close()
                bot_instance.return_db_connection(conn)
                
                if stats:
                    total_games = stats[0]
                    games_won = stats[1]
                    win_rate = (games_won / total_games * 100) if total_games > 0 else 0
                    
                    message = f"""🎯 <b>GUESS GAME STATISTICS</b>

📊 <b>Total Games:</b> {total_games}
🏆 <b>Games Won:</b> {games_won}
📈 <b>Win Rate:</b> {win_rate:.1f}%
💔 <b>Games Lost:</b> {total_games - games_won}"""
                else:
                    message = "🎯 No guess game data available."
            else:
                message = "❌ Database connection error."
        except Exception as e:
            logger.error(f"Error getting guess stats: {e}")
            message = "❌ Error retrieving guess statistics."
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_dailystats":
        try:
            from datetime import datetime, timedelta
            today = datetime.now().date()
            
            conn = bot_instance.get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        COUNT(DISTINCT p.id) as active_players,
                        COUNT(cg.id) as chase_games,
                        COUNT(gg.id) as guess_games
                    FROM players p
                    LEFT JOIN chase_games cg ON p.telegram_id = cg.telegram_id AND DATE(cg.completed_at) = %s
                    LEFT JOIN guess_games gg ON p.telegram_id = gg.telegram_id AND DATE(gg.completed_at) = %s
                    WHERE p.updated_at >= %s
                """, (today, today, today - timedelta(days=1)))
                
                stats = cursor.fetchone()
                cursor.close()
                bot_instance.return_db_connection(conn)
                
                if stats:
                    message = f"""📈 <b>DAILY ACTIVITY REPORT</b>
📅 <b>Date:</b> {today.strftime('%Y-%m-%d')}

👥 <b>Active Players (24h):</b> {stats[0]}
🏏 <b>Chase Games Today:</b> {stats[1]}
🎯 <b>Guess Games Today:</b> {stats[2]}
🎮 <b>Total Games Today:</b> {stats[1] + stats[2]}"""
                else:
                    message = "📈 No daily activity data available."
            else:
                message = "❌ Database connection error."
        except Exception as e:
            logger.error(f"Error getting daily stats: {e}")
            message = "❌ Error retrieving daily statistics."
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_chasecleanup":
        message = """🏏 <b>CHASE GAME CLEANUP</b>

<b>Usage:</b> /cleanupchase
This will force-end all active chase games.

⚠️ <b>Warning:</b> This action cannot be undone!
Use only if games are stuck or causing issues."""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_guesscleanup":
        message = """🎯 <b>GUESS GAME CLEANUP</b>

<b>Usage:</b> /cleanup guess
This will clear all expired guess games.

⚠️ <b>Warning:</b> This action cannot be undone!
Use only if games are stuck or causing issues."""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_gameoverview":
        try:
            active_chase = len(bot_instance.get_all_chase_games())
            active_guess = sum(1 for p in bot_instance.get_all_players() if bot_instance.get_guess_game(p['telegram_id']))
            active_nightmare = sum(1 for p in bot_instance.get_all_players() if bot_instance.get_nightmare_game(p['telegram_id']))
            
            message = f"""📊 <b>ACTIVE GAMES OVERVIEW</b>

🏏 <b>Chase Games:</b> {active_chase} active
🎯 <b>Guess Games:</b> {active_guess} active  
🌙 <b>Nightmare Games:</b> {active_nightmare} active
🎮 <b>Total Active:</b> {active_chase + active_guess + active_nightmare}

<b>Use cleanup commands if games are stuck!</b>"""
        except Exception as e:
            logger.error(f"Error getting game overview: {e}")
            message = "❌ Error retrieving game overview."
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_resetdaily":
        message = """🔄 <b>RESET DAILY LEADERBOARDS</b>

Daily leaderboards reset automatically at midnight.
Manual reset coming soon!

<b>Current daily data will be archived.</b>"""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data.startswith("action_listusers"):
        try:
            page = 0
            if "_page_" in data:
                page = int(data.split("_page_")[1])
            
            players = bot_instance.get_all_players()
            if players:
                per_page = 10  # Show 10 users per page
                total_pages = (len(players) - 1) // per_page + 1
                start_idx = page * per_page
                end_idx = min(start_idx + per_page, len(players))
                
                message = f"👥 <b>REGISTERED USERS ({len(players)} total)</b>\n"
                message += f"📄 <b>Page {page + 1} of {total_pages}</b>\n\n"
                
                for i, player in enumerate(players[start_idx:end_idx], start_idx + 1):
                    message += f"{i}. <b>{player['display_name']}</b>"
                    if player.get('username'):
                        message += f" (@{player['username']})"
                    message += f"\n   ID: {player['telegram_id']}"
                    if player.get('shards', 0) > 0:
                        message += f" | 💎 {player['shards']}"
                    if player.get('title'):
                        message += f"\n   👑 {player['title']}"
                    message += "\n\n"
                
                nav_buttons = []
                if page > 0:
                    nav_buttons.append(InlineKeyboardButton("⬅️ Previous", callback_data=f"action_listusers_page_{page-1}"))
                if page < total_pages - 1:
                    nav_buttons.append(InlineKeyboardButton("➡️ Next", callback_data=f"action_listusers_page_{page+1}"))
                
                keyboard = []
                if nav_buttons:
                    keyboard.append(nav_buttons)
                keyboard.append(back_button)
            else:
                message = "👥 No users registered yet."
                keyboard = [back_button]
        except Exception as e:
            logger.error(f"Error listing users: {e}")
            message = "❌ Error retrieving user list."
            keyboard = [back_button]
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_searchuser":
        message = """🔍 <b>SEARCH USER</b>

<b>Usage:</b> /finduser @username or /finduser UserID
<b>Example:</b> /finduser @player123 or /finduser 123456789

This will show detailed user information.
Use this command in the main chat to execute."""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_banuser":
        message = """🚫 <b>BAN USER</b>

<b>Usage:</b> /banuser @username [reason]
<b>Example:</b> /banuser @baduser Spam and abuse

This will restrict the user from using the bot.
Use this command in the main chat to execute."""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_unbanuser":
        message = """✅ <b>UNBAN USER</b>

<b>Usage:</b> /unbanuser @username
<b>Example:</b> /unbanuser @rehabilitateduser

This will restore the user's access to the bot.
Use this command in the main chat to execute."""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_userstats":
        try:
            conn = bot_instance.get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM players")
                total_users = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM players WHERE updated_at > NOW() - INTERVAL '7 days'")
                active_weekly = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM players WHERE updated_at > NOW() - INTERVAL '1 day'")
                active_daily = cursor.fetchone()[0]
                
                cursor.close()
                bot_instance.return_db_connection(conn)
                
                message = f"""📊 <b>USER STATISTICS</b>

👥 <b>Total Registered:</b> {total_users}
📅 <b>Active Today:</b> {active_daily}
📊 <b>Active This Week:</b> {active_weekly}
💤 <b>Inactive Users:</b> {total_users - active_weekly}

📈 <b>Activity Rate:</b> {(active_weekly/total_users*100):.1f}% weekly"""
            else:
                message = "❌ Database connection error."
        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            message = "❌ Error retrieving user statistics."
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_broadcast":
        message = """📢 <b>SEND BROADCAST</b>

<b>Usage:</b> /broadcast Your message here
<b>Example:</b> /broadcast 🎉 New features available! Check them out!

This will send a message to ALL registered users.
⚠️ Use responsibly - avoid spam!

Use this command in the main chat to execute."""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_draftmessage":
        message = """📝 <b>DRAFT MESSAGE</b>

Draft and test your broadcast messages before sending:

1. Use /draftbroadcast [message] to create a draft
2. Use /testbroadcast to send to admins only
3. Use /broadcast to send to all users

This helps avoid sending messages with errors."""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_broadcaststats":
        message = """👥 <b>BROADCAST STATISTICS</b>

📊 Broadcast history and statistics:
• Total broadcasts sent: Coming soon!
• Success rate: Coming soon!
• Last broadcast: Coming soon!

Feature in development."""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_awardachievement":
        message = """🏆 <b>AWARD ACHIEVEMENT</b>

<b>Usage:</b> /giveachievement @username achievement_name
<b>Example:</b> /giveachievement @player123 winner

Available achievements: winner, orange cap, purple cap, mvp
Use this command in the main chat to execute."""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_removeachievement":
        message = """🗑️ <b>REMOVE ACHIEVEMENT</b>

<b>Usage:</b> /rmachievement @username achievement_name
<b>Example:</b> /rmachievement @player123 winner

This will remove the specified achievement from the user.
Use this command in the main chat to execute."""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_achievementstats":
        try:
            conn = bot_instance.get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT achievement_name, COUNT(*) as count 
                    FROM achievements 
                    GROUP BY achievement_name 
                    ORDER BY count DESC
                """)
                
                stats = cursor.fetchall()
                cursor.close()
                bot_instance.return_db_connection(conn)
                
                if stats:
                    message = "📊 <b>ACHIEVEMENT DISTRIBUTION</b>\n\n"
                    total_achievements = sum(stat[1] for stat in stats)
                    
                    for achievement, count in stats:
                        emoji = bot_instance.get_achievement_emoji(achievement)
                        message += f"{emoji} <b>{achievement.title()}:</b> {count} awarded\n"
                    
                    message += f"\n🎖️ <b>Total Achievements:</b> {total_achievements}"
                else:
                    message = "📊 No achievements awarded yet."
            else:
                message = "❌ Database connection error."
        except Exception as e:
            logger.error(f"Error getting achievement stats: {e}")
            message = "❌ Error retrieving achievement statistics."
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_managetitles":
        message = """👑 <b>MANAGE TITLES</b>

<b>Award Title:</b> /givetitle @username "Custom Title"
<b>Remove Title:</b> /rmtitle @username
<b>Example:</b> /givetitle @player123 "Cricket Legend 🏏"

Titles appear in player profiles and leaderboards.
Use this command in the main chat to execute."""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_bulkachievements":
        message = """🔄 <b>BULK OPERATIONS</b>

Bulk achievement operations:
• Mass award achievements: Coming soon!
• Mass remove achievements: Coming soon!
• Achievement backup/restore: Coming soon!

Feature in development for safety."""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_restartbot":
        message = """🔄 <b>RESTART BOT</b>

⚠️ <b>WARNING:</b> This will restart the entire bot!

All active games and sessions will be lost.
Use only if the bot is malfunctioning.

<b>Usage:</b> /restart
⚠️ This action cannot be undone!"""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_backupdb":
        message = """💾 <b>BACKUP DATABASE</b>

Create a complete backup of the database:

<b>Usage:</b> /backup
This will create a timestamped backup file.

Backups include all user data, achievements, and game history.
Store backups securely!"""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_cleancache":
        message = """🧹 <b>CLEAN CACHE</b>

Clear all cached data to free memory:

<b>Usage:</b> /cleancache
This will clear:
• Profile cache
• Leaderboard cache  
• GOAT cache
• Roast cache

Bot performance may improve after cache cleanup."""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_systemhealth":
        try:
            import psutil
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
            
            message = f"""📊 <b>SYSTEM HEALTH CHECK</b>

🖥️ <b>CPU Usage:</b> {cpu_usage}%
💾 <b>Memory Usage:</b> {memory_usage}%
🔗 <b>Database Pool:</b> {'✅ Active' if bot_instance.db_pool else '❌ Inactive'}
⚡ <b>Bot Status:</b> ✅ Running

🕒 <b>Uptime:</b> Bot running normally"""
        except ImportError:
            message = """📊 <b>SYSTEM HEALTH CHECK</b>

🖥️ <b>CPU Usage:</b> N/A (psutil not installed)
💾 <b>Memory Usage:</b> N/A (psutil not installed)
🔗 <b>Database Pool:</b> {'✅ Active' if bot_instance.db_pool else '❌ Inactive'}
⚡ <b>Bot Status:</b> ✅ Running

📦 <b>Note:</b> Install psutil for detailed system metrics"""
        except Exception as e:
            message = f"""📊 <b>SYSTEM HEALTH CHECK</b>

❌ <b>Error getting system info:</b> {str(e)}
🔗 <b>Database Pool:</b> {'✅ Active' if bot_instance.db_pool else '❌ Inactive'}
⚡ <b>Bot Status:</b> ✅ Running"""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_listauctions":
        message = "📋 <b>ALL AUCTIONS MANAGEMENT</b>\n\n"
        
        if bot_instance.auction_proposals:
            message += f"📝 <b>Pending Proposals ({len(bot_instance.auction_proposals)}):</b>\n"
            for proposal_id, proposal in bot_instance.auction_proposals.items():
                message += f"• ID: {proposal_id} - {proposal.name} (by {proposal.creator_name})\n"
            message += "\n"
        
        if bot_instance.approved_auctions:
            message += f"✅ <b>Approved Auctions ({len(bot_instance.approved_auctions)}):</b>\n"
            for auction_id, auction in bot_instance.approved_auctions.items():
                status_emoji = {
                    'setup': '🔧', 'captain_reg': '👑', 'player_reg': '👥',
                    'ready': '⚡', 'active': '🔥', 'completed': '✅'
                }.get(auction.status, '❓')
                message += f"• ID: {auction_id} - {auction.name} {status_emoji} ({auction.status})\n"
            message += "\n"
        
        if not bot_instance.auction_proposals and not bot_instance.approved_auctions:
            message += "📭 <b>No auctions found</b>\n\n"
        
        message += "🛠️ <b>Management Commands:</b>\n"
        message += "• <code>/delete_auction [id]</code> - Delete auction\n"
        message += "• <code>/force_end_auction [id]</code> - Force end active\n"
        message += "• <code>/pending_auctions</code> - View pending only"
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_deleteauction":
        message = "🗑️ <b>DELETE AUCTION</b>\n\n"
        message += "⚠️ <b>How to delete an auction:</b>\n"
        message += "1️⃣ Use <code>/list_auctions</code> to see all IDs\n"
        message += "2️⃣ Use <code>/delete_auction [id]</code> to delete\n\n"
        message += "<b>Example:</b> <code>/delete_auction 5</code>\n\n"
        message += "⚠️ <b>WARNING:</b> This action is permanent!"
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_forceendauction":
        active_auctions = [(aid, a) for aid, a in bot_instance.approved_auctions.items() if a.status == "active"]
        
        message = "⏹️ <b>FORCE END AUCTION</b>\n\n"
        
        if active_auctions:
            message += f"🔥 <b>Active Auctions ({len(active_auctions)}):</b>\n"
            for auction_id, auction in active_auctions:
                message += f"• ID: {auction_id} - {auction.name} (by {auction.creator_name})\n"
            message += "\n📝 <b>How to force end:</b>\n"
            message += "Use <code>/force_end_auction [id]</code>\n\n"
            message += "<b>Example:</b> <code>/force_end_auction 3</code>"
        else:
            message += "📭 <b>No active auctions to end</b>\n\n"
            message += "ℹ️ Only auctions with status 'active' can be force ended."
        
        keyboard = [back_button] 
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_pendingauctions":
        pending_proposals = [(pid, p) for pid, p in bot_instance.auction_proposals.items()]
        
        message = "📝 <b>PENDING AUCTION PROPOSALS</b>\n\n"
        
        if pending_proposals:
            message += f"⏳ <b>Awaiting Approval ({len(pending_proposals)}):</b>\n\n"
            for proposal_id, proposal in pending_proposals:
                message += f"🆔 <b>ID:</b> {proposal_id}\n"
                message += f"🏆 <b>Name:</b> {proposal.name}\n"
                message += f"👤 <b>Creator:</b> {proposal.creator_name}\n"
                message += f"👥 <b>Teams:</b> {len(proposal.teams)}\n"
                message += f"💰 <b>Purse:</b> {proposal.purse}Cr\n"
                message += f"📅 <b>Submitted:</b> {proposal.created_at.strftime('%Y-%m-%d %H:%M')}\n"
                message += "━━━━━━━━━━━━━━\n"
            
            message += "\n✅ Use admin approval buttons to approve/reject"
        else:
            message += "📭 <b>No pending proposals</b>\n\n"
            message += "ℹ️ All auction proposals have been processed."
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_auctionstats":
        total_proposals = len(bot_instance.auction_proposals)
        total_approved = len(bot_instance.approved_auctions)
        
        status_counts = {}
        total_teams = 0
        total_registered_captains = 0
        total_registered_players = 0
        
        for auction in bot_instance.approved_auctions.values():
            status = auction.status
            status_counts[status] = status_counts.get(status, 0) + 1
            total_teams += len(auction.teams)
            total_registered_captains += len(auction.approved_captains)
            total_registered_players += len(auction.approved_players)
        
        message = "📊 <b>AUCTION SYSTEM STATISTICS</b>\n\n"
        message += f"📈 <b>Overall Numbers:</b>\n"
        message += f"• Total Proposals: {total_proposals}\n"
        message += f"• Approved Auctions: {total_approved}\n"
        message += f"• Total Teams Created: {total_teams}\n"
        message += f"• Total Captains: {total_registered_captains}\n"
        message += f"• Total Players: {total_registered_players}\n\n"
        
        if status_counts:
            message += "📊 <b>Auction Status Breakdown:</b>\n"
            status_emojis = {
                'setup': '🔧', 'captain_reg': '👑', 'player_reg': '👥',
                'ready': '⚡', 'active': '🔥', 'completed': '✅'
            }
            for status, count in status_counts.items():
                emoji = status_emojis.get(status, '❓')
                message += f"• {emoji} {status.title()}: {count}\n"
        
        message += "\n🎯 <b>System Health:</b> " + ("✅ Operational" if bot_instance.approved_auctions or bot_instance.auction_proposals else "📭 No Active Auctions")
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "action_resetalldata":
        message = """⚠️ <b>NUCLEAR OPTION - RESET ALL DATA</b> ⚠️

🚨 <b>EXTREME DANGER ZONE!</b> 🚨

This will DELETE EVERYTHING:
• All user accounts
• All achievements  
• All game history
• All statistics

<b>Usage:</b> /resetall CONFIRM
⚠️ This action is IRREVERSIBLE!
💾 Create a backup first!

Only use in absolute emergencies!"""
        
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    else:
        message = "❓ <b>Unknown Action</b>\n\nThis feature is not yet implemented."
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))

# ============ GUESS THE NUMBER GAME COMMANDS ============

@check_banned
async def guess_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start a new Guess the Number game"""
    user = update.effective_user
    chat_id = update.effective_chat.id
    
    try:
        player = bot_instance.find_player_by_identifier(str(user.id))
        if not player:
            await update.message.reply_text(
                "❌ <b>Not registered!</b>\n\n"
                "Please use /start to register first!",
                parse_mode='HTML'
            )
            return
        
        active_game = bot_instance.get_guess_game(user.id)
        if active_game:
            transition_msg, has_conflicts = bot_instance.create_game_switch_message(user.id, "a new Guess Game")
            if has_conflicts:
                await update.message.reply_text(transition_msg, parse_mode='HTML')
                return
        
        args = context.args
        if args and len(args) > 0:
            difficulty = args[0].lower()
            if difficulty in bot_instance.guess_difficulties:
                unlocked_levels = bot_instance.get_unlocked_levels(user.id)
                if difficulty not in unlocked_levels:
                    await update.message.reply_text(
                        f"🔒 <b>Level Locked!</b>\n\n"
                        f"You need to complete easier levels first to unlock <b>{difficulty.title()}</b>.\n"
                        f"Currently unlocked: {', '.join(unlocked_levels)}",
                        parse_mode='HTML'
                    )
                    return
                    
                game = bot_instance.create_guess_game(
                    user.id, difficulty, player['display_name'], chat_id
                )
                if game:
                    message = format_guess_game_start(game)
                    await update.message.reply_text(message, parse_mode='HTML')
                else:
                    await update.message.reply_text(
                        "❌ <b>Error!</b> Failed to start game. Please try again.",
                        parse_mode='HTML'
                    )
                return
            else:
                await update.message.reply_text(
                    f"❌ <b>Invalid difficulty!</b>\n\n"
                    f"Available: beginner, easy, medium, hard, expert",
                    parse_mode='HTML'
                )
                return
        
        unlocked_levels = bot_instance.get_unlocked_levels(user.id)
        
        keyboard = []
        for difficulty, config in bot_instance.guess_difficulties.items():
            range_str = f"{config['range'][0]}-{config['range'][1]}"
            
            if difficulty in unlocked_levels:
                button_text = f"{config['emoji']} {difficulty.title()} ({range_str})"
                keyboard.append([InlineKeyboardButton(
                    button_text, callback_data=f"guess_start:{difficulty}"
                )])
            else:
                button_text = f"🔒 {difficulty.title()} - LOCKED"
                keyboard.append([InlineKeyboardButton(
                    button_text, callback_data=f"guess_locked:{difficulty}"
                )])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        unlocked_count = len(unlocked_levels)
        total_levels = len(bot_instance.guess_difficulties)
        
        message = (
            "🎲 <b>GUESS THE NUMBER</b> 🎯\n\n"
            f"🔄 <b>Progress:</b> {unlocked_count}/{total_levels} levels unlocked\n"
            f"✅ <b>Available:</b> {', '.join(unlocked_levels)}\n\n"
            "🎮 <b>Select difficulty to play:</b>\n\n"
            "💡 <b>Win games to unlock harder levels!</b>\n"
            "🏆 <b>Higher difficulties = Higher scores!</b>"
        )
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Guess command error for user {user.id}: {e}")
        await update.message.reply_text(
            "❌ <b>Error!</b> Please try again later.",
            parse_mode='HTML'
        )

@check_banned
async def guess_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """View guess game statistics - Public command with personal stats"""
    user = update.effective_user
    
    bot_instance.create_or_update_player(user.id, user.username or "", user.full_name or user.first_name or f"User{user.id}")
    
    try:
        personal_stats = bot_instance.get_guess_game_stats(user.id)
        
        global_stats = bot_instance.get_guess_game_stats()
        
        message = (
            f"🎯 <b>ARENA OF CHAMPIONS - GUESS STATISTICS</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"👤 <b>YOUR STATS</b>\n"
            f"🎮 <b>Games Played:</b> {personal_stats.get('total_games', 0)}\n"
            f"🏆 <b>Games Won:</b> {personal_stats.get('games_won', 0)}\n"
            f"📈 <b>Win Rate:</b> {personal_stats.get('win_rate', 0)}%\n"
            f"💎 <b>Best Score:</b> {personal_stats.get('highest_score', 0)}\n"
            f"⚡ <b>Perfect Guesses:</b> {personal_stats.get('perfect_guesses', 0)}\n"
            f"🌟 <b>Daily Challenges:</b> {personal_stats.get('daily_completed', 0)}\n\n"
            f"🌍 <b>GLOBAL STATS</b>\n"
            f"🎮 <b>Total Games:</b> {global_stats.get('total_games', 0)}\n"
            f"🏆 <b>Games Won:</b> {global_stats.get('games_won', 0)}\n"
            f"📈 <b>Global Win Rate:</b> {global_stats.get('win_rate', 0)}%\n"
            f"💎 <b>Highest Score:</b> {global_stats.get('highest_score', 0)}\n"
            f"⚡ <b>Total Perfect Guesses:</b> {global_stats.get('total_perfect', 0)}\n"
            f"👥 <b>Active Players:</b> {global_stats.get('active_players', 0)}\n\n"
            f"🎯 <b>Want to play?</b> Use /guess to start!"
        )
        
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        keyboard = [
            [InlineKeyboardButton("🎯 Play Guess", callback_data="start_guess"),
             InlineKeyboardButton("🏆 Leaderboard", callback_data="guess_leaderboard")],
            [InlineKeyboardButton("🎯 Daily Challenge", callback_data="daily_challenge")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, parse_mode='HTML', reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"Error in guess_stats_command: {e}")
        await update.message.reply_text(
            "❌ <b>Error retrieving statistics!</b>\n\n"
            "Please try again later.",
            parse_mode='HTML'
        )

@check_banned
async def guess_leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show leaderboard command options"""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    
    keyboard = [
        [InlineKeyboardButton("🏆 Highest Scores", callback_data="guess_lb_highest")],
        [InlineKeyboardButton("🎯 Total Scores", callback_data="guess_lb_total")],
        [InlineKeyboardButton("🎮 Most Games", callback_data="guess_lb_games")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    message = (
        "📊 <b>ARENA OF CHAMPIONS - GUESS LEADERBOARDS</b>\n\n"
        "Choose which leaderboard you want to view:\n\n"
        "🏆 <b>Highest Scores</b> - Best single game scores\n"
        "🎯 <b>Total Scores</b> - Cumulative points earned\n"
        " <b>Most Games</b> - Players with most games played\n\n"
        "👆 <b>Click the buttons below to view:</b>"
    )
    
    await update.message.reply_text(
        message,
        parse_mode='HTML',
        reply_markup=reply_markup
    )

async def show_guess_leaderboard_highest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show highest scores leaderboard"""
    from telegram.helpers import escape_markdown as H
    
    query = update.callback_query
    if query:
        await query.answer()
    
    try:
        conn = bot_instance.get_db_connection()
        leaderboard = []
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT p.display_name, p.highest_score, 
                           (SELECT COUNT(*) FROM guess_games gg WHERE gg.telegram_id = p.telegram_id) as games_played,
                           (SELECT COUNT(*) FROM guess_games gg WHERE gg.telegram_id = p.telegram_id AND gg.game_outcome = 'won') as games_won
                    FROM players p 
                    WHERE p.highest_score > 0
                    ORDER BY p.highest_score DESC 
                    LIMIT 10
                """)
                results = cursor.fetchall()
                if results:
                    for row in results:
                        leaderboard.append({
                            'player_name': row[0] or 'Unknown Player',
                            'highest_score': row[1] or 0,
                            'games_played': row[2] or 0,
                            'games_won': row[3] or 0
                        })
            except Exception as e:
                logger.error(f"Error fetching highest score leaderboard: {e}")
            finally:
                bot_instance.return_db_connection(conn)
        
        if not leaderboard:
            message = (
                "📊 <b>HIGHEST SCORES LEADERBOARD</b>\n\n"
                "🚫 <b>No data available yet!</b>\n\n"
                "Play some games to populate the leaderboard! 🎲"
            )
        else:
            message = "📊 <b>GUESS GAME LEADERBOARD</b>\n"
            message += "🏆 <b>HIGHEST SCORES</b>\n\n"
            
            for i, player in enumerate(leaderboard, 1):
                name = H(player.get('player_name', 'Unknown'))
                
                if i == 1:
                    pos = "🥇"
                elif i == 2:
                    pos = "🥈"
                elif i == 3:
                    pos = "🥉"
                else:
                    pos = f"{i}."
                
                value = player.get('highest_score', 0)
                games = player.get('games_played', 0)
                won = player.get('games_won', 0)
                message += f"{pos} <b>{name}</b> - {value} pts\n"
                message += f"    📈 {won}/{games} games won\n"
            
            message += f"\n🎲 Use /guess to compete for the top spot!"
        
        if query:
            await query.edit_message_text(message, parse_mode='HTML')
        else:
            await update.message.reply_text(message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Highest scores leaderboard error: {e}")
        error_message = "❌ <b>Error!</b> Failed to load leaderboard."
        if query:
            await query.edit_message_text(error_message, parse_mode='HTML')
        else:
            await update.message.reply_text(error_message, parse_mode='HTML')

async def show_guess_leaderboard_total(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show total scores leaderboard"""
    from telegram.helpers import escape_markdown as H
    
    query = update.callback_query
    if query:
        await query.answer()
    
    try:
        conn = bot_instance.get_db_connection()
        leaderboard = []
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT p.display_name, p.total_score,
                           (SELECT COUNT(*) FROM guess_games gg WHERE gg.telegram_id = p.telegram_id) as games_played,
                           (SELECT COUNT(*) FROM guess_games gg WHERE gg.telegram_id = p.telegram_id AND gg.game_outcome = 'won') as games_won
                    FROM players p 
                    WHERE p.total_score > 0
                    ORDER BY p.total_score DESC 
                    LIMIT 10
                """)
                results = cursor.fetchall()
                if results:
                    for row in results:
                        leaderboard.append({
                            'player_name': row[0] or 'Unknown Player',
                            'total_score': row[1] or 0,
                            'games_played': row[2] or 0,
                            'games_won': row[3] or 0
                        })
            except Exception as e:
                logger.error(f"Error fetching total score leaderboard: {e}")
            finally:
                bot_instance.return_db_connection(conn)
        
        if not leaderboard:
            message = (
                "📊 <b>TOTAL SCORES LEADERBOARD</b>\n\n"
                "🚫 <b>No data available yet!</b>\n\n"
                "Play some games to populate the leaderboard! 🎲"
            )
        else:
            message = "📊 <b>GUESS GAME LEADERBOARD</b>\n"
            message += "🎯 <b>TOTAL SCORES</b>\n\n"
            
            for i, player in enumerate(leaderboard, 1):
                name = H(player.get('player_name', 'Unknown'))
                
                if i == 1:
                    pos = "🥇"
                elif i == 2:
                    pos = "🥈"
                elif i == 3:
                    pos = "🥉"
                else:
                    pos = f"{i}."
                
                value = player.get('total_score', 0)
                games = player.get('games_played', 0)
                won = player.get('games_won', 0)
                avg = int(value / won) if won > 0 else 0
                message += f"{pos} <b>{name}</b> - {value} pts total\n"
                message += f"    📊 {won} wins, {avg} avg score\n"
            
            message += f"\n🎲 Use /guess to compete for the top spot!"
        
        if query:
            await query.edit_message_text(message, parse_mode='HTML')
        else:
            await update.message.reply_text(message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Total scores leaderboard error: {e}")
        error_message = "❌ <b>Error!</b> Failed to load leaderboard."
        if query:
            await query.edit_message_text(error_message, parse_mode='HTML')
        else:
            await update.message.reply_text(error_message, parse_mode='HTML')

async def show_guess_leaderboard_games(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show most games leaderboard"""
    from telegram.helpers import escape_markdown as H
    
    query = update.callback_query
    if query:
        await query.answer()
    
    try:
        leaderboard = bot_instance.get_guess_leaderboard('games', 10)
        
        if not leaderboard:
            message = (
                "📊 <b>MOST GAMES LEADERBOARD</b>\n\n"
                "🚫 <b>No data available yet!</b>\n\n"
                "Play some games to populate the leaderboard! 🎲"
            )
        else:
            message = "📊 <b>GUESS GAME LEADERBOARD</b>\n"
            message += "🎮 <b>MOST GAMES</b>\n\n"
            
            for i, player in enumerate(leaderboard, 1):
                name = H(player.get('player_name', 'Unknown'))
                
                if i == 1:
                    pos = "🥇"
                elif i == 2:
                    pos = "🥈"
                elif i == 3:
                    pos = "🥉"
                else:
                    pos = f"{i}."
                
                value = player.get('games_played', 0)
                won = player.get('games_won', 0)
                win_rate = (won / value * 100) if value > 0 else 0
                message += f"{pos} <b>{name}</b> - {value} games ({win_rate:.1f}% win)\n"
            
            message += f"\n🎲 Use /guess to compete for the top spot!"
        
        if query:
            await query.edit_message_text(message, parse_mode='HTML')
        else:
            await update.message.reply_text(message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Most games leaderboard error: {e}")
        error_message = "❌ <b>Error!</b> Failed to load leaderboard."
        if query:
            await query.edit_message_text(error_message, parse_mode='HTML')
        else:
            await update.message.reply_text(error_message, parse_mode='HTML')

@check_banned
async def daily_guess_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Play daily guess challenge - ONE ATTEMPT ONLY per day"""
    user = update.effective_user
    chat_id = update.effective_chat.id
    today = date.today().isoformat()
    
    try:
        player = bot_instance.find_player_by_identifier(str(user.id))
        if not player:
            await update.message.reply_text(
                "❌ <b>Not registered!</b>\n\n"
                "Please use /start to register first!",
                parse_mode='HTML'
            )
            return
        
        conn = bot_instance.get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT game_outcome, final_score, guesses_used FROM guess_games 
                    WHERE telegram_id = %s AND daily_challenge = TRUE 
                    AND challenge_date = %s
                """, (user.id, today))
                previous_attempt = cursor.fetchone()
                
                if previous_attempt:
                    outcome, score, guesses = previous_attempt
                    status_emoji = "🏆" if outcome == 'won' else "❌"
                    result_text = "Won" if outcome == 'won' else ("Lost" if outcome == 'lost' else "Timed Out")
                    
                    await update.message.reply_text(
                        f"{status_emoji} <b>Daily Challenge - Already Attempted!</b>\n\n"
                        f"📅 <b>Today's Result:</b> {result_text}\n"
                        f"📊 <b>Score:</b> {score}\n"
                        f"🎯 <b>Guesses Used:</b> {guesses}\n\n"
                        f"⚠️ <b>One attempt per day only!</b>\n"
                        f"🎯 Come back tomorrow for a new challenge.\n\n"
                        f"💡 Try /guess for regular games!",
                        parse_mode='HTML'
                    )
                    return
            finally:
                bot_instance.return_db_connection(conn)
        
        import hashlib
        
        combined_seed = f"{today}_spl_secret_2024_{user.id}"
        date_seed = int(hashlib.md5(combined_seed.encode()).hexdigest(), 16) % 1000000
        random.seed(date_seed)
        
        challenge_difficulty = 'medium'  # Use medium difficulty for daily challenges
        target = random.randint(1, 100)  # Each player gets unique number
        
        random.seed()
        
        game_state = {
            'user_id': user.id,
            'player_name': player['display_name'],
            'chat_id': chat_id,
            'difficulty': challenge_difficulty,
            'target_number': target,
            'attempts_used': 0,
            'max_attempts': 8,  # Reduced attempts to make sharing less effective
            'time_limit': 240,  # 4 minutes - shorter time limit
            'start_time': time.time(),
            'hint_used': False,
            'guesses': [],
            'game_active': True,
            'range_min': 1,
            'range_max': 100,
            'is_daily_challenge': True,
            'challenge_date': today
        }
        
        bot_instance.guess_games[user.id] = game_state
        
        message = (
            "🌟 <b>DAILY CHALLENGE</b> 🎯\n\n"
            f"📅 <b>Date:</b> {today}\n"
            f"🎮 <b>Player:</b> {H(player['display_name'])}\n"
            f"🎯 <b>Range:</b> 1-100\n"
            f"🎲 <b>Attempts:</b> 8 (Only ONE chance per day!)\n"
            f"⏱️ <b>Time Limit:</b> 4 minutes\n\n"
            f"🎁 <b>Challenge Bonus:</b> +50% score multiplier!\n"
            f"🎮 <b>Each player gets a unique number!</b>\n"
            f"⚠️ <b>WARNING: Only ONE attempt per day allowed!</b>\n"
            f"🚫 <b>No retries - Win or lose, you're done!</b>\n\n"
            f"💭 Type your guess (1-100):"
        )
        
        await update.message.reply_text(message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Daily challenge error for user {user.id}: {e}")
        await update.message.reply_text(
            "❌ <b>Error!</b> Please try again later.",
            parse_mode='HTML'
        )

async def cleanup_guess_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Force cleanup of expired guess games (Admin only)"""
    user_id = update.effective_user.id
    
    if not bot_instance.is_admin(user_id):
        await update.message.reply_text(
            "❌ <b>ACCESS DENIED!</b>\n\n"
            "🛡️ Admin access required.",
            parse_mode='HTML'
        )
        return
    
    try:
        cleaned = bot_instance.cleanup_expired_guess_games()
        
        message = (
            "🧹 <b>GUESS GAME CLEANUP</b>\n\n"
            f"🗑️ <b>Cleaned up:</b> {cleaned} expired games\n"
            f"🎮 <b>Active games:</b> {len(bot_instance.guess_games)}\n\n"
            "✅ Cleanup completed successfully!"
        )
        
        await update.message.reply_text(message, parse_mode='HTML')
        logger.info(f"Admin {user_id} performed guess game cleanup: {cleaned} games removed")
        
    except Exception as e:
        logger.error(f"Cleanup guess command error: {e}")
        await update.message.reply_text(
            "❌ <b>Error!</b> Failed to perform cleanup.",
            parse_mode='HTML'
        )

def format_guess_game_start(game: dict) -> str:
    """Format the initial game state message"""
    config = bot_instance.guess_difficulties[game['difficulty']]
    range_str = f"{config['range'][0]}-{config['range'][1]}"
    
    message = (
        f"🎲 <b>GUESS THE NUMBER</b> {config['emoji']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>Difficulty:</b> {game['difficulty'].title()}\n"
        f"📊 <b>Range:</b> {range_str}\n"
        f"🎮 <b>Attempts:</b> {game['max_attempts']}\n"
        f"⏰ <b>Time Limit:</b> {game['time_limit']}s\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💭 I'm thinking of a number...\n"
        f"🎯 Reply with your guess!"
    )
    return message

async def reset_all_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset database for new Arena Of Champions bot (Super Admin only)"""
    if not bot_instance.is_super_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n👑 Only Super Admin can reset data!", parse_mode='HTML')
        return
    
    if len(context.args) == 0 or context.args[0] != 'CONFIRM':
        await update.message.reply_text(
            "⚠️ <b>ARENA OF CHAMPIONS RESET</b> ⚠️\n\n"
            "🔥 <b>This will CLEAR:</b>\n"
            "• All shard balances (set to 0)\n"
            "• All game statistics\n"
            "• All chase game records\n"
            "• All guess game records\n"
            "• All daily reward streaks\n"
            "• All daily bonus claims\n"
            "• All transaction history\n\n"
            "✅ <b>This will PRESERVE:</b>\n"
            "• User registrations\n"
            "• All achievements earned\n"
            "• User titles\n"
            "• Admin privileges\n\n"
            "⚡ <b>To proceed, use:</b>\n"
            "<code>/resetall CONFIRM</code>",
            parse_mode='HTML'
        )
        return
    
    conn = bot_instance.get_db_connection()
    if not conn:
        await update.message.reply_text("❌ <b>Database connection failed!</b>", parse_mode='HTML')
        return
    
    try:
        cursor = conn.cursor()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS players_backup_{timestamp} AS 
            SELECT * FROM players
        """)
        
        cursor.execute("UPDATE players SET shards = 0")
        
        try:
            cursor.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name='players' AND column_name='daily_streak'
            """)
            if cursor.fetchone():
                cursor.execute("UPDATE players SET daily_streak = 0")
        except Exception as e:
            logger.info(f"Daily streak column not found: {e}")
            
        try:
            cursor.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name='players' AND column_name='last_daily_claim'
            """)
            if cursor.fetchone():
                cursor.execute("UPDATE players SET last_daily_claim = NULL")
        except Exception as e:
            logger.info(f"Last daily claim column not found: {e}")
        
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'shard_transactions'
            )
        """)
        if cursor.fetchone()[0]:
            cursor.execute("DELETE FROM shard_transactions")
        
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'daily_shard_bonuses'
            )
        """)
        if cursor.fetchone()[0]:
            cursor.execute("DELETE FROM daily_shard_bonuses")
        
        conn.commit()
        
        await update.message.reply_text(
            "🏆 <b>ARENA OF CHAMPIONS RESET COMPLETE!</b> 🏆\n\n"
            "🔥 <b>Data Cleared:</b>\n"
            "• ✅ All shard balances reset to 0\n"
            "• ✅ Game statistics cleared\n"
            "• ✅ Daily streaks reset\n"
            "• ✅ Daily bonus claims cleared\n"
            "• ✅ Transaction history cleared\n\n"
            "💎 <b>Data Preserved:</b>\n"
            "• ✅ User registrations intact\n"
            "• ✅ All achievements kept\n"
            "• ✅ User titles maintained\n\n"
            f"💾 <b>Backup created:</b> players_backup_{timestamp}\n\n"
            "🚀 <b>Arena Of Champions ready for fresh battles!</b>",
            parse_mode='HTML'
        )
        
    except Exception as e:
        conn.rollback()
        await update.message.reply_text(
            f"❌ <b>RESET FAILED!</b>\n\nError: {str(e)[:100]}",
            parse_mode='HTML'
        )
    finally:
        bot_instance.return_db_connection(conn)

async def list_players_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all registered players (Super Admin only)"""
    if not bot_instance.is_super_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n👑 Only Super Admin can view player list!", parse_mode='HTML')
        return
    
    conn = bot_instance.get_db_connection()
    if not conn:
        await update.message.reply_text("❌ <b>Database error!</b>\n\nPlease try again later.", parse_mode='HTML')
        return
    
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT p.id, p.telegram_id, p.username, p.display_name, p.title, p.created_at,
                COUNT(a.id) as achievement_count,
                SUM(a.count) as total_awards
            FROM players p
            LEFT JOIN achievements a ON p.id = a.player_id
            GROUP BY p.id, p.telegram_id, p.username, p.display_name, p.title, p.created_at
            ORDER BY p.created_at DESC
        """)
        
        players = cursor.fetchall()
        
        if not players:
            await update.message.reply_text(
                "📝 <b>NO PLAYERS REGISTERED</b>\n\n"
                "🚫 No players found in database.",
                parse_mode='HTML'
            )
            return
        
        message = f"👥 <b>REGISTERED PLAYERS ({len(players)})</b> 📋\n\n"
        message += "━━━━━━━━━━━━━━━━━━━━\n"
        
        for i, player in enumerate(players, 1):
            player_id, telegram_id, username, display_name, title, created_at, ach_count, total_awards = player
            
            message += f"<b>{i}.</b> {H(display_name)}\n"
            
            if username:
                message += f"   📱 @{H(username)}\n"
            
            message += f"   🆔 {telegram_id}\n"
            
            if title:
                message += f"   👑 <b>{H(title)}</b>\n"
            
            achievements_text = f"{ach_count or 0} types" if ach_count else "0 types"
            awards_text = f"{total_awards or 0} total" if total_awards else "0 total"
            message += f"   🏆 {achievements_text}, {awards_text}\n"
            
            reg_date = created_at.strftime('%m/%d/%Y')
            message += f"   📅 {reg_date}\n\n"
            
            if len(message) > 3500:  # Telegram limit is ~4096
                message += "━━━━━━━━━━━━━━━━━━━━\n"
                message += f"📊 <b>Showing {i} of {len(players)} players</b>"
                await update.message.reply_text(message, parse_mode='HTML')
                
                message = f"👥 <b>REGISTERED PLAYERS (continued)</b> 📋\n\n"
                message += "━━━━━━━━━━━━━━━━━━━━\n"
        
        if not message.endswith("players</b>"):  # Only if we haven't sent a split message
            message += "━━━━━━━━━━━━━━━━━━━━\n"
            message += f"📊 <b>Total Players:</b> {len(players)}"
            await update.message.reply_text(message, parse_mode='HTML')
            
    except Exception as e:
        logger.error(f"Error in list_players command: {e}")
        await update.message.reply_text(
            "❌ <b>Something went wrong!</b>\n\nPlease try again later.",
            parse_mode='HTML'
        )
    finally:
        bot_instance.return_db_connection(conn)

async def bot_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show bot and database status with statistics (Super Admin only)"""
    if not bot_instance.is_super_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n👑 Only Super Admin can view bot status!", parse_mode='HTML')
        return
    
    conn = bot_instance.get_db_connection()
    if not conn:
        await update.message.reply_text("❌ <b>Database error!</b>\n\nCannot connect to database.", parse_mode='HTML')
        return
    
    try:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM players")
        total_users = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT COUNT(*) FROM players WHERE title IS NOT NULL AND title != ''")
        users_with_titles = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT COUNT(*) FROM achievements")
        total_achievement_records = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT SUM(count) FROM achievements")
        total_awards = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT COUNT(DISTINCT achievement_name) FROM achievements")
        unique_achievements = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT COUNT(*) FROM admins")
        admin_count = cursor.fetchone()[0] or 0
        
        cursor.execute("""
            SELECT COUNT(*) FROM achievements 
            WHERE created_at >= NOW() - INTERVAL '7 days'
        """)
        recent_achievements = cursor.fetchone()[0] or 0
        
        cursor.execute("""
            SELECT COUNT(*) FROM players 
            WHERE created_at >= NOW() - INTERVAL '7 days'
        """)
        recent_registrations = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT SUM(shards) FROM players WHERE shards > 0")
        total_shard_circulation = cursor.fetchone()[0] or 0
        
        cursor.execute("""
            SELECT p.display_name, SUM(a.count) as total_awards
            FROM players p
            JOIN achievements a ON p.id = a.player_id
            GROUP BY p.id, p.display_name
            ORDER BY total_awards DESC
            LIMIT 3
        """)
        top_achievers = cursor.fetchall()
        
        from datetime import datetime
        current_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        
        message = f"🤖 <b>ARENA OF CHAMPIONS STATUS</b> 📊\n\n"
        message += f"⏰ <b>Status Check:</b> {current_time}\n"
        message += f"━━━━━━━━━━━━━━━━━━━━\n\n"
        
        message += "💾 <b>DATABASE STATUS:</b>\n"
        message += f"✅ <b>Connection:</b> Active\n"
        message += f"👥 <b>Total Users:</b> {total_users}\n"
        message += f"👑 <b>Users with Titles:</b> {users_with_titles}\n"
        message += f"🛡️ <b>Total Admins:</b> {admin_count}\n"
        message += f"💠 <b>Total Shard Circulation:</b> {total_shard_circulation:,} 💠\n\n"
        
        message += "🏆 <b>ACHIEVEMENT STATS:</b>\n"
        message += f"🎯 <b>Total Awards Given:</b> {total_awards}\n"
        message += f"📝 <b>Achievement Records:</b> {total_achievement_records}\n"
        message += f"🔢 <b>Unique Types:</b> {unique_achievements}\n\n"
        
        message += "📈 <b>RECENT ACTIVITY (7 days):</b>\n"
        message += f"🏅 <b>New Awards:</b> {recent_achievements}\n"
        message += f"👤 <b>New Users:</b> {recent_registrations}\n\n"
        
        if top_achievers:
            message += "🥇 <b>TOP ACHIEVERS:</b>\n"
            for i, (name, awards) in enumerate(top_achievers, 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                message += f"{medal} {name}: <b>{awards}</b> awards\n"
        else:
            message += "🥇 <b>TOP ACHIEVERS:</b> None yet\n"
        
        message += f"\n━━━━━━━━━━━━━━━━━━━━\n"
        message += f"🚀 <b>Bot Status:</b> Running Smoothly ✅"
        
        await update.message.reply_text(message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Error in bot_status command: {e}")
        await update.message.reply_text(
            "❌ <b>Status Check Failed!</b>\n\n"
            f"Error: {str(e)}\n\n"
            "Please contact developer if this persists.",
            parse_mode='HTML'
        )
    finally:
        bot_instance.return_db_connection(conn)

# ============ MISSING ADMIN PANEL COMMANDS ============

async def finduser_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Find user information (Admin only)"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text(
            "❌ <b>ACCESS DENIED!</b>\n\n🛡️ This command requires administrator privileges.",
            parse_mode='HTML'
        )
        return

    if len(context.args) < 1:
        await update.message.reply_text(
            "❌ <b>USAGE ERROR!</b>\n\n"
            "<b>Usage:</b> /finduser @username or /finduser UserID\n"
            "<b>Example:</b> /finduser @player123 or /finduser 123456789",
            parse_mode='HTML'
        )
        return

    identifier = context.args[0]
    player = bot_instance.find_player_by_identifier(identifier)
    
    if not player:
        await update.message.reply_text(
            f"❌ <b>USER NOT FOUND!</b>\n\n"
            f"🔍 No user found with identifier: <b>{identifier}</b>",
            parse_mode='HTML'
        )
        return
    
    achievements = bot_instance.get_player_achievements(player['id'])
    total_achievements = sum(count for _, count in achievements)
    
    message = f"""👤 <b>USER INFORMATION</b>

━━━━━━━━━━━━━━━━━━━━
👤 <b>Name:</b> {player['display_name']}
🆔 <b>Telegram ID:</b> {player['telegram_id']}
{"📧 <b>Username:</b> @" + player['username'] if player.get('username') else "📧 <b>Username:</b> Not set"}
💰 <b>Shards:</b> {player.get('shards', 0):,}
🏆 <b>Total Achievements:</b> {total_achievements}
{"👑 <b>Title:</b> " + player['title'] if player.get('title') else "👑 <b>Title:</b> None"}
📅 <b>Last Active:</b> {player.get('updated_at', 'Unknown')}
━━━━━━━━━━━━━━━━━━━━"""
    
    await update.message.reply_text(message, parse_mode='HTML')

async def unbanuser_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Unban user (Admin only)"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text(
            "❌ <b>ACCESS DENIED!</b>\n\n🛡️ This command requires administrator privileges.",
            parse_mode='HTML'
        )
        return

    if len(context.args) < 1:
        await update.message.reply_text(
            "❌ <b>USAGE ERROR!</b>\n\n"
            "<b>Usage:</b> /unbanuser @username\n"
            "<b>Example:</b> /unbanuser @rehabilitateduser",
            parse_mode='HTML'
        )
        return

    identifier = context.args[0]
    player = bot_instance.find_player_by_identifier(identifier)
    
    if not player:
        await update.message.reply_text(
            f"❌ <b>USER NOT FOUND!</b>\n\n"
            f"🔍 No user found with identifier: <b>{identifier}</b>",
            parse_mode='HTML'
        )
        return
    
    try:
        success = bot_instance.unban_user(player['telegram_id'])
        
        if success:
            await update.message.reply_text(
                f"✅ <b>USER UNBANNED SUCCESSFULLY</b>\n\n"
                f"👤 <b>Unbanned:</b> {player['display_name']}\n"
                f"� <b>Unbanned by:</b> {update.effective_user.first_name}\n\n"
                f"🎉 User can now access bot functions again.",
                parse_mode='HTML'
            )
        else:
            await update.message.reply_text(
                f"❌ <b>UNBAN FAILED</b>\n\n"
                f"Failed to unban user {player['display_name']}. User may not be banned.",
                parse_mode='HTML'
            )
    except Exception as e:
        log_exception("unban_user_command", e, update.effective_user.id)
        await update.message.reply_text(
            "❌ <b>ERROR</b>\n\nFailed to execute unban command. Please try again.",
            parse_mode='HTML'
        )

async def restart_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Restart the bot (Super Admin only)"""
    if not bot_instance.is_super_admin(update.effective_user.id):
        await update.message.reply_text(
            "❌ <b>ACCESS DENIED!</b>\n\n👑 Only Super Admin can restart the bot!",
            parse_mode='HTML'
        )
        return

    try:
        await update.message.reply_text(
            "🔄 <b>BOT RESTART INITIATED!</b>\n\n"
            "⚠️ All active games will be lost!\n"
            "🕒 Bot will restart in a few seconds...",
            parse_mode='HTML'
        )
        
        logger.info(f"Bot restart requested by Super Admin {update.effective_user.id}")
        
        await asyncio.sleep(2)
        
        import os
        import sys
        
        bot_instance.shutdown_flag = True
        
        logger.info("Bot shutting down for restart...")
        os._exit(0)
        
    except Exception as e:
        log_exception("restart_command", e, update.effective_user.id)
        await update.message.reply_text(
            "❌ <b>RESTART FAILED</b>\n\nError occurred during restart. Please check logs.",
            parse_mode='HTML'
        )
    

# ============ GUESS GAME CALLBACK HANDLERS ============

async def guess_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle guess game callback queries"""
    query = update.callback_query
    
    user = query.from_user
    data = query.data
    
    try:
        if data == 'guess_hint':
            game = bot_instance.get_guess_game(user.id)
            if not game or not game.get('game_active'):
                await query.answer("⚠️ You don't have an active guess game!", show_alert=True)
                return
        
        await query.answer()
        
        if data.startswith('guess_start:'):
            difficulty = data.split(':')[1]
            
            player = bot_instance.find_player_by_identifier(str(user.id))
            if not player:
                await safe_edit_message(query,
                    "❌ <b>Not registered!</b>\n\n"
                    "Please use /start to register first!",
                    parse_mode='HTML'
                )
                return
            
            unlocked_levels = bot_instance.get_unlocked_levels(user.id)
            if difficulty not in unlocked_levels:
                await safe_edit_message(query,
                    f"🔒 <b>Level Locked!</b>\n\n"
                    f"You need to complete easier levels first to unlock <b>{difficulty.title()}</b>.\n"
                    f"Currently unlocked: {', '.join(unlocked_levels)}",
                    parse_mode='HTML'
                )
                return
            
            active_game = bot_instance.get_guess_game(user.id)
            if active_game:
                transition_msg, has_conflicts = bot_instance.create_game_switch_message(user.id, f"{difficulty.title()} Guess Game")
                if has_conflicts:
                    await safe_edit_message(query, transition_msg, parse_mode='HTML')
                    return
            
            game = bot_instance.create_guess_game(
                user.id, difficulty, player['display_name'], query.message.chat_id
            )
            
            if game:
                message = format_guess_game_start(game)
                
                keyboard = [[InlineKeyboardButton("🍀 Lucky Hint", callback_data="guess_hint")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await safe_edit_message(query, message, parse_mode='HTML', reply_markup=reply_markup)
            else:
                await safe_edit_message(query,
                    "❌ <b>Error!</b> Failed to start game. Please try again.",
                    parse_mode='HTML'
                )
        
        elif data.startswith('guess_locked:'):
            difficulty = data.split(':')[1]
            unlocked_levels = bot_instance.get_unlocked_levels(user.id)
            
            await safe_edit_message(query,
                f"🔒 <b>Level Locked!</b>\n\n"
                f"<b>{difficulty.title()}</b> is not yet available.\n"
                f"Complete easier levels to unlock it!\n\n"
                f"🔓 <b>Currently unlocked:</b> {', '.join(unlocked_levels)}\n\n"
                f"💡 <b>Tip:</b> Win games to progressively unlock harder levels!",
                parse_mode='HTML'
            )
                
        elif data == 'guess_hint':
            game = bot_instance.get_guess_game(user.id)
            if not game or not game.get('game_active'):
                await safe_edit_message(query,
                    "❌ <b>No active game!</b>\n\nUse /guess to start a new game.",
                    parse_mode='HTML'
                )
                return
            
            if game['hint_used']:
                await query.answer("You've already used your hint for this game!", show_alert=True)
                return
            
            hint = bot_instance.generate_hint(game)
            game['hint_used'] = True
            
            config = bot_instance.guess_difficulties[game['difficulty']]
            range_str = f"{config['range'][0]}-{config['range'][1]}"
            elapsed = int(time.time() - game['start_time'])
            time_left = max(0, game['time_limit'] - elapsed)
            
            message = (
                f"🎲 <b>GUESS THE NUMBER</b> {config['emoji']}\n\n"
                f"🎯 <b>Difficulty:</b> {game['difficulty'].title()}\n"
                f"📊 <b>Range:</b> {range_str}\n"
                f"🎮 <b>Attempts Left:</b> {game['max_attempts'] - game['attempts_used']}\n"
                f"⏰ <b>Time Left:</b> {time_left}s\n\n"
                f"{hint}\n\n"
                f"💭 <b>I'm thinking of a number...</b>\n"
                f"🎯 Reply with your guess!"
            )
            
            await safe_edit_message(query, message, parse_mode='HTML')
            
        elif data == 'guess_lb_highest':
            await show_guess_leaderboard_highest(update, context)
            
        elif data == 'guess_lb_total':
            await show_guess_leaderboard_total(update, context)
            
        elif data == 'guess_lb_games':
            await show_guess_leaderboard_games(update, context)
            
    except BadRequest as e:
        logger.warning(f"BadRequest in guess_callback: {e}")
    except Exception as e:
        logger.error(f"Guess callback error: {e}")
        try:
            await query.answer("❌ Error processing request.", show_alert=True)
        except:
            pass

async def handle_manual_auction_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle manual auction input - captain bids and admin controls"""
    try:
        user = update.effective_user
        text = update.message.text.strip()
        chat_id = update.effective_chat.id
        
        if not bot_instance.approved_auctions:
            return
        
        has_active_auction = any(
            auction.status == "active" 
            for auction in bot_instance.approved_auctions.values()
        )
        if not has_active_auction:
            return  # No active auctions, skip processing
        
        active_auction = None
        user_captain = None
        
        for auction_id, auction in bot_instance.approved_auctions.items():
            if auction.status == "active" and auction.current_player:
                for captain in auction.approved_captains.values():
                    if captain.user_id == user.id:
                        active_auction = auction
                        user_captain = captain
                        break
                
                if not active_auction and (bot_instance.is_admin(user.id) or auction.creator_id == user.id):
                    active_auction = auction
                    break
        
        if not active_auction:
            return  # No active auction for this user
        
        if hasattr(active_auction, 'is_paused') and active_auction.is_paused:
            return  # Auction paused, ignore all input
        
        if active_auction.group_chat_id and chat_id != active_auction.group_chat_id:
            return  # Message not from the designated auction group chat
        
        logger.debug(f"Auction handler processing - User: {user.id}, Text: '{text}', Auction: {active_auction.id}")
        
        if active_auction.status != "active":
            return
        
        if bot_instance.is_admin(user.id) or active_auction.creator_id == user.id:
            if update.message.reply_to_message:
                if text == "..":
                    if active_auction.highest_bidder:
                        captain = None
                        for cap in active_auction.approved_captains.values():
                            if cap.user_id == active_auction.highest_bidder:
                                captain = cap
                                break
                        
                        keyboard = [
                            [
                                InlineKeyboardButton("✅ Confirm Sale", callback_data=f"confirm_sale_{active_auction.id}"),
                                InlineKeyboardButton("🔄 Continue Bidding", callback_data=f"continue_bid_{active_auction.id}")
                            ]
                        ]
                        reply_markup = InlineKeyboardMarkup(keyboard)
                        
                        confirm_message = (
                            f"⚠️ <b>GOING ONCE... GOING TWICE!</b> ⚠️\n\n"
                            f"👤 <b>{active_auction.current_player.name}</b>\n"
                            f"💰 <b>Final Bid:</b> {format_amount(active_auction.highest_bid)}\n"
                            f"👑 <b>Winning Team:</b> {captain.team_name if captain else 'Unknown'}\n\n"
                            f"❓ <b>Confirm this sale?</b>"
                        )
                        
                        await update.message.reply_text(
                            confirm_message,
                            parse_mode='HTML',
                            reply_markup=reply_markup
                        )
                    else:
                        unsold_message = (
                            f"📤 <b>UNSOLD!</b>\n\n"
                            f"👤 <b>{active_auction.current_player.name}</b>\n"
                            f"💰 No bids received (Base: {format_amount(active_auction.base_price)})"
                        )
                        
                        await update.message.reply_text(unsold_message, parse_mode='HTML')
                        
                        if not hasattr(active_auction, 'sold_players'):
                            active_auction.sold_players = {}
                        active_auction.sold_players[active_auction.current_player.user_id] = {
                            'player': active_auction.current_player,
                            'team': 'UNSOLD',
                            'amount': 0,
                            'captain': None
                        }
                        
                        active_auction.current_player_index += 1
                        
                        if active_auction.current_player_index < len(active_auction.player_queue):
                            next_player = active_auction.player_queue[active_auction.current_player_index]
                            active_auction.current_player = next_player
                            active_auction.highest_bidder = None  # Sync with current_bids
                            active_auction.highest_bid = active_auction.base_price
                            active_auction.current_bids = {}  # Reset bids for new player
                            
                            captain_purses = "\n".join([
                                f"👑 {cap.team_name}: {format_amount(cap.purse)}"
                                for cap in active_auction.approved_captains.values()
                            ])
                            
                            username_display = f"@{next_player.username}" if hasattr(next_player, 'username') and next_player.username else ""
                            next_message = (
                                f"🔥 <b>NEXT PLAYER</b>\n\n"
                                f"👤 <b>{next_player.name}</b> {username_display}\n"
                                f"💎 <b>Base:</b> {format_amount(next_player.base_price)}\n\n"
                                f"🎯 <b>Type amount to bid!</b>\n"
                                f"📝 Admin: Use /sell {active_auction.id} or reply '..' to sell"
                            )
                            
                            await update.message.reply_text(next_message, parse_mode='HTML')
                        else:
                            active_auction.status = "completed"
                            await update.message.reply_text(
                                f"🏆 <b>AUCTION COMPLETED!</b>\n\n🎊 All players auctioned for {active_auction.name}!",
                                parse_mode='HTML'
                            )
                    return
        
        logger.info(f"Manual auction input - User {user.id}, text: '{text}', user_captain: {user_captain is not None}")
        
        logger.info(f"Total approved auctions: {len(bot_instance.approved_auctions)}")
        for aid, auction in bot_instance.approved_auctions.items():
            logger.info(f"Auction {aid}: status={auction.status}, current_player={auction.current_player is not None}, captains={len(auction.approved_captains)}")
            logger.info(f"Auction {aid} approved captains: {[f'{cap.user_id}:{cap.name}:{cap.team_name}' for cap in auction.approved_captains.values()]}")
            
            if user.id in auction.approved_captains:
                logger.info(f"User {user.id} is a captain in auction {aid}")
            else:
                logger.info(f"User {user.id} is NOT a captain in auction {aid}")
        
        if active_auction:
            logger.info(f"Active auction found - ID: {active_auction.id}, Status: {active_auction.status}, Current highest_bid: {active_auction.highest_bid}, Current highest_bidder: {active_auction.highest_bidder}")
        else:
            logger.info(f"No active auction found for user {user.id}")
        
        if user_captain:
            logger.info(f"Captain found - Name: {user_captain.name}, Team: {user_captain.team_name}, Purse: {user_captain.purse}")
        else:
            logger.info(f"No captain found for user {user.id}")
        
        if user_captain:
            bid_amount = parse_bid_amount(text)
            logger.info(f"Parsed bid amount: {bid_amount}")
            if bid_amount:
                if not active_auction.current_player:
                    await update.message.reply_text(
                        "⚠️ <b>No active player!</b>\n\n"
                        "Wait for next player to be announced.",
                        parse_mode='HTML'
                    )
                    return
                
                min_increment = 0.5
                min_required_bid = active_auction.highest_bid + min_increment
                
                if bid_amount < min_required_bid:
                    await update.message.reply_text(
                        f"❌ <b>Bid too low!</b>\n\n"
                        f"💰 <b>Your bid:</b> {format_amount(bid_amount)}\n"
                        f"🏆 <b>Current highest:</b> {format_amount(active_auction.highest_bid)}\n"
                        f"📊 <b>Minimum increment:</b> {format_amount(min_increment)}\n\n"
                        f"💡 <b>Minimum required:</b> {format_amount(min_required_bid)}",
                        parse_mode='HTML'
                    )
                    return
                
                if bid_amount > user_captain.purse:
                    await update.message.reply_text(
                        f"❌ <b>Insufficient funds!</b>\n\n"
                        f"💰 <b>Bid amount:</b> {format_amount(bid_amount)}\n"
                        f"👑 <b>Team purse:</b> {format_amount(user_captain.purse)}\n\n"
                        f"💡 <b>Max bid:</b> {format_amount(user_captain.purse)}",
                        parse_mode='HTML'
                    )
                    return
                
                lock_acquired = active_auction._bid_lock.acquire(blocking=False)
                if not lock_acquired:
                    await update.message.reply_text("⏳ Previous bid processing, please wait...")
                    return
                
                try:
                    if not active_auction.current_player:
                        await update.message.reply_text("⚠️ Player state changed, please try again.", parse_mode='HTML')
                        return
                    
                    min_required_bid = active_auction.highest_bid + 0.5
                    if bid_amount < min_required_bid:
                        return
                    
                    active_auction.last_bid_time = datetime.now()
                    
                    active_auction.highest_bid = bid_amount
                    active_auction.highest_bidder = user.id
                    
                    if not hasattr(active_auction, 'current_bids') or active_auction.current_bids is None:
                        active_auction.current_bids = {}
                    
                    active_auction.current_bids[user.id] = {
                        'captain': user_captain,
                        'amount': bid_amount,
                        'timestamp': datetime.now()
                    }
                    
                    logger.info(f"Bid updated: {user_captain.name} bid {bid_amount}Cr")
                    
                    await update.message.reply_text(
                        f"✅ <b>Bid Accepted!</b>\n\n"
                        f"💰 <b>Amount:</b> {format_amount(bid_amount)}\n"
                        f"👑 <b>Team:</b> {user_captain.team_name}\n"
                        f"📈 <b>New Highest Bid!</b>\n\n"
                        f"💡 <i>Use /status to check auction state</i>",
                        parse_mode='HTML'
                    )
                    
                except Exception as bid_error:
                    logger.error(f"Error processing bid: {bid_error}")
                    try:
                        await update.message.reply_text("⚠️ Bid processing issue. Please check with admin.")
                    except:
                        pass
                finally:
                    active_auction._bid_lock.release()
                
    
    except Exception as e:
        logger.error(f"Error in handle_manual_auction_input: {e}")
        try:
            await update.message.reply_text("❌ Bid processing error. Please try again.")
        except:
            pass

@check_banned
async def auction_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show current auction status"""
    try:
        user = update.effective_user
        
        active_auction = None
        for auction in bot_instance.approved_auctions.values():
            if auction.status == "active":
                active_auction = auction
                break
        
        if not active_auction:
            await update.message.reply_text("❌ No active auction found.")
            return
        
        current_player = active_auction.current_player
        if not current_player:
            await update.message.reply_text("❌ No player currently being auctioned.")
            return
        
        user_captain = active_auction.approved_captains.get(user.id)
        captain_info = f"\n👑 <b>Your Team:</b> {user_captain.team_name} - {format_amount(user_captain.purse)}" if user_captain else ""
        
        leading_team = "None"
        if active_auction.highest_bidder:
            leading_captain = active_auction.approved_captains.get(active_auction.highest_bidder)
            if leading_captain:
                leading_team = leading_captain.team_name
        
        status_msg = (
            f"🔥 <b>AUCTION STATUS</b>\n\n"
            f"👤 <b>Current Player:</b> {current_player.name}\n"
            f"💰 <b>Highest Bid:</b> {format_amount(active_auction.highest_bid)}\n"
            f"👑 <b>Leading Team:</b> {leading_team}\n"
            f"💎 <b>Base Price:</b> {format_amount(active_auction.base_price)}"
            f"{captain_info}\n\n"
            f"💡 <b>Type amount to bid!</b>"
        )
        
        await update.message.reply_text(status_msg, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Error in auction_status_command: {e}")
        try:
            await update.message.reply_text("❌ Error getting auction status.")
        except:
            pass

@check_banned
async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check bot responsiveness and user connection"""
    try:
        start_time = datetime.now()
        await update.message.reply_text(
            "🏓 <b>Pong!</b>\n\n"
            "✅ <b>Bot is responsive</b>\n"
            "💡 If you're experiencing delays, try:\n"
            "• Use /status to check auction state\n"
            "• Wait for confirmations before sending new bids\n"
            "• Check your internet connection",
            parse_mode='HTML'
        )
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds() * 1000
        logger.info(f"Ping response time: {response_time:.0f}ms")
    except Exception as e:
        logger.error(f"Error in ping_command: {e}")

@check_banned
async def handle_guess_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle number guesses from users"""
    if update.edited_message:
        return
    
    user = update.effective_user
    text = update.message.text
    
    nightmare_handled = await handle_nightmare_guess(update, context)
    if nightmare_handled:
        return  # Nightmare mode handled the input
    
    game = bot_instance.get_guess_game(user.id)
    if not game or not game.get('game_active'):
        return  # Not playing, ignore message
    
    try:
        try:
            guess = int(text.strip())
        except ValueError:
            return  # Not a number, ignore
        
        elapsed = time.time() - game['start_time']
        if elapsed > game['time_limit']:
            bot_instance.end_guess_game(user.id, 'timeout')
            
            if game.get('is_daily_challenge', False):
                await update.message.reply_text(
                    "⏰ <b>Daily Challenge Time's Up!</b>\n\n"
                    f"⚠️ <b>Answer kept secret to ensure fair play!</b>\n\n"
                    f"🎯 Come back tomorrow for a new challenge!\n"
                    f"💡 Use /guess for regular games! 🎲",
                    parse_mode='HTML'
                )
            else:
                await update.message.reply_text(
                    "⏰ <b>Time's up!</b>\n\n"
                    f"🎯 The number was: <b>{game['target_number']}</b>\n\n"
                    "Better luck next time! Use /guess to play again! 🎲",
                    parse_mode='HTML'
                )
            return
        
        config = bot_instance.guess_difficulties[game['difficulty']]
        if guess < config['range'][0] or guess > config['range'][1]:
            await update.message.reply_text(
                f"❌ <b>Invalid guess!</b>\n\n"
                f"Please guess between {config['range'][0]}-{config['range'][1]}",
                parse_mode='HTML'
            )
            return
        
        game['attempts_used'] += 1
        game['guesses'].append(guess)
        
        if guess == game['target_number']:
            success = bot_instance.end_guess_game(user.id, 'won')
            time_taken = int(elapsed)
            final_score = bot_instance.calculate_guess_score(game, 'won', time_taken)
            
            if game.get('is_daily_challenge', False):
                message = (
                    "✅ <b>DAILY CHALLENGE COMPLETE!</b> 🎉\n\n"
                    f"🎮 Attempts used: <b>{game['attempts_used']}</b>\n"
                    f"⏱️ Time taken: <b>{time_taken}s</b>\n"
                    f"🏆 Final score: <b>{final_score} points</b> (+50% bonus)\n"
                    f"🌟 <b>Great job on today's challenge!</b>\n\n"
                    f"⚠️ <b>Answer kept secret to ensure fair play!</b>\n"
                    f"🎯 Come back tomorrow for a new challenge!\n\n"
                    f"💡 Use /guess for regular games!"
                )
            else:
                message = (
                    "✅ <b>PERFECT!</b> 🎉\n\n"
                    f"🎯 The number was: <b>{game['target_number']}</b>\n"
                    f"🎮 Attempts used: <b>{game['attempts_used']}</b>\n"
                    f"⏱️ Time taken: <b>{time_taken}s</b>\n"
                    f"🏆 Final score: <b>{final_score} points</b>\n\n"
                )
            
            if game.get('shard_reward'):
                message += f"💠 <b>Shards Earned:</b> +{game['shard_reward']}\n\n"
                
                if game['attempts_used'] == 1:
                    message += "⚡ <b>Perfect guess!</b> Outstanding! 🌟\n"
                elif game['attempts_used'] <= 3:
                    message += "🔥 <b>Excellent guessing!</b> Great job! 👏\n"
                else:
                    message += "👍 <b>Well done!</b> Good game! 🎯\n"
                
                if hasattr(game, 'new_level_unlocked') and game['new_level_unlocked']:
                    message += f"\n🎊 <b>NEW LEVEL UNLOCKED!</b>\n🔓 You can now play <b>{game['new_level_unlocked'].title()}</b> difficulty!\n"
                
                message += "\n🎲 Use /guess to play again!"
            
            await update.message.reply_text(message, parse_mode='HTML')
            
        elif game['attempts_used'] >= game['max_attempts']:
            bot_instance.end_guess_game(user.id, 'lost')
            
            if game.get('is_daily_challenge', False):
                message = (
                    "💔 <b>Daily Challenge Failed!</b>\n\n"
                    f"🎮 You used all {game['max_attempts']} attempts.\n"
                    f"⚠️ <b>Answer kept secret to ensure fair play!</b>\n\n"
                    f"🎯 Come back tomorrow for a new challenge!\n"
                    f"💡 Use /guess for regular games! 🎲"
                )
            else:
                message = (
                    "💔 <b>Game Over!</b>\n\n"
                    f"🎯 The number was: <b>{game['target_number']}</b>\n"
                    f"🎮 You used all {game['max_attempts']} attempts.\n\n"
                    "Better luck next time! Use /guess to play again! 🎲"
                )
            
            if game.get('shard_reward'):
                message += f"\n💠 <b>Shards Earned:</b> +{game['shard_reward']}"
            
            await update.message.reply_text(message, parse_mode='HTML')
            
        else:
            attempts_left = game['max_attempts'] - game['attempts_used']
            time_left = max(0, game['time_limit'] - int(elapsed))
            
            if guess < game['target_number']:
                hint_msg = "🔼 <b>Higher!</b>"
                game['range_min'] = max(game['range_min'], guess + 1)
            else:
                hint_msg = "🔽 <b>Lower!</b>"
                game['range_max'] = min(game['range_max'], guess - 1)
            
            total_range = config['range'][1] - config['range'][0]
            current_range = game['range_max'] - game['range_min']
            progress = max(0, 1 - (current_range / total_range))
            filled = int(progress * 10)
            bar = "▓" * filled + "░" * (10 - filled)
            
            message = (
                f"{hint_msg}\n\n"
                f"🎮 <b>Attempts left:</b> {attempts_left}\n"
                f"⏰ <b>Time left:</b> {time_left}s\n"
                f"📊 <b>Progress:</b> {bar} {int(progress * 100)}%\n\n"
                f"🎯 Keep guessing!"
            )
            
            await update.message.reply_text(message, parse_mode='HTML')
            
    except Exception as e:
        logger.error(f"Guess input error for user {user.id}: {e}")
        await update.message.reply_text(
            "❌ <b>Error!</b> Please try again.",
            parse_mode='HTML'
        )

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle callback queries from inline keyboards"""
    query = update.callback_query
    await query.answer()
    
    if not bot_instance.is_admin(query.from_user.id):
        await query.edit_message_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can handle approvals!", parse_mode='HTML')
        return
    
    data = query.data
    
    try:
        if data.startswith('approve:'):
            _, pending_id, user_id = data.split(':')
            success = bot_instance.approve_pending_achievement(int(pending_id), int(user_id), query.from_user.id)
            
            if success:
                await query.edit_message_text(
                    f"✅ <b>APPROVED!</b>\n\nAchievement approved by {query.from_user.full_name}",
                    parse_mode='HTML'
                )
            else:
                await query.edit_message_text("❌ <b>FAILED!</b>\n\nCould not approve achievement.", parse_mode='HTML')
                
        elif data.startswith('deny:'):
            _, pending_id, user_id = data.split(':')
            success = bot_instance.deny_pending_achievement(int(pending_id), query.from_user.id)
            
            if success:
                await query.edit_message_text(
                    f"❌ <b>DENIED!</b>\n\nAchievement denied by {query.from_user.full_name}",
                    parse_mode='HTML'
                )
            else:
                await query.edit_message_text("❌ <b>FAILED!</b>\n\nCould not deny achievement.", parse_mode='HTML')
                
        
    except Exception as e:
        logger.error(f"Error handling callback: {e}")
        await query.edit_message_text("❌ <b>ERROR!</b>\n\nSomething went wrong.", parse_mode='HTML')

from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import os
import asyncio
import aiohttp
import time
import requests
import signal

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"Bot is running!")

def run_health_server():
    """Run the health check server with automatic recovery"""
    while True:
        try:
            port = int(os.environ.get("PORT", 10000))
            server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
            logger.info(f"Starting health check server on port {port}")
            server.timeout = 120  # 2 minute timeout
            server.serve_forever()
        except Exception as e:
            logger.error(f"Health server error: {e}")
            logger.info("Restarting health server in 5 seconds...")
            time.sleep(5)  # Wait before retrying

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"Bot is alive!")
    
    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()

def run_health_server():
    """Run the health check server"""
    port = int(os.environ.get("PORT", 10000))
    retries = 0
    max_retries = 3
    
    while retries < max_retries:
        try:
            server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
            logger.info(f"Health check server started on port {port}")
            
            server.timeout = 30
            
            server.serve_forever()
            
        except OSError as e:
            if e.errno == 98:  # Address already in use
                logger.warning(f"Port {port} is in use, attempt {retries + 1}/{max_retries}")
                retries += 1
                time.sleep(5)
            else:
                logger.error(f"Server error: {e}")
                break
        except Exception as e:
            logger.error(f"Health server error: {e}")
            break
    
    logger.warning("Health check server stopped. Bot may shutdown after inactivity.")

async def keep_alive(application: Application):
    """Keep the bot alive by sending periodic updates"""
    base_url = os.environ.get("RENDER_EXTERNAL_URL")
    while True:
        try:
            await application.bot.get_me()
            logger.debug("Bot health check passed")
            
            if base_url:
                async with aiohttp.ClientSession() as session:
                    async with session.get(base_url) as response:
                        logger.debug(f"Self-ping status: {response.status}")
            
        except Exception as e:
            logger.error(f"Error in keep_alive: {e}")
        
        await asyncio.sleep(300)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates."""
    logger.warning(f'Update "{update}" caused error "{context.error}"')

    error_message = "❌ <b>An error occurred!</b>\n\nPlease try again or contact support if the issue persists."
    
    try:
        if update and update.effective_message:
            await update.effective_message.reply_text(error_message, parse_mode='HTML')
    except Exception as e:
        logger.error(f"Error in error handler: {e}")
    
    if isinstance(context.error, telegram_error.Conflict):
        logger.error("Bot instance conflict detected. Make sure no other instances are running.")
        return

async def button_click(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button clicks for various game actions"""
    query = update.callback_query
    await query.answer()  # Answer callback query to remove loading state
    
    data = query.data
    
    try:
        if data == "start_guess":
            await query.edit_message_text(
                text="🎯 <b>GUESS GAME STARTED!</b>\n\nUse /guess to play the number guessing game!",
                parse_mode='HTML'
            )
        
        elif data == "start_chase":
            await query.edit_message_text(
                text="🏏 <b>CHASE GAME STARTED!</b>\n\nUse /chase to play the cricket chase game!",
                parse_mode='HTML'
            )
        
        elif data == "action_guessstats":
            await guess_stats_command(update, None)
            
        elif data == "action_chasestats":
            await chase_stats_command(update, None)
            
        elif data == "daily_challenge":
            await query.edit_message_text(
                text="🎯 <b>DAILY CHALLENGE!</b>\n\nUse /guess and select the 'Daily Challenge' option to participate in today's challenge!",
                parse_mode='HTML'
            )
            
        else:
            await query.edit_message_text(
                text="❌ Unknown action. Please try again.",
                parse_mode='HTML'
            )
        
    except Exception as e:
        logger.error(f"Error in button_click: {e}")
        await query.edit_message_text(
            text=query.message.text + "\n\n❌ Error processing action",
            parse_mode='HTML'
        )

async def track_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Track all incoming messages to record chat information for broadcast"""
    try:
        chat = update.effective_chat
        if chat:
            bot_instance.track_chat(
                chat_id=chat.id,
                chat_type=chat.type,
                title=getattr(chat, 'title', None),
                username=getattr(chat, 'username', None)
            )
    except Exception as e:
        logger.error(f"Error tracking message: {e}")

async def periodic_cleanup():
    """Periodic cleanup task to prevent memory leaks"""
    while True:
        try:
            await asyncio.sleep(600)  # Run every 10 minutes
            
            chase_cleaned = cleanup_expired_games()
            if chase_cleaned > 0:
                logger.info(f"Periodic cleanup: removed {chase_cleaned} expired chase games")
            
            guess_cleaned = bot_instance.cleanup_expired_guess_games()
            if guess_cleaned > 0:
                logger.info(f"Periodic cleanup: removed {guess_cleaned} expired guess games")
                
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")

# ====================================
# ====================================

@check_banned
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Welcome message and bot introduction"""
    try:
        user = update.effective_user
        display_name = user.full_name or user.first_name or f"User{user.id}"
        
        success, is_new_user = bot_instance.create_or_update_player(
            telegram_id=user.id,
            username=user.username,
            display_name=display_name
        )
        
        if success and is_new_user:
            welcome_bonus = 1000
            bot_instance.add_shards(user.id, welcome_bonus)
        
        welcome_message = f"""
╭─────────────────────╮
│   🏆 ARENA OF CHAMPIONS 🏏   │
╰─────────────────────╯

🌟 Welcome back, {display_name}! 🎮

━━━━━━━━━━━━━━━━━━━━━━━━
🎉 CHAMPION RETURNS 🎉
━━━━━━━━━━━━━━━━━━━━━━━━

🏆 Your achievements are safe & ready
💠 Your shards are waiting to be spent
📈 New challenges await your skills

━━━━━━━━━━━━━━━━━━━━━━━━
🎮 JUMP RIGHT IN 🎮
━━━━━━━━━━━━━━━━━━━━━━━━
🏏 /chase - Cricket Action
🎯 /guess - Mind Games
💀 /nightmare - Ultimate Test

━━━━━━━━━━━━━━━━━━━━━━━━
⚡ PLAYER HUB ⚡
━━━━━━━━━━━━━━━━━━━━━━━━
👤 /profile - Your stats
🏆 /achievements - Your glory
💠 /shardlb - Rich list
📊 /leaderboard - Top players

🔥 Time to reclaim your throne! 🔥

━━━━━━━━━━━━━━━━━━━━━━━━
❤️ Crafted with passion for cricket lovers ❤️
━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        await update.message.reply_text(welcome_message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Error in start_command: {e}")
        await update.message.reply_text(
            "❌ <b>Welcome to Arena Of Champions!</b>\n\nUse /help to see available commands.",
            parse_mode='HTML'
        )

# ====================================
# ====================================

@check_banned
async def register_auction_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start auction registration process (ANY USER CAN START)"""
    try:
        user = update.effective_user
        user_id = user.id
        user_name = user.full_name or user.first_name or "Unknown"
        
        if user_id in bot_instance.registration_states:
            await update.message.reply_text(
                "⚠️ You already have an active auction registration.\n"
                "Please complete it first or contact an admin to reset it."
            )
            return
        
        proposal_id = bot_instance.create_auction_proposal(user_id, user_name)
        
        bot_instance.registration_states[user_id] = AuctionRegistrationState()
        
        await update.message.reply_text(
            f"🏆 <b>Arena Of Champions Auction Registration</b>\n\n"
            f"👤 <b>Creator:</b> {user_name}\n\n"
            f"📝 <b>Step 1/4: Auction Name</b>\n"
            f"Please enter the name of your auction:\n\n"
            f"<i>Example: IPL 2024 Mini Tournament</i>",
            parse_mode='HTML'
        )
        
        bot_instance.registration_states[user_id].data['proposal_id'] = proposal_id
        
    except Exception as e:
        logger.error(f"Error in register_auction_command: {e}")
        await update.message.reply_text("❌ An error occurred. Please try again.")

async def handle_auction_registration(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle auction registration steps"""
    try:
        user = update.effective_user
        user_id = user.id
        text = update.message.text
        
        if user_id not in bot_instance.registration_states:
            return  # No active registration
        
        state = bot_instance.registration_states[user_id]
        proposal_id = state.data.get('proposal_id')
        
        if not proposal_id or proposal_id not in bot_instance.auction_proposals:
            return
        
        if state.step == "name":
            state.data['name'] = text.strip()
            state.step = "teams"
            
            await update.message.reply_text(
                f"✅ <b>Auction Name:</b> {text.strip()}\n\n"
                f"📝 <b>Step 2/4: Team Names</b>\n"
                f"Please enter the team names separated by commas:\n\n"
                f"<i>Example: Mumbai Indians, Chennai Super Kings, Delhi Capitals</i>",
                parse_mode='HTML'
            )
            
        elif state.step == "teams":
            teams = [team.strip() for team in text.split(',') if team.strip()]
            if len(teams) < 2:
                await update.message.reply_text(
                    "❌ Please provide at least 2 team names separated by commas.\n"
                    f"<i>Example: Team A, Team B, Team C</i>",
                    parse_mode='HTML'
                )
                return
            
            state.data['teams'] = teams
            state.step = "purse_and_base"
            
            await update.message.reply_text(
                f"✅ <b>Teams ({len(teams)}):</b> {', '.join(teams[:3])}"
                f"{'...' if len(teams) > 3 else ''}\n\n"
                f"📝 <b>Step 3/3: Purse & Base Price (in Crores)</b>\n"
                f"Enter team purse and base price in crores:\n\n"
                f"💡 <b>Format:</b> purse,base_price\n"
                f"💡 <b>Example:</b> 100,1\n"
                f"💡 <b>Or:</b> 100 1\n\n"
                f"<i>(100cr purse with 1cr base price)</i>",
                parse_mode='HTML'
            )
            
        elif state.step == "purse_and_base":
            try:
                if ',' in text:
                    parts = text.split(',')
                elif ' ' in text:
                    parts = text.split()
                else:
                    raise ValueError("Invalid format")
                
                if len(parts) != 2:
                    await update.message.reply_text(
                        "❌ <b>Invalid Format!</b>\n\n"
                        "Please enter both purse and base price (in crores):\n\n"
                        "💡 <b>Format:</b> purse,base_price\n"
                        "💡 <b>Example:</b> 100,1\n"
                        "💡 <b>Or:</b> 100 1\n\n"
                        "<i>(100cr purse with 1cr base price)</i>",
                        parse_mode='HTML'
                    )
                    return
                
                purse = float(parts[0].strip().replace(',', ''))
                base_price = float(parts[1].strip().replace(',', ''))
                
                if purse <= 0:
                    raise ValueError("Purse must be positive")
                if base_price <= 0:
                    raise ValueError("Base price must be positive")
                if base_price > purse:
                    await update.message.reply_text(
                        "❌ Base price cannot be higher than team purse!\n\n"
                        f"💰 <b>Your Purse:</b> {format_amount(purse)}\n"
                        f"💎 <b>Your Base:</b> {format_amount(base_price)}",
                        parse_mode='HTML'
                    )
                    return
                
                state.data['purse'] = purse
                state.data['base_price'] = base_price
                state.step = "complete"
                
                bot_instance.update_proposal_data(proposal_id, state.data)
                
                await send_proposal_to_admins(update, proposal_id, context)
            
                await update.message.reply_text(
                    f"🎉 <b>Auction Registration Complete!</b>\n\n"
                    f"📋 <b>Summary:</b>\n"
                    f"🏆 <b>Name:</b> {state.data['name']}\n"
                    f"👥 <b>Teams:</b> {len(state.data['teams'])}\n"
                    f"💰 <b>Team Purse:</b> {format_amount(purse)}\n"
                    f"💎 <b>Base Price:</b> {format_amount(base_price)}\n\n"
                    f"📤 <b>Your proposal has been sent to admins for approval.</b>\n"
                    f"You will be notified once it's reviewed!",
                    parse_mode='HTML'
                )
                
                del bot_instance.registration_states[user_id]
                
            except ValueError:
                await update.message.reply_text(
                    "❌ Please enter valid numbers for purse and base price.\n"
                    f"💡 <b>Example:</b> 1000000,5000",
                    parse_mode='HTML'
                )
                return
                
    except Exception as e:
        logger.error(f"Error in handle_auction_registration: {e}")

async def send_proposal_to_admins(update: Update, proposal_id: int, context: ContextTypes.DEFAULT_TYPE = None) -> None:
    """Send auction proposal to all admins for approval"""
    try:
        if proposal_id not in bot_instance.auction_proposals:
            return
        
        proposal = bot_instance.auction_proposals[proposal_id]
        
        keyboard = [
            [
                InlineKeyboardButton("✅ Approve", callback_data=f"approve_auction_{proposal_id}"),
                InlineKeyboardButton("❌ Reject", callback_data=f"reject_auction_{proposal_id}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = (
            f"🏆 <b>New Auction Proposal</b>\n\n"
            f"👤 <b>Creator:</b> {proposal.creator_name}\n"
            f"🆔 <b>Creator ID:</b> {proposal.creator_id}\n\n"
            f"📋 <b>Auction Details:</b>\n"
            f"🏆 <b>Name:</b> {proposal.name}\n"
            f"👥 <b>Teams ({len(proposal.teams)}):</b>\n"
        )
        
        for i, team in enumerate(proposal.teams, 1):
            message += f"    {i}. {team}\n"
        
        message += (
            f"\n💰 <b>Purse per Team:</b> {proposal.purse}Cr\n"
            f"💎 <b>Base Price:</b> {proposal.base_price}Cr\n\n"
            f"⏰ <b>Submitted:</b> {proposal.created_at.strftime('%Y-%m-%d %H:%M')}"
        )
        
        bot = context.bot if context else update.get_bot()
        
        admin_ids = bot_instance.get_all_admin_ids()
        logger.info(f"Sending auction proposal to {len(admin_ids)} admins: {admin_ids}")
        
        if not admin_ids or proposal.creator_id in admin_ids:
            if proposal.creator_id not in admin_ids:
                admin_ids.append(proposal.creator_id)
                logger.info(f"Added creator {proposal.creator_id} to admin notification list")
        
        notification_sent = False
        for admin_id in admin_ids:
            try:
                await bot.send_message(
                    chat_id=admin_id,
                    text=message,
                    parse_mode='HTML',
                    reply_markup=reply_markup
                )
                notification_sent = True
                logger.info(f"Auction proposal sent to admin {admin_id}")
            except Exception as e:
                logger.error(f"Failed to send proposal to admin {admin_id}: {e}")
        
        if not notification_sent:
            logger.warning("Failed to send proposal to any admin - sending fallback message to creator")
            try:
                fallback_message = (
                    f"⚠️ <b>Admin Notification Issue</b>\n\n"
                    f"Your auction proposal was created but couldn't be sent to admins.\n"
                    f"Since you appear to be an admin, you can approve it yourself using:\n\n"
                    f"<code>/hostpanel {proposal.id}</code>\n\n"
                    f"Or check the admin panel for pending approvals."
                )
                await bot.send_message(
                    chat_id=proposal.creator_id,
                    text=fallback_message,
                    parse_mode='HTML'
                )
            except Exception as e:
                logger.error(f"Failed to send fallback message: {e}")
                
    except Exception as e:
        logger.error(f"Error sending proposal to admins: {e}")
#====================================
# ====================================

async def handle_admin_auction_approval(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle admin approval/rejection of auction proposals"""
    try:
        query = update.callback_query
        user = query.from_user
        
        if not bot_instance.is_admin(user.id):
            await query.edit_message_text("❌ Admin access required!")
            return
        
        data = query.data
        if data.startswith("approve_auction_"):
            proposal_id = int(data.split("_")[-1])
            auction_id = bot_instance.approve_auction_proposal(proposal_id, user.id, user.full_name or user.first_name)
            
            if auction_id:
                proposal = bot_instance.auction_proposals[proposal_id]
                
                try:
                    await context.bot.send_message(
                        chat_id=proposal.creator_id,
                        text=f"🎉 <b>Your auction is approved!</b>\n\n"
                             f"🏆 <b>Auction:</b> {proposal.name}\n"
                             f"🆔 <b>Auction ID:</b> {auction_id}\n\n"
                             f"🎮 Use <code>/hostpanel {auction_id}</code> to control everything!",
                        parse_mode='HTML'
                    )
                except:
                    pass
                
                await query.edit_message_text(
                    text=f"✅ <b>APPROVED</b> by {user.first_name}\n\n"
                         f"🏆 <b>Auction:</b> {proposal.name}\n"
                         f"👤 <b>Creator:</b> {proposal.creator_name}\n"
                         f"🆔 <b>Auction ID:</b> {auction_id}\n"
                         f"⏰ <b>Approved:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    parse_mode='HTML'
                )
                
        elif data.startswith("reject_auction_"):
            proposal_id = int(data.split("_")[-1])
            bot_instance.reject_auction_proposal(proposal_id, user.id, user.full_name or user.first_name)
            
            if proposal_id in bot_instance.auction_proposals:
                proposal = bot_instance.auction_proposals[proposal_id]
                
                try:
                    await context.bot.send_message(
                        chat_id=proposal.creator_id,
                        text=f"❌ <b>Your auction proposal was rejected</b>\n\n"
                             f"🏆 <b>Auction:</b> {proposal.name}\n\n"
                             f"You can create a new proposal with <code>/register</code>",
                        parse_mode='HTML'
                    )
                except:
                    pass
                
                await query.edit_message_text(
                    text=f"❌ <b>REJECTED</b> by {user.first_name}\n\n"
                         f"🏆 <b>Auction:</b> {proposal.name}\n"
                         f"👤 <b>Creator:</b> {proposal.creator_name}\n"
                         f"⏰ <b>Rejected:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    parse_mode='HTML'
                )
        
    except telegram_error.BadRequest as e:
        if "query is too old" in str(e).lower() or "timeout" in str(e).lower():
            logger.warning(f"Admin approval callback timeout: {e}")
            return
        else:
            logger.error(f"BadRequest in admin auction approval: {e}")
    except Exception as e:
        logger.error(f"Error in admin auction approval: {e}")
        try:
            await query.edit_message_text("❌ An error occurred! Please try again.")
        except:
            pass

# ====================================
# ====================================

@check_banned
async def hostpanel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Host panel for auction control"""
    try:
        user = update.effective_user
        
        if len(context.args) < 1:
            await update.message.reply_text(
                "❌ <b>Usage:</b> <code>/hostpanel [auction_id]</code>\n\n"
                "<b>Example:</b> <code>/hostpanel 5</code>",
                parse_mode='HTML'
            )
            return
        
        try:
            auction_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("❌ Please provide a valid auction ID number.")
            return
        
        auction = bot_instance.get_approved_auction(auction_id)
        if not auction:
            await update.message.reply_text("❌ Auction not found!")
            return
        
        if auction.creator_id != user.id and not bot_instance.is_admin(user.id):
            await update.message.reply_text("❌ Only the auction creator can access the host panel!")
            return
        
        keyboard = []
        
        if auction.status == "setup":
            keyboard.extend([
                [InlineKeyboardButton("🚀 Start Captain Registration", callback_data=f"host_start_captain_{auction_id}")],
                [InlineKeyboardButton("💬 Set Group Chat", callback_data=f"host_set_gc_{auction_id}")],
                [InlineKeyboardButton("📊 Auction Info", callback_data=f"host_info_{auction_id}")]
            ])
        elif auction.status == "captain_reg":
            keyboard.extend([
                [InlineKeyboardButton("✅ Approve Captains", callback_data=f"host_approve_captains_{auction_id}")],
                [InlineKeyboardButton("👥 Start Player Registration", callback_data=f"host_start_player_{auction_id}")],
                [InlineKeyboardButton("📊 Auction Info", callback_data=f"host_info_{auction_id}")]
            ])
        elif auction.status == "player_reg":
            keyboard.extend([
                [InlineKeyboardButton("✅ Approve Captains", callback_data=f"host_approve_captains_{auction_id}")],
                [InlineKeyboardButton("✅ Approve Players", callback_data=f"host_approve_players_{auction_id}")],
                [InlineKeyboardButton("🔒 Close Registration", callback_data=f"host_close_reg_{auction_id}")],
                [InlineKeyboardButton("📊 Auction Info", callback_data=f"host_info_{auction_id}")]
            ])
        elif auction.status == "ready":
            keyboard.extend([
                [InlineKeyboardButton("🔥 START AUCTION", callback_data=f"host_start_auction_{auction_id}")],
                [InlineKeyboardButton("📊 Auction Info", callback_data=f"host_info_{auction_id}")]
            ])
        elif auction.status == "active":
            pause_text = "▶️ Resume Auction" if auction.is_paused else "⏸️ Pause Auction"
            keyboard.extend([
                [InlineKeyboardButton(pause_text, callback_data=f"host_pause_{auction_id}")],
                [InlineKeyboardButton("⏭️ Skip Player (Unsold)", callback_data=f"host_skip_{auction_id}")],
                [InlineKeyboardButton("🔄 Rebid Current", callback_data=f"host_rebid_current_{auction_id}")],
                [InlineKeyboardButton("👨‍⚖️ Manual Assign", callback_data=f"host_assign_{auction_id}")],
                [InlineKeyboardButton("⏹️ End Auction", callback_data=f"host_end_{auction_id}")],
                [InlineKeyboardButton("📊 Auction Info", callback_data=f"host_info_{auction_id}")]
            ])
        elif auction.status == "completed":
            keyboard.extend([
                [InlineKeyboardButton("📊 Final Results", callback_data=f"host_results_{auction_id}")]
            ])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        status_emoji = {
            'setup': '🔧',
            'captain_reg': '👑',
            'player_reg': '👥',
            'ready': '⚡',
            'active': '🔥',
            'completed': '✅'
        }.get(auction.status, '❓')
        
        message = (
            f"🎮 <b>Host Panel - {auction.name}</b>\n\n"
            f"🆔 <b>ID:</b> {auction_id}\n"
            f"{status_emoji} <b>Status:</b> {auction.status.title()}\n"
            f"👑 <b>Captains:</b> {len(auction.approved_captains)}\n"
            f"👥 <b>Players:</b> {len(auction.approved_players)}\n\n"
            f"🎯 <b>Choose an action:</b>"
        )
        
        await update.message.reply_text(message, parse_mode='HTML', reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"Error in hostpanel_command: {e}")
        await update.message.reply_text("❌ An error occurred!")

# ====================================
# ====================================

@check_banned
async def registercaptain_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Register as captain for an auction"""
    try:
        user = update.effective_user
        
        if len(context.args) < 2:
            await update.message.reply_text(
                "❌ <b>Usage:</b> <code>/regcap [auction_id] [team_name]</code>\n\n"
                "<b>Example:</b> <code>/regcap 5 Mumbai Indians</code>",
                parse_mode='HTML'
            )
            return
        
        try:
            auction_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("❌ Please provide a valid auction ID.")
            return
        
        team_name = ' '.join(context.args[1:])
        captain_name = user.full_name or user.first_name
        
        auction = bot_instance.get_approved_auction(auction_id)
        if not auction:
            await update.message.reply_text("❌ Auction not found!")
            return
        
        if auction.status not in ["captain_reg", "player_reg", "ready"]:
            await update.message.reply_text("❌ Captain registration is not open for this auction!")
            return
        
        if team_name not in auction.teams:
            team_list = ', '.join(auction.teams)
            await update.message.reply_text(
                f"❌ Invalid team name!\n\n"
                f"🏏 <b>Available teams:</b>\n{team_list}",
                parse_mode='HTML'
            )
            return
        
        for captain in auction.approved_captains.values():
            if captain.team_name == team_name:
                await update.message.reply_text(f"❌ Team '{team_name}' is already taken!")
                return
        
        success = bot_instance.register_captain_request(
            auction_id, user.id, captain_name, team_name
        )
        
        if success:
            keyboard = [
                [
                    InlineKeyboardButton("✅ Approve", callback_data=f"approve_captain_{auction_id}_{user.id}"),
                    InlineKeyboardButton("❌ Reject", callback_data=f"reject_captain_{auction_id}_{user.id}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            try:
                await context.bot.send_message(
                    chat_id=auction.creator_id,
                    text=f"👑 <b>New Captain Registration!</b>\n\n"
                         f"🏆 <b>Auction:</b> {auction.name}\n"
                         f"👤 <b>Captain:</b> {user.full_name or user.first_name}\n"
                         f"🆔 <b>User ID:</b> {user.id}\n"
                         f"🏏 <b>Team:</b> {team_name}\n"
                         f"📝 <b>Captain Name:</b> {captain_name}\n\n"
                         f"⏰ <b>Registered:</b> {datetime.now().strftime('%H:%M:%S')}",
                    reply_markup=reply_markup,
                    parse_mode='HTML'
                )
            except Exception as e:
                logger.error(f"Failed to send captain approval to host: {e}")
            
            await update.message.reply_text(
                f"✅ <b>Captain Registration Submitted!</b>\n\n"
                f"🏏 <b>Team:</b> {team_name}\n"
                f"🏆 <b>Auction:</b> {auction.name}\n\n"
                f"⏳ Waiting for host approval...",
                parse_mode='HTML'
            )
        else:
            await update.message.reply_text("❌ Registration failed! You may have already registered.")
            
    except Exception as e:
        logger.error(f"Error in registercaptain_command: {e}")
        await update.message.reply_text("❌ An error occurred!")

# ====================================
# ====================================

@check_banned
async def registerplayer_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Register as player for an auction (SIMPLE - ONE COMMAND ONLY)"""
    try:
        user = update.effective_user
        
        if len(context.args) < 1:
            await update.message.reply_text(
                "❌ <b>Usage:</b> <code>/regplay [auction_id]</code>\n\n"
                "<b>Example:</b> <code>/regplay 5</code>",
                parse_mode='HTML'
            )
            return
        
        try:
            auction_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("❌ Please provide a valid auction ID.")
            return
        
        auction = bot_instance.get_approved_auction(auction_id)
        if not auction:
            await update.message.reply_text("❌ Auction not found!")
            return
        
        if auction.status not in ["captain_reg", "player_reg", "ready"]:
            await update.message.reply_text("❌ Player registration is not open for this auction!")
            return
        
        if user.id in auction.registered_captains or user.id in auction.approved_captains:
            await update.message.reply_text("❌ You are already registered as a captain! Cannot be both captain and player.")
            return
        
        if user.id in auction.registered_players:
            await update.message.reply_text("❌ You are already registered as a player!")
            return
        
        success = bot_instance.register_player_request(
            auction_id, user.id, user.full_name or user.first_name, user.username
        )
        
        if success:
            keyboard = [
                [
                    InlineKeyboardButton("✅ Approve", callback_data=f"approve_player_{auction_id}_{user.id}"),
                    InlineKeyboardButton("❌ Reject", callback_data=f"reject_player_{auction_id}_{user.id}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            try:
                await context.bot.send_message(
                    chat_id=auction.creator_id,
                    text=f"👥 <b>New Player Registered!</b>\n\n"
                         f"🏆 <b>Auction:</b> {auction.name}\n"
                         f"👤 <b>Player:</b> {user.full_name or user.first_name} (@{user.username or 'no_username'})\n\n"
                         f"<b>Approve or Reject?</b>",
                    parse_mode='HTML',
                    reply_markup=reply_markup
                )
            except Exception as e:
                logger.error(f"Failed to notify host: {e}")
            
            await update.message.reply_text(
                f"✅ <b>Player Registration Submitted!</b>\n\n"
                f"🏆 <b>Auction:</b> {auction.name}\n"
                f"👤 <b>Player:</b> {user.full_name or user.first_name}\n\n"
                f"⏳ Waiting for host approval...",
                parse_mode='HTML'
            )
        else:
            await update.message.reply_text("❌ Registration failed! You may have already registered.")
            
    except Exception as e:
        logger.error(f"Error in registerplayer_command: {e}")
        await update.message.reply_text("❌ An error occurred!")

# ====================================
# ====================================

async def handle_host_panel_callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle all host panel button callbacks"""
    try:
        query = update.callback_query
        user = query.from_user
        data = query.data
        
        auction_id = int(data.split('_')[-1])
        auction = bot_instance.get_approved_auction(auction_id)
        
        is_host = (auction and (auction.creator_id == user.id or bot_instance.is_admin(user.id)))
        if not is_host:
            await query.edit_message_text("❌ Only auction host/admin can use host panel!")
            return
        
        if auction.group_chat_id is None and update.effective_chat.type in ['group', 'supergroup']:
            auction.group_chat_id = update.effective_chat.id
            logger.info(f"Set group chat ID {auction.group_chat_id} for auction {auction_id}")
        
        if data.startswith("host_start_captain_"):
            success = bot_instance.start_captain_registration(auction_id)
            if success:
                await query.edit_message_text(
                    f"✅ <b>Captain Registration Started!</b>\n\n"
                    f"🏆 <b>Auction:</b> {auction.name}\n"
                    f"👑 <b>Status:</b> Captain Registration Open\n\n"
                    f"📢 <b>Captains can now register with:</b>\n"
                    f"<code>/regcap {auction_id} [team_name]</code>\n\n"
                    f"🏏 <b>Available Teams:</b>\n" + 
                    '\n'.join(f"• {team}" for team in auction.teams),
                    parse_mode='HTML'
                )
        
        elif data.startswith("host_start_player_"):
            success = bot_instance.start_player_registration(auction_id)
            if success:
                await query.edit_message_text(
                    f"✅ <b>Player Registration Started!</b>\n\n"
                    f"🏆 <b>Auction:</b> {auction.name}\n"
                    f"👥 <b>Status:</b> Player Registration Open\n\n"
                    f"📢 <b>Players can now register with:</b>\n"
                    f"<code>/regplay {auction_id}</code>\n\n"
                    f"⚡ <b>Simple Process:</b> Players just send one command!",
                    parse_mode='HTML'
                )
        
        elif data.startswith("host_approve_captains_"):
            pending_captains = [cap for cap in auction.registered_captains.values() if cap.status == "pending"]
            
            approved_count = 0
            for captain in pending_captains:
                success = bot_instance.approve_captain(auction_id, captain.user_id)
                if success:
                    approved_count += 1
            
            if approved_count > 0:
                await query.edit_message_text(
                    f"✅ <b>{approved_count} Captains Approved!</b>\n\n"
                    f"🏆 <b>Auction:</b> {auction.name}\n"
                    f"👑 <b>Total Approved Captains:</b> {len(auction.approved_captains)}\n\n"
                    f"🚀 <b>Ready to start player registration!</b>",
                    parse_mode='HTML'
                )
                
                if auction.group_chat_id:
                    try:
                        captain_names = [cap.name for cap in auction.approved_captains.values()][-approved_count:]
                        captain_list = "\n".join([f"👑 {name}" for name in captain_names])
                        await context.bot.send_message(
                            chat_id=auction.group_chat_id,
                            text=f"🎉 <b>Captains Approved!</b>\n\n"
                                 f"🏆 <b>Auction:</b> {auction.name}\n"
                                 f"✅ <b>Approved:</b> {approved_count} captains\n\n"
                                 f"{captain_list}\n\n"
                                 f"👥 <b>Total Captains:</b> {len(auction.approved_captains)}\n"
                                 f"🚀 <b>Player registration is now open!</b>",
                            parse_mode='HTML'
                        )
                    except Exception as e:
                        logger.error(f"Failed to send group notification: {e}")
            else:
                await query.edit_message_text("❌ No pending captains to approve!")
        
        elif data.startswith("host_approve_players_"):
            pending_players = [player for player in auction.registered_players.values() if player.status == "pending"]
            
            approved_count = 0
            for player in pending_players:
                success = bot_instance.approve_player(auction_id, player.user_id)
                if success:
                    approved_count += 1
            
            if approved_count > 0:
                await query.edit_message_text(
                    f"✅ <b>{approved_count} Players Approved!</b>\n\n"
                    f"🏆 <b>Auction:</b> {auction.name}\n"
                    f"👥 <b>Total Approved Players:</b> {len(auction.approved_players)}\n\n"
                    f"🚀 <b>Ready to close registration and start auction!</b>",
                    parse_mode='HTML'
                )
            else:
                await query.edit_message_text("❌ No pending players to approve!")
        
        elif data.startswith("host_close_reg_"):
            keyboard = [
                [InlineKeyboardButton("🎲 Random Order", callback_data=f"host_random_yes_{auction_id}")],
                [InlineKeyboardButton("📋 Sequential Order", callback_data=f"host_random_no_{auction_id}")],
                [InlineKeyboardButton("🔙 Cancel", callback_data=f"host_panel_{auction_id}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                f"🎲 <b>PLAYER ORDER</b>\n\n"
                f"How should players appear during auction?\n\n"
                f"🎲 <b>Random Order:</b>\n"
                f"   • Players shuffled randomly\n"
                f"   • Unpredictable sequence\n"
                f"   • More excitement\n\n"
                f"📋 <b>Sequential Order:</b>\n"
                f"   • Players in registration order\n"
                f"   • Predictable sequence\n"
                f"   • Organized approach\n\n"
                f"❓ <b>Choose your preference:</b>",
                parse_mode='HTML',
                reply_markup=reply_markup
            )
        
        elif data.startswith("host_random_yes_") or data.startswith("host_random_no_"):
            randomize = data.startswith("host_random_yes_")
            auction.randomize_players = randomize
            
            success = bot_instance.close_registration(auction_id)
            if success:
                order_text = "🎲 Random" if randomize else "📋 Sequential"
                await query.edit_message_text(
                    f"🔒 <b>Registration Closed!</b>\n\n"
                    f"🏆 <b>Auction:</b> {auction.name}\n"
                    f"🎯 <b>Player Order:</b> {order_text}\n"
                    f"📊 <b>Final Stats:</b>\n"
                    f"👑 Captains: {len(auction.approved_captains)}\n"
                    f"👥 Players: {len(auction.approved_players)}\n\n"
                    f"🚀 <b>Ready to start auction!</b>\n"
                    f"Use host panel to begin bidding.",
                    parse_mode='HTML'
                )
        
        elif data.startswith("host_panel_"):
            await hostpanel_command(update, context)
        
        elif data.startswith("host_start_auction_"):
            success = bot_instance.start_auction_bidding(auction_id)
            if success:
                current_player = auction.current_player
                username_display = f"@{current_player.username}" if hasattr(current_player, 'username') and current_player.username else ""
                await query.edit_message_text(
                    f"🔥 <b>AUCTION STARTED!</b>\n\n"
                    f"👤 <b>{current_player.name}</b> {username_display}\n"
                    f"💎 <b>Base:</b> {format_amount(current_player.base_price)}\n\n"
                    f"🎯 <b>Type amount to bid!</b>\n"
                    f"📝 Admin: Use /sell {auction_id} or reply '..' to sell",
                    parse_mode='HTML'
                )

        elif data.startswith("host_pause_"):
            try:
                result = bot_instance.pause_auction(auction_id)
                if result:
                    status = "⏸️ PAUSED" if auction.is_paused else "▶️ RESUMED"
                    action = "paused" if auction.is_paused else "resumed"
                    await query.answer(f"✅ Auction {action}!", show_alert=True)
                    await hostpanel_command(update, context)
                else:
                    await query.answer("❌ Failed to toggle pause!", show_alert=True)
            except Exception as e:
                logger.error(f"Error in host_pause callback: {e}")
                await query.answer("✅ Pause toggled!", show_alert=True)
        
        elif data.startswith("host_rebid_current_"):
            if not auction.current_player:
                await query.answer("❌ No active player to rebid!", show_alert=True)
                return
            
            player_username = getattr(auction.current_player, 'username', '') or auction.current_player.name
            success, message_text, player = bot_instance.rebid_player(auction_id, player_username)
            if success:
                current_player = auction.current_player
                username_display = f"@{current_player.username}" if hasattr(current_player, 'username') and current_player.username else ""
                
                await query.edit_message_text(
                    f"🔄 <b>REBID STARTED!</b>\n\n"
                    f"👤 <b>{current_player.name}</b> {username_display}\n"
                    f"💎 <b>Base:</b> {format_amount(current_player.base_price)}\n\n"
                    f"🎯 <b>Fresh start - Type amount to bid!</b>\n"
                    f"📝 Admin: Use /sell {auction_id} or reply '..' to sell",
                    parse_mode='HTML'
                )
            else:
                await query.answer("❌ Failed to rebid player!", show_alert=True)
        
        elif data.startswith("host_rebid_"):
            if auction.current_player:
                auction.highest_bidder = None
                auction.highest_bid = auction.base_price
                if hasattr(auction, 'current_bids'):
                    auction.current_bids = {}
                
                current_player = auction.current_player
                username_display = f"@{current_player.username}" if hasattr(current_player, 'username') and current_player.username else ""
                
                await query.edit_message_text(
                    f"🔄 <b>REBID STARTED!</b>\n\n"
                    f"👤 <b>{current_player.name}</b> {username_display}\n"
                    f"💎 <b>Base:</b> {format_amount(current_player.base_price)}\n\n"
                    f"🎯 <b>Fresh start - Type amount to bid!</b>\n"
                    f"📝 Admin: Use /sell {auction_id} or reply '..' to sell",
                    parse_mode='HTML'
                )
                
                if auction.group_chat_id:
                    try:
                        await context.bot.send_message(
                            chat_id=auction.group_chat_id,
                            text=f"🔄 <b>REBID - {current_player.name}</b>\n\n"
                                 f"💎 <b>Base:</b> {format_amount(current_player.base_price)}\n"
                                 f"🎯 <b>Fresh bidding starts now!</b>",
                            parse_mode='HTML'
                        )
                    except Exception as e:
                        logger.error(f"Failed to send rebid notification to group: {e}")
            else:
                await query.edit_message_text("❌ No current player to rebid!")
        
        elif data.startswith("host_skip_"):
            success = bot_instance.skip_current_player(auction_id)
            if success:
                next_player = auction.current_player
                if next_player:
                    await query.edit_message_text(
                        f"⏭️ <b>Player Skipped!</b>\n\n"
                        f"👤 <b>Next Player:</b> {next_player.name}\n"
                        f"💎 <b>Base Price:</b> {format_amount(next_player.base_price)}\n\n"
                        f"🎯 Ready for bidding!",
                        parse_mode='HTML'
                    )
                else:
                    await query.edit_message_text("🏁 <b>Auction Complete!</b> All players processed.")
        
        elif data.startswith("host_end_"):
            success = bot_instance.end_auction(auction_id)
            if success:
                await query.edit_message_text(
                    f"🏁 <b>Auction Ended!</b>\n\n"
                    f"🏆 <b>Auction:</b> {auction.name}\n"
                    f"📊 <b>Final Results:</b>\n"
                    f"👑 Teams: {len(auction.approved_captains)}\n"
                    f"👥 Players Sold: {len(auction.sold_players)}\n\n"
                    f"✅ <b>Auction Complete!</b>",
                    parse_mode='HTML'
                )
        
        elif data.startswith("host_set_gc_"):
            await query.edit_message_text(
                f"🔧 <b>Group Chat Setup</b>\n\n"
                f"📝 <b>Please provide the Group Chat ID where auction messages will be sent.</b>\n\n"
                f"🔍 <b>How to get Group Chat ID:</b>\n"
                f"1. Add @RawDataBot to your group\n"
                f"2. Send /start in the group\n"
                f"3. Copy the chat ID number\n\n"
                f"💬 <b>Send:</b> <code>/setgc {auction_id} [chat_id]</code>\n\n"
                f"📋 <b>Example:</b> <code>/setgc {auction_id} -1001234567890</code>",
                parse_mode='HTML'
            )
        
        elif data.startswith("host_info_"):
            message = get_auction_info_message(auction)
            await query.edit_message_text(message, parse_mode='HTML')
        
    except telegram_error.BadRequest as e:
        if "query is too old" in str(e).lower() or "timeout" in str(e).lower():
            logger.warning(f"Host callback timeout: {e}")
            return
        else:
            logger.error(f"BadRequest in host panel callbacks: {e}")
    except Exception as e:
        logger.error(f"Error in host panel callbacks: {e}")
        try:
            await query.edit_message_text("❌ An error occurred! Please try again.")
        except:
            pass

async def handle_captain_approval_callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle individual captain approve/reject callbacks"""
    try:
        query = update.callback_query
        user = query.from_user
        data = query.data
        
        parts = data.split('_')
        auction_id = int(parts[2])
        captain_id = int(parts[3])
        
        auction = bot_instance.get_approved_auction(auction_id)
        if not auction or (auction.creator_id != user.id and not bot_instance.is_admin(user.id)):
            await query.edit_message_text("❌ Access denied!")
            return
        
        if data.startswith("approve_captain_"):
            success = bot_instance.approve_captain(auction_id, captain_id)
            if success:
                captain = auction.approved_captains[captain_id]
                
                try:
                    await context.bot.send_message(
                        chat_id=captain_id,
                        text=f"🎉 <b>You're approved as captain!</b>\n\n"
                             f"🏆 <b>Auction:</b> {auction.name}\n"
                             f"🏏 <b>Team:</b> {captain.team_name}\n"
                             f"💰 <b>Budget:</b> {captain.purse}Cr\n\n"
                             f"👑 <b>You're now ready for the auction!</b>",
                        parse_mode='HTML'
                    )
                except:
                    pass
                
                await query.edit_message_text(
                    f"✅ <b>CAPTAIN APPROVED</b> by {user.first_name}\n\n"
                    f"👑 <b>Captain:</b> {captain.name}\n"
                    f"🏏 <b>Team:</b> {captain.team_name}\n"
                    f"🏆 <b>Auction:</b> {auction.name}\n"
                    f"⏰ <b>Approved:</b> {datetime.now().strftime('%H:%M:%S')}",
                    parse_mode='HTML'
                )
                
                if auction.group_chat_id:
                    try:
                        await context.bot.send_message(
                            chat_id=auction.group_chat_id,
                            text=f"👑 <b>Captain Approved!</b>\n\n"
                                 f"👤 <b>Captain:</b> {captain.name}\n"
                                 f"🏏 <b>Team:</b> {captain.team_name}\n"
                                 f"🏆 <b>Auction:</b> {auction.name}\n"
                                 f"👑 <b>Approved by:</b> {user.first_name}\n\n"
                                 f"🎉 <b>{captain.name} is ready to lead {captain.team_name}!</b>",
                            parse_mode='HTML'
                        )
                    except Exception as e:
                        logger.error(f"Failed to send group notification: {e}")
        
        elif data.startswith("reject_captain_"):
            success = bot_instance.reject_captain(auction_id, captain_id)
            if success:
                registration = auction.registered_captains.get(captain_id)
                captain_name = registration.name if registration else "Unknown"
                team_name = registration.team_name if registration else "Unknown"
                
                try:
                    await context.bot.send_message(
                        chat_id=captain_id,
                        text=f"❌ <b>Your captain registration was rejected</b>\n\n"
                             f"🏆 <b>Auction:</b> {auction.name}\n"
                             f"🏏 <b>Team:</b> {team_name}\n\n"
                             f"You can try registering for other teams or auctions.",
                        parse_mode='HTML'
                    )
                except:
                    pass
                
                await query.edit_message_text(
                    f"❌ <b>CAPTAIN REJECTED</b> by {user.first_name}\n\n"
                    f"👤 <b>Captain:</b> {captain_name}\n"
                    f"🏏 <b>Team:</b> {team_name}\n"
                    f"🏆 <b>Auction:</b> {auction.name}\n"
                    f"⏰ <b>Rejected:</b> {datetime.now().strftime('%H:%M:%S')}",
                    parse_mode='HTML'
                )
                
                if auction.group_chat_id:
                    try:
                        await context.bot.send_message(
                            chat_id=auction.group_chat_id,
                            text=f"❌ <b>Captain Rejected</b>\n\n"
                                 f"👤 <b>Captain:</b> {captain_name}\n"
                                 f"🏏 <b>Team:</b> {team_name}\n"
                                 f"🏆 <b>Auction:</b> {auction.name}\n"
                                 f"👑 <b>Rejected by:</b> {user.first_name}",
                            parse_mode='HTML'
                        )
                    except Exception as e:
                        logger.error(f"Failed to send group notification: {e}")
        
        await query.answer()
        
    except Exception as e:
        logger.error(f"Error in captain approval callbacks: {e}")
        await query.answer("❌ An error occurred!")

async def handle_auction_sale_callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle auction sale confirmation callbacks"""
    try:
        query = update.callback_query
        user = query.from_user
        data = query.data
        
        if data.startswith("confirm_sale_"):
            parts = data.split('_')
            auction_id = int(parts[2])
            
            auction = bot_instance.get_approved_auction(auction_id)
            if not auction:
                await query.edit_message_text("❌ Auction not found!")
                return
            
            if auction.creator_id != user.id and not bot_instance.is_admin(user.id):
                await query.edit_message_text("❌ Only auction host or admin can confirm sales!")
                return
            
            lock_acquired = auction._bid_lock.acquire(blocking=False)
            if not lock_acquired:
                await query.answer("⏳ Sale already being processed!")
                return
            
            try:
                current_player = auction.current_player
                if not current_player:
                    await query.edit_message_text("❌ No current player!")
                    return
                
                if hasattr(auction, 'sold_players') and current_player.user_id in auction.sold_players:
                    await query.edit_message_text(
                        f"⚠️ <b>Already Sold!</b>\n\n"
                        f"👤 <b>{current_player.name}</b> has already been sold.\n"
                        f"Use /next {auction_id} to move to next player.",
                        parse_mode='HTML'
                    )
                    return
                
                if auction.highest_bidder and auction.current_player:
                    winning_captain = None
                    for cap in auction.approved_captains.values():
                        if cap.user_id == auction.highest_bidder:
                            winning_captain = cap
                            break
                    
                    if winning_captain:
                        if winning_captain.purse < auction.highest_bid:
                            await query.edit_message_text(
                                f"❌ <b>Error!</b> {winning_captain.team_name} has insufficient funds!\n\n"
                                f"💎 <b>Bid:</b> {format_amount(auction.highest_bid)}\n"
                                f"💰 <b>Purse:</b> {format_amount(winning_captain.purse)}\n\n"
                                f"🔄 Use /rebid {auction_id} to restart bidding.",
                                parse_mode='HTML'
                            )
                            return
                        
                        winning_captain.purse -= auction.highest_bid
                        winning_captain.spent = getattr(winning_captain, 'spent', 0) + auction.highest_bid
                        
                        if not hasattr(winning_captain, 'players'):
                            winning_captain.players = []
                        winning_captain.players.append(auction.current_player.name)
                        
                        if not hasattr(auction, 'sold_players'):
                            auction.sold_players = {}
                        auction.sold_players[auction.current_player.user_id] = {
                            'player': auction.current_player,
                            'team': winning_captain.team_name,
                            'amount': auction.highest_bid,
                            'captain': winning_captain.name
                        }
                    
                        sold_message = (
                            f"✅ <b>SOLD!</b> ✅\n\n"
                            f"👤 <b>Player:</b> {current_player.name}\n"
                            f"👑 <b>Team:</b> {winning_captain.team_name}\n"
                            f"💰 <b>Sold for:</b> {format_amount(auction.highest_bid)}\n"
                            f"💳 <b>Remaining Purse:</b> {format_amount(winning_captain.purse)}\n\n"
                        )
                    
                    auction.current_player_index += 1
                    
                    if auction.current_player_index < len(auction.player_queue):
                        next_player = auction.player_queue[auction.current_player_index]
                        auction.current_player = next_player
                        auction.highest_bidder = None  # Sync with current_bids reset
                        auction.highest_bid = auction.base_price
                        auction.current_bids = {}  # Reset bids for new player
                        
                        captain_purses = "\n".join([
                            f"👑 {cap.team_name}: {format_amount(cap.purse)}"
                            for cap in auction.approved_captains.values()
                        ])
                        
                        username_display = f"@{next_player.username}" if hasattr(next_player, 'username') and next_player.username else ""
                        sold_message += (
                            f"\n🎯 <b>NEXT PLAYER</b>\n\n"
                            f"👤 <b>{next_player.name}</b> {username_display}\n"
                            f"💰 <b>Base Price:</b> {format_amount(next_player.base_price)}\n"
                            f"📊 <b>Player {auction.current_player_index + 1}/{len(auction.player_queue)}</b>\n\n"
                            f"💳 <b>Team Purses:</b>\n{captain_purses}\n\n"
                            f"🎯 <b>Captains, type your bid!</b>"
                        )
                    else:
                        auction.status = "completed"
                        sold_message += "🏆 <b>AUCTION COMPLETED!</b>\n\nAll players have been sold!"
                    
                    await query.edit_message_text(sold_message, parse_mode='HTML')
                    
                    await query.answer(f"✅ {current_player.name} sold to {winning_captain.team_name if winning_captain else 'Unknown'}!")
                    
                    if auction.group_chat_id:
                        try:
                            await context.bot.send_message(
                                chat_id=auction.group_chat_id,
                                text=sold_message,
                                parse_mode='HTML'
                            )
                        except Exception as e:
                            logger.error(f"Failed to send sale notification to group: {e}")
                else:
                    await query.edit_message_text("❌ Captain not found for sale confirmation!")
            
            finally:
                auction._bid_lock.release()
        
        elif data.startswith("continue_bid_"):
            auction_id = int(data.split('_')[2])
            auction = bot_instance.get_approved_auction(auction_id)
            
            current_status = ""
            if auction and auction.current_player:
                current_status = (
                    f"👤 <b>Current:</b> {auction.current_player.name}\n"
                    f"💰 <b>Highest Bid:</b> {format_amount(auction.highest_bid)}\n\n"
                )
            
            await query.edit_message_text(
                f"🔄 <b>Bidding Continues!</b>\n\n"
                f"{current_status}"
                f"💡 <b>Captains can continue placing bids.</b>\n"
                f"📝 <b>Admin:</b> Reply '..' or use /sell {auction_id} when ready.",
                parse_mode='HTML'
            )
            await query.answer("Bidding resumed!")
        
        await query.answer()
        
    except telegram_error.BadRequest as e:
        if "query is too old" in str(e).lower():
            logger.warning(f"Sale callback query too old: {e}")
            return
        else:
            logger.error(f"BadRequest in auction sale callbacks: {e}")
            try:
                await query.answer("❌ Request failed - please try again")
            except:
                pass
    except Exception as e:
        logger.error(f"Error in auction sale callbacks: {e}")
        try:
            await query.answer("❌ An error occurred!")
        except:
            pass

async def handle_player_approval_callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle player approve/reject callbacks"""
    try:
        query = update.callback_query
        user = query.from_user
        data = query.data
        
        parts = data.split('_')
        auction_id = int(parts[2])
        player_id = int(parts[3])
        
        auction = bot_instance.get_approved_auction(auction_id)
        if not auction or (auction.creator_id != user.id and not bot_instance.is_admin(user.id)):
            await query.edit_message_text("❌ Access denied!")
            return
        
        if data.startswith("approve_player_"):
            success = bot_instance.approve_player(auction_id, player_id)
            if success:
                player_name = auction.approved_players[player_id].name
                
                try:
                    await context.bot.send_message(
                        chat_id=player_id,
                        text=f"🎉 <b>You're approved for the auction!</b>\n\n"
                             f"🏆 <b>Auction:</b> {auction.name}\n"
                             f"💎 <b>Base Price:</b> {auction.base_price}Cr\n\n"
                             f"✅ <b>You're now in the player pool!</b>",
                        parse_mode='HTML'
                    )
                except:
                    pass
                
                await query.edit_message_text(
                    f"✅ <b>APPROVED</b> by {user.first_name}\n\n"
                    f"👤 <b>Player:</b> {player_name}\n"
                    f"🏆 <b>Auction:</b> {auction.name}\n"
                    f"⏰ <b>Approved:</b> {datetime.now().strftime('%H:%M:%S')}",
                    parse_mode='HTML'
                )
                
                if auction.group_chat_id:
                    try:
                        await context.bot.send_message(
                            chat_id=auction.group_chat_id,
                            text=f"✅ <b>Player Approved!</b>\n\n"
                                 f"👤 <b>Player:</b> {player_name}\n"
                                 f"🏆 <b>Auction:</b> {auction.name}\n"
                                 f"👑 <b>Approved by:</b> {user.first_name}\n\n"
                                 f"🎉 <b>{player_name} is now eligible for the auction!</b>",
                            parse_mode='HTML'
                        )
                    except Exception as e:
                        logger.error(f"Failed to send group notification: {e}")
        
        elif data.startswith("reject_player_"):
            success = bot_instance.reject_player(auction_id, player_id)
            if success:
                registration = auction.registered_players.get(player_id)
                player_name = registration.name if registration else "Unknown"
                
                try:
                    await context.bot.send_message(
                        chat_id=player_id,
                        text=f"❌ <b>Your player registration was rejected</b>\n\n"
                             f"🏆 <b>Auction:</b> {auction.name}\n\n"
                             f"You can try registering for other auctions.",
                        parse_mode='HTML'
                    )
                except:
                    pass
                
                await query.edit_message_text(
                    f"❌ <b>REJECTED</b> by {user.first_name}\n\n"
                    f"👤 <b>Player:</b> {player_name}\n"
                    f"🏆 <b>Auction:</b> {auction.name}\n"
                    f"⏰ <b>Rejected:</b> {datetime.now().strftime('%H:%M:%S')}",
                    parse_mode='HTML'
                )
                
                if auction.group_chat_id:
                    try:
                        await context.bot.send_message(
                            chat_id=auction.group_chat_id,
                            text=f"❌ <b>Player Rejected</b>\n\n"
                                 f"👤 <b>Player:</b> {player_name}\n"
                                 f"🏆 <b>Auction:</b> {auction.name}\n"
                                 f"👑 <b>Rejected by:</b> {user.first_name}",
                            parse_mode='HTML'
                        )
                    except Exception as e:
                        logger.error(f"Failed to send group notification: {e}")
        
        await query.answer()
        
    except Exception as e:
        logger.error(f"Error in player approval callbacks: {e}")
        await query.answer("❌ An error occurred!")

# ====================================
# ====================================

@check_banned  
async def myteam_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show captain's current team"""
    try:
        user = update.effective_user
        
        user_auction = None
        captain = None
        
        for auction in bot_instance.approved_auctions.values():
            if user.id in auction.approved_captains:
                user_auction = auction
                captain = auction.approved_captains[user.id]
                break
        
        if not user_auction or not captain:
            await update.message.reply_text("❌ You are not a captain in any auction!")
            return
        
        message = (
            f"🏏 <b>My Team - {captain.team_name}</b>\n\n"
            f"🏆 <b>Auction:</b> {user_auction.name}\n"
            f"💰 <b>Remaining Purse:</b> {format_amount(captain.purse)}\n"
            f"👥 <b>Players ({len(captain.players)}):</b>\n\n"
        )
        
        total_spent = 0
        for i, player_name in enumerate(captain.players, 1):
            player_price = 0
            if hasattr(user_auction, 'sold_players'):
                for sold_info in user_auction.sold_players.values():
                    if sold_info['player'].name == player_name and sold_info['captain'] == captain.name:
                        player_price = sold_info['amount']
                        break
            
            message += f"{i}. {player_name} - {format_amount(player_price)}\n"
            total_spent += player_price
        
        if not captain.players:
            message += "<i>No players bought yet</i>\n\n"
        else:
            message += f"\n💸 <b>Total Spent:</b> {format_amount(total_spent)}\n"
        
        await update.message.reply_text(message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Error in myteam_command: {e}")
        await update.message.reply_text("❌ An error occurred!")

@check_banned
async def purse_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show captain's remaining purse"""
    try:
        user = update.effective_user
        
        for auction in bot_instance.approved_auctions.values():
            if user.id in auction.approved_captains:
                captain = auction.approved_captains[user.id]
                
                spent = 0
                if hasattr(auction, 'sold_players'):
                    for sold_info in auction.sold_players.values():
                        if sold_info['captain'] == captain.name:
                            spent += sold_info['amount']
                
                await update.message.reply_text(
                    f"💰 <b>Team Purse - {captain.team_name}</b>\n\n"
                    f"🏆 <b>Auction:</b> {auction.name}\n"
                    f"💵 <b>Total Budget:</b> {format_amount(auction.purse)}\n"
                    f"💸 <b>Spent:</b> {format_amount(spent)}\n"
                    f"💰 <b>Remaining:</b> {format_amount(captain.purse)}\n"
                    f"👥 <b>Players:</b> {len(captain.players)}",
                    parse_mode='HTML'
                )
                return
        
        await update.message.reply_text("❌ You are not a captain in any auction!")
        
    except Exception as e:
        logger.error(f"Error in purse_command: {e}")
        await update.message.reply_text("❌ An error occurred!")

@check_banned
async def transfercap_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Transfer captaincy to another user during auction"""
    try:
        user = update.effective_user
        
        if len(context.args) < 2:
            await update.message.reply_text(
                "❌ <b>Usage:</b> <code>/transfercap [auction_id] [new_captain_user_id or @username]</code>\n\n"
                "💡 <b>Example:</b>\n"
                "• <code>/transfercap 1 123456789</code>\n"
                "• <code>/transfercap 1 @username</code>\n\n"
                "📝 <b>Note:</b> You must be the current captain to transfer!",
                parse_mode='HTML'
            )
            return
        
        try:
            auction_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("❌ Invalid auction ID!")
            return
        
        auction = bot_instance.get_approved_auction(auction_id)
        if not auction:
            await update.message.reply_text("❌ Auction not found!")
            return
        
        if user.id not in auction.approved_captains:
            await update.message.reply_text("❌ You are not a captain in this auction!")
            return
        
        current_captain = auction.approved_captains[user.id]
        
        new_captain_identifier = context.args[1]
        
        new_captain_id = None
        new_captain_name = None
        
        if new_captain_identifier.startswith('@'):
            username = new_captain_identifier[1:]
            try:
                chat = await context.bot.get_chat(f"@{username}")
                new_captain_id = chat.id
                new_captain_name = chat.full_name or chat.first_name or username
            except Exception as e:
                await update.message.reply_text(f"❌ Could not find user @{username} on Telegram!")
                return
        else:
            try:
                new_captain_id = int(new_captain_identifier)
                try:
                    chat = await context.bot.get_chat(new_captain_id)
                    new_captain_name = chat.full_name or chat.first_name or f"User {new_captain_id}"
                except:
                    new_captain_name = f"User {new_captain_id}"
            except ValueError:
                await update.message.reply_text("❌ Invalid user ID!")
                return
        
        if new_captain_id in auction.approved_captains:
            await update.message.reply_text("❌ That user is already a captain in this auction!")
            return
        
        if new_captain_id in auction.approved_players or new_captain_id in auction.registered_players:
            await update.message.reply_text("❌ That user is registered as a player! Cannot be both captain and player.")
            return
        
        new_approved_captain = ApprovedCaptain(
            user_id=new_captain_id,
            name=new_captain_name,
            team_name=current_captain.team_name,
            purse=current_captain.purse
        )
        
        new_approved_captain.spent = current_captain.spent
        new_approved_captain.players = current_captain.players.copy()
        new_approved_captain.approved_at = current_captain.approved_at
        
        del auction.approved_captains[user.id]
        auction.approved_captains[new_captain_id] = new_approved_captain
        
        if hasattr(auction, 'highest_bidder') and auction.highest_bidder == user.id:
            auction.highest_bidder = new_captain_id
        
        if hasattr(auction, 'current_bids') and user.id in auction.current_bids:
            auction.current_bids[new_captain_id] = auction.current_bids[user.id]
            del auction.current_bids[user.id]
        
        if hasattr(auction, 'sold_players') and auction.sold_players:
            for player_id, sold_info in auction.sold_players.items():
                if sold_info.get('captain_id') == user.id:
                    sold_info['captain_id'] = new_captain_id
                    sold_info['captain'] = new_captain_name
        
        await update.message.reply_text(
            f"✅ <b>Captaincy Transferred!</b>\n\n"
            f"🏏 <b>Team:</b> {current_captain.team_name}\n"
            f"👤 <b>From:</b> {user.full_name or user.first_name}\n"
            f"➡️ <b>To:</b> {new_captain_name}\n"
            f"💰 <b>Purse:</b> {format_amount(current_captain.purse)}\n"
            f"👥 <b>Players:</b> {len(current_captain.players)}\n\n"
            f"📢 <b>New captain can now bid and manage the team!</b>",
            parse_mode='HTML'
        )
        
        try:
            await context.bot.send_message(
                chat_id=new_captain_id,
                text=f"🎉 <b>You are now the captain!</b>\n\n"
                     f"🏏 <b>Team:</b> {current_captain.team_name}\n"
                     f"🏆 <b>Auction:</b> {auction.name}\n"
                     f"💰 <b>Purse:</b> {format_amount(current_captain.purse)}\n"
                     f"👥 <b>Players:</b> {len(current_captain.players)}\n\n"
                     f"📝 <b>You can now:</b>\n"
                     f"• Bid in the auction\n"
                     f"• Check team: /myteam\n"
                     f"• Check purse: /purse\n"
                     f"• Transfer to someone else: /transfercap {auction_id} [user_id]",
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Could not notify new captain: {e}")
        
    except Exception as e:
        logger.error(f"Error in transfercap_command: {e}")
        await update.message.reply_text("❌ An error occurred!")

@check_banned
async def auctionhelp_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show comprehensive auction system help"""
    try:
        message = (
            f"🏆 <b>CRICKET AUCTION SYSTEM HELP</b>\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📋 <b>GETTING STARTED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            
            f"<b>1️⃣ Create Auction:</b>\n"
            f"• <code>/register</code> - Start new auction proposal\n"
            f"• Fill auction details step by step\n"
            f"• Wait for admin approval\n\n"
            
            f"<b>2️⃣ Host Panel:</b>\n"
            f"• <code>/hostpanel [id]</code> - Access host controls\n"
            f"• Manage registrations and start auction\n\n"
            
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"👑 <b>CAPTAIN COMMANDS</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            
            f"<b>Registration:</b>\n"
            f"• <code>/regcap [id] [team_name]</code>\n"
            f"• Wait for host approval\n\n"
            
            f"<b>During Auction:</b>\n"
            f"• Type amount to bid (e.g., <code>5</code> for 5)\n"
            f"• <code>/out [id]</code> - Stop bidding\n"
            f"• <code>/myteam [id]</code> - View your team\n"
            f"• <code>/purse [id]</code> - Check remaining budget\n\n"
            
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"👥 <b>PLAYER COMMANDS</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            
            f"• <code>/regplay [id]</code>\n"
            f"• Wait for host approval to join auction pool\n\n"
            
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🔧 <b>MANAGEMENT COMMANDS</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            
            f"<b>Information:</b>\n"
            f"• <code>/participants [id]</code> - View all participants\n"
            f"• <code>/list_auctions</code> - List all auctions (Admin)\n"
            f"• <code>/pending_auctions</code> - View pending proposals (Admin)\n\n"
            
            f"<b>Group Chat Setup:</b>\n"
            f"• <code>/setgc [id]</code> - Set group for notifications\n\n"
            
            f"<b>Admin Controls:</b>\n"
            f"• <code>/delete_auction [id]</code> - Delete auction\n"
            f"• <code>/force_end_auction [id]</code> - Force end active auction\n"
            f"• <code>/rebid [id]</code> - Reset bidding for current player\n"
            f"• <code>/addpauc [id]</code> - Manually add players (reply to usernames)\n"
            f"• <code>/removepauc [id]</code> - Remove players (reply to usernames)\n\n"
            
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>AUCTION FLOW</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            
            f"<b>Step 1:</b> Create auction with /register\n"
            f"<b>Step 2:</b> Admin approves proposal\n"
            f"<b>Step 3:</b> Host starts captain registration\n"
            f"<b>Step 4:</b> Captains register for teams\n"
            f"<b>Step 5:</b> Host approves captains\n"
            f"<b>Step 6:</b> Host starts player registration\n"
            f"<b>Step 7:</b> Players register for auction\n"
            f"<b>Step 8:</b> Host approves players\n"
            f"<b>Step 9:</b> Host starts live auction\n"
            f"<b>Step 10:</b> Captains bid on players\n\n"
            
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💡 <b>PRO TIPS</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            
            f"• Set up group chat with /setgc for live updates\n"
            f"• Use host panel for easy management\n"
            f"• Players get 10-second countdown during auction\n"
            f"• All auction data is saved across bot restarts\n"
            f"• Captains can view team and budget anytime\n\n"
            
            f"🆘 <b>Need help?</b> Contact admins or use /help for general bot commands!"
        )
        
        await update.message.reply_text(message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Error in auctionhelp_command: {e}")
        await update.message.reply_text("❌ An error occurred!")

@check_banned
async def auction_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start manual auction with first player"""
    try:
        user = update.effective_user
        
        if len(context.args) < 1:
            await update.message.reply_text(
                "❌ <b>Usage:</b> <code>/auction [auction_id]</code>\n\n"
                "📋 <b>Example:</b> <code>/auction 1</code>",
                parse_mode='HTML'
            )
            return
        
        try:
            auction_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("❌ Invalid auction ID!")
            return
        
        auction = bot_instance.get_approved_auction(auction_id)
        if not auction:
            await update.message.reply_text("❌ Auction not found!")
            return
        
        if auction.creator_id != user.id and not bot_instance.is_admin(user.id):
            await update.message.reply_text("❌ Only auction host or admin can start auction!")
            return
        
        if auction.status != "ready":
            await update.message.reply_text("❌ Auction is not ready to start!")
            return
        
        if not auction.approved_players:
            await update.message.reply_text("❌ No approved players found!")
            return
        
        success = bot_instance.start_auction_bidding(auction_id)
        if success:
            current_player = auction.current_player
            username_display = f"@{current_player.username}" if hasattr(current_player, 'username') and current_player.username else ""
            
            message = (
                f"🔥 <b>AUCTION STARTED!</b> 🔥\n\n"
                f"🏆 <b>Auction:</b> {auction.name}\n"
                f"👤 <b>Current Player:</b> {current_player.name}\n"
                f"🆔 <b>Username:</b> {username_display}\n"
                f"💎 <b>Base Price:</b> {format_amount(current_player.base_price)}\n\n"
                f"👑 <b>Captains ({len(auction.approved_captains)}):</b>\n"
            )
            
            for captain in auction.approved_captains.values():
                message += f"• {captain.team_name} - {format_amount(captain.purse)} remaining\n"
            
            message += (
                f"\n🎯 <b>Captains:</b> Type your bid (e.g., 4, 10, 15)\n"
                f"🎪 <b>Admin:</b> Reply with '..' or use /sell {auction_id}\n"
                f"⚡ <b>Let the bidding begin!</b>"
            )
            
            await update.message.reply_text(message, parse_mode='HTML')
            
            if auction.group_chat_id and auction.group_chat_id != update.effective_chat.id:
                try:
                    await context.bot.send_message(
                        chat_id=auction.group_chat_id,
                        text=message,
                        parse_mode='HTML'
                    )
                except Exception as e:
                    logger.error(f"Failed to send auction start to group: {e}")
        else:
            await update.message.reply_text("❌ Failed to start auction!")
            
    except Exception as e:
        logger.error(f"Error in auction_command: {e}")
        await update.message.reply_text("❌ An error occurred!")

@check_banned
async def rebid_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset bidding for current player (Admin/Host only)"""
    try:
        user = update.effective_user
        
        if len(context.args) < 1:
            await update.message.reply_text(
                "❌ <b>Usage:</b> <code>/rebid [auction_id]</code>\n\n"
                "💡 <b>Example:</b> <code>/rebid 1</code>\n\n"
                "<i>This will reset bidding for the current player</i>",
                parse_mode='HTML'
            )
            return
        
        try:
            auction_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("❌ Invalid auction ID!")
            return
        
        auction = bot_instance.get_approved_auction(auction_id)
        if not auction:
            await update.message.reply_text("❌ Auction not found!")
            return
        
        if auction.creator_id != user.id and not bot_instance.is_admin(user.id):
            await update.message.reply_text("❌ Only auction host or admin can rebid!")
            return
        
        if auction.status != "active":
            await update.message.reply_text("❌ Auction is not active!")
            return
        
        if not auction.current_player:
            await update.message.reply_text("❌ No current player to rebid!")
            return
        
        current_player = auction.current_player
        auction.highest_bidder = None
        auction.highest_bid = auction.base_price
        if hasattr(auction, 'current_bids'):
            auction.current_bids = {}
        
        if hasattr(auction, 'sold_players') and current_player.user_id in auction.sold_players:
            del auction.sold_players[current_player.user_id]
            logger.info(f"Removed {current_player.name} from sold_players for rebid")
        
        username_display = f"@{current_player.username}" if hasattr(current_player, 'username') and current_player.username else ""
        rebid_message = (
            f"🔄 <b>REBID STARTED!</b>\n\n"
            f"👤 <b>{current_player.name}</b> {username_display}\n"
            f"💎 <b>Base:</b> {format_amount(current_player.base_price)}\n\n"
            f"🎯 <b>Fresh start - Type amount to bid!</b>\n"
            f"📝 Admin: Reply '..' or use /sell {auction_id} to sell"
        )
        
        await update.message.reply_text(rebid_message, parse_mode='HTML')
        
        if auction.group_chat_id and auction.group_chat_id != update.effective_chat.id:
            try:
                await context.bot.send_message(
                    chat_id=auction.group_chat_id,
                    text=rebid_message,
                    parse_mode='HTML'
                )
            except Exception as e:
                logger.error(f"Failed to send rebid notification to group: {e}")
        
        logger.info(f"Rebid initiated for player {current_player.name} in auction {auction_id}")
        
    except Exception as e:
        logger.error(f"Error in rebid_command: {e}")
        await update.message.reply_text("❌ An error occurred!")

@check_banned
async def sell_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sell current player to highest bidder (Admin/Host only)"""
    try:
        user = update.effective_user
        
        if len(context.args) < 1:
            await update.message.reply_text(
                "❌ <b>Usage:</b> <code>/sell [auction_id]</code>\n\n"
                "💡 <b>Example:</b> <code>/sell 1</code>",
                parse_mode='HTML'
            )
            return
        
        try:
            auction_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("❌ Invalid auction ID!")
            return
        
        auction = bot_instance.get_approved_auction(auction_id)
        if not auction:
            await update.message.reply_text("❌ Auction not found!")
            return
        
        if auction.creator_id != user.id and not bot_instance.is_admin(user.id):
            await update.message.reply_text("❌ Only auction host or admin can sell players!")
            return
        
        if auction.status != "active":
            await update.message.reply_text("❌ Auction is not active!")
            return
        
        if not auction.current_player:
            await update.message.reply_text("❌ No current player to sell!")
            return
        
        if auction.highest_bidder:
            captain = None
            for cap in auction.approved_captains.values():
                if cap.user_id == auction.highest_bidder:
                    captain = cap
                    break
            
            keyboard = [
                [
                    InlineKeyboardButton("✅ Confirm Sale", callback_data=f"confirm_sale_{auction.id}"),
                    InlineKeyboardButton("🔄 Continue Bidding", callback_data=f"continue_bid_{auction.id}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            confirm_message = (
                f"⚠️ <b>GOING ONCE... GOING TWICE!</b> ⚠️\n\n"
                f"👤 <b>{auction.current_player.name}</b>\n"
                f"💰 <b>Final Bid:</b> {format_amount(auction.highest_bid)}\n"
                f"👑 <b>Winning Team:</b> {captain.team_name if captain else 'Unknown'}\n\n"
                f"❓ <b>Confirm this sale?</b>"
            )
            
            await update.message.reply_text(
                confirm_message,
                parse_mode='HTML',
                reply_markup=reply_markup
            )
        else:
            await update.message.reply_text(
                f"📤 <b>UNSOLD!</b>\n\n"
                f"👤 <b>{auction.current_player.name}</b>\n"
                f"💔 No bids received (Base: {auction.current_player.base_price}Cr)",
                parse_mode='HTML'
            )
            
            if not hasattr(auction, 'sold_players'):
                auction.sold_players = {}
            auction.sold_players[auction.current_player.user_id] = {
                'player': auction.current_player,
                'team': 'UNSOLD',
                'amount': 0,
                'captain': None
            }
            
            auction.current_player_index += 1
            
            if auction.current_player_index < len(auction.player_queue):
                next_player = auction.player_queue[auction.current_player_index]
                auction.current_player = next_player
                auction.highest_bidder = None  # Sync with current_bids reset
                auction.highest_bid = auction.base_price
                auction.current_bids = {}  # Reset bids for new player
                
                username_display = f"@{next_player.username}" if hasattr(next_player, 'username') and next_player.username else ""
                next_message = (
                    f"🔥 <b>NEXT PLAYER</b>\n\n"
                    f"👤 <b>{next_player.name}</b> {username_display}\n"
                    f"💎 <b>Base:</b> {format_amount(next_player.base_price)}\n\n"
                    f"🎯 <b>Type amount to bid!</b>\n"
                    f"📝 Admin: Use /sell {auction_id} or reply '..' to sell"
                )
                
                await update.message.reply_text(next_message, parse_mode='HTML')
            else:
                auction.status = "completed"
                await update.message.reply_text(
                    f"🏆 <b>AUCTION COMPLETED!</b>\n\n🎊 All players auctioned for {auction.name}!",
                    parse_mode='HTML'
                )
                
    except Exception as e:
        logger.error(f"Error in sell_command: {e}")
        try:
            await update.message.reply_text("❌ An error occurred!")
        except:
            pass

@check_banned
async def setgc_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set group chat ID for auction"""
    try:
        user = update.effective_user
        
        if len(context.args) < 2:
            await update.message.reply_text(
                "❌ <b>Usage:</b> <code>/setgc [auction_id] [group_chat_id]</code>\n\n"
                "<b>Example:</b> <code>/setgc 5 -1001234567890</code>",
                parse_mode='HTML'
            )
            return
        
        try:
            auction_id = int(context.args[0])
            group_chat_id = int(context.args[1])
        except ValueError:
            await update.message.reply_text("❌ Please provide valid numbers for auction ID and chat ID.")
            return
        
        auction = bot_instance.get_approved_auction(auction_id)
        if not auction:
            await update.message.reply_text("❌ Auction not found!")
            return
        
        if auction.creator_id != user.id and not bot_instance.is_admin(user.id):
            await update.message.reply_text("❌ Only the auction host can set group chat!")
            return
        
        auction.group_chat_id = group_chat_id
        await update.message.reply_text(
            f"✅ <b>Group Chat Set!</b>\n\n"
            f"🏆 <b>Auction:</b> {auction.name}\n"
            f"💬 <b>Group Chat ID:</b> {group_chat_id}\n\n"
            f"📝 <b>All auction messages will be sent to this group.</b>",
            parse_mode='HTML'
        )
        
    except Exception as e:
        logger.error(f"Error in setgc_command: {e}")
        await update.message.reply_text("❌ An error occurred!")

@check_banned
async def unsold_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """View all unsold players in an auction"""
    try:
        if len(context.args) < 1:
            await update.message.reply_text(
                "❌ <b>Usage:</b> <code>/unsold [auction_id]</code>\n\n"
                "<b>Example:</b> <code>/unsold 5</code>",
                parse_mode='HTML'
            )
            return
        
        auction_id = int(context.args[0])
        auction = bot_instance.get_approved_auction(auction_id)
        
        if not auction:
            await update.message.reply_text("❌ Auction not found!")
            return
        
        if not hasattr(auction, 'unsold_players'):
            auction.unsold_players = {}
        
        if not auction.unsold_players:
            await update.message.reply_text(
                f"✅ <b>No Unsold Players</b>\n\n"
                f"🏆 <b>Auction:</b> {auction.name}\n"
                f"📊 All players have been sold or are still in queue!",
                parse_mode='HTML'
            )
            return
        
        message = f"🔴 <b>UNSOLD PLAYERS</b>\n\n🏆 <b>Auction:</b> {auction.name}\n\n"
        
        for i, player in enumerate(auction.unsold_players.values(), 1):
            username_display = f"@{player.username}" if hasattr(player, 'username') and player.username else "N/A"
            message += f"{i}. <b>{player.name}</b>\n"
            message += f"   🆔 ID: {player.user_id}\n"
            message += f"   👤 Username: {username_display}\n"
            message += f"   💎 Base: {format_amount(player.base_price)}\n\n"
        
        message += f"📝 <b>Total Unsold:</b> {len(auction.unsold_players)}\n\n"
        message += f"💡 Use <code>/addpt {auction_id} [player_name] [team_name] [amount]</code> to assign"
        
        await update.message.reply_text(message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Error in unsold_command: {e}")
        await update.message.reply_text("❌ An error occurred!")

@check_banned
async def addpt_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manually assign unsold player to a team using username and team name"""
    try:
        user = update.effective_user
        
        if len(context.args) < 4:
            await update.message.reply_text(
                "❌ <b>Usage:</b> <code>/addpt [auction_id] [player_name] [team_name] [amount]</code>\n\n"
                "<b>Examples:</b>\n"
                "• <code>/addpt 5 @john TeamA 10</code>\n"
                "• <code>/addpt 5 \"Kevin Pietersen\" Parshwa 10</code>\n"
                "• <code>/addpt 2 Kevin_Pietersen Parshwa 1</code> (use underscore for spaces)\n\n"
                "💡 For multi-word names, use quotes or underscores",
                parse_mode='HTML'
            )
            return
        
        try:
            auction_id = int(context.args[0])
            amount = int(context.args[-1])  # Last argument is amount
            
            middle_args = context.args[1:-1]
            
            if len(middle_args) < 2:
                await update.message.reply_text(
                    "❌ <b>Invalid format!</b>\n\n"
                    "Usage: <code>/addpt [auction_id] [player_name] [team_name] [amount]</code>\n\n"
                    "💡 For names with spaces use underscore: Kevin_Pietersen",
                    parse_mode='HTML'
                )
                return
            
            team_name = middle_args[-1].strip()
            player_username = ' '.join(middle_args[:-1]).strip().replace('_', ' ')
        except (ValueError, IndexError):
            await update.message.reply_text(
                "❌ <b>Invalid format!</b>\n\n"
                "Usage: <code>/addpt [auction_id] [player_name] [team_name] [amount]</code>\n\n"
                "<b>Example:</b> <code>/addpt 2 Kevin_Pietersen Parshwa 1</code>",
                parse_mode='HTML'
            )
            return
        
        auction = bot_instance.get_approved_auction(auction_id)
        
        if not auction:
            await update.message.reply_text("❌ Auction not found!")
            return
        
        if auction.creator_id != user.id and not bot_instance.is_admin(user.id):
            await update.message.reply_text("❌ Only the auction host or admin can assign unsold players!")
            return
        
        success, message_text, player, captain = bot_instance.assign_unsold_player(auction_id, player_username, team_name, amount)
        
        if success:
            message = (
                f"✅ <b>UNSOLD PLAYER ASSIGNED!</b>\n\n"
                f"👤 <b>Player:</b> {player.name}\n"
                f"👑 <b>Team:</b> {captain.team_name}\n"
                f"💰 <b>Amount:</b> {format_amount(amount)}\n"
                f"💼 <b>Remaining Purse:</b> {format_amount(captain.purse)}"
            )
            
            await update.message.reply_text(message, parse_mode='HTML')
            
            if auction.group_chat_id:
                try:
                    await context.bot.send_message(
                        chat_id=auction.group_chat_id,
                        text=message,
                        parse_mode='HTML'
                    )
                except:
                    pass
        else:
            if message_text == "Insufficient funds" and player and captain:
                await update.message.reply_text(
                    f"❌ <b>Insufficient Funds!</b>\n\n"
                    f"👑 <b>Captain:</b> {captain.name}\n"
                    f"💰 <b>Available:</b> {format_amount(captain.purse)}\n"
                    f"💸 <b>Required:</b> {format_amount(amount)}",
                    parse_mode='HTML'
                )
            else:
                await update.message.reply_text(f"❌ {message_text}")
        
    except ValueError:
        await update.message.reply_text("❌ Invalid auction ID or amount! Must be numbers.")
    except Exception as e:
        logger.error(f"Error in addpt_command: {e}")
        await update.message.reply_text("❌ An error occurred!")

@check_banned
async def pauseauc_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Pause or resume an active auction"""
    try:
        user = update.effective_user
        
        if len(context.args) < 1:
            await update.message.reply_text(
                "❌ <b>Usage:</b> <code>/pauseauc [auction_id]</code>\n\n"
                "<b>Example:</b> <code>/pauseauc 5</code>",
                parse_mode='HTML'
            )
            return
        
        auction_id = int(context.args[0])
        auction = bot_instance.get_approved_auction(auction_id)
        
        if not auction:
            await update.message.reply_text("❌ Auction not found!")
            return
        
        if auction.status != "active":
            await update.message.reply_text("❌ Auction is not active!")
            return
        
        if auction.creator_id != user.id and not bot_instance.is_admin(user.id):
            await update.message.reply_text("❌ Only the auction host or admin can pause/resume!")
            return
        
        success = bot_instance.pause_auction(auction_id)
        
        if success:
            status_text = "PAUSED" if auction.is_paused else "RESUMED"
            emoji = "⏸️" if auction.is_paused else "▶️"
            
            message = (
                f"{emoji} <b>AUCTION {status_text}!</b>\n\n"
                f"🏆 <b>Auction:</b> {auction.name}\n"
                f"📊 <b>Status:</b> {status_text}\n\n"
            )
            
            if auction.is_paused:
                message += "⏸️ <b>Bidding is temporarily paused</b>\n💡 Use /pauseauc again to resume"
            else:
                message += "▶️ <b>Bidding has resumed!</b>\n🎯 Captains can continue bidding"
            
            await update.message.reply_text(message, parse_mode='HTML')
            
            if auction.group_chat_id:
                try:
                    await context.bot.send_message(
                        chat_id=auction.group_chat_id,
                        text=message,
                        parse_mode='HTML'
                    )
                except:
                    pass
        else:
            await update.message.reply_text("❌ Failed to pause/resume auction!")
        
    except Exception as e:
        logger.error(f"Error in pauseauc_command: {e}")
        await update.message.reply_text("❌ An error occurred!")

@check_banned
async def rebid_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Bring back any player for rebidding using username"""
    try:
        user = update.effective_user
        
        if len(context.args) < 2:
            await update.message.reply_text(
                "❌ <b>Usage:</b> <code>/rebid [auction_id] [player_username]</code>\n\n"
                "<b>Examples:</b>\n"
                "• <code>/rebid 5 @john</code>\n"
                "• <code>/rebid 5 john</code>\n\n"
                "💡 Works with @username or player name\n"
                "♻️ Works for sold players (will be removed from team) or unsold players",
                parse_mode='HTML'
            )
            return
        
        auction_id = int(context.args[0])
        player_username = ' '.join(context.args[1:]).strip()  # Support names with spaces
        
        auction = bot_instance.get_approved_auction(auction_id)
        
        if not auction:
            await update.message.reply_text("❌ Auction not found!")
            return
        
        if auction.status != "active":
            await update.message.reply_text("❌ Auction is not active!")
            return
        
        if auction.creator_id != user.id and not bot_instance.is_admin(user.id):
            await update.message.reply_text("❌ Only the auction host or admin can rebid players!")
            return
        
        success, message_text, player = bot_instance.rebid_player(auction_id, player_username)
        
        if success:
            current_player = auction.current_player
            message = (
                f"🔄 <b>PLAYER BROUGHT BACK FOR REBIDDING!</b>\n\n"
                f"👤 <b>Player:</b> {current_player.name}\n"
                f"💎 <b>Base Price:</b> {format_amount(current_player.base_price)}\n\n"
                f"🎯 <b>Captains, place your bids now!</b>"
            )
            
            await update.message.reply_text(message, parse_mode='HTML')
            
            if auction.group_chat_id:
                try:
                    await context.bot.send_message(
                        chat_id=auction.group_chat_id,
                        text=message,
                        parse_mode='HTML'
                    )
                except:
                    pass
        else:
            await update.message.reply_text(f"❌ {message_text}")
        
    except ValueError:
        await update.message.reply_text("❌ Invalid auction ID! Must be a number.")
    except Exception as e:
        logger.error(f"Error in rebid_command: {e}")
        await update.message.reply_text("❌ An error occurred!")

@check_banned
async def participants_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show all registered participants for an auction"""
    try:
        user = update.effective_user
        
        if len(context.args) < 1:
            await update.message.reply_text(
                "❌ <b>Usage:</b> <code>/participants [auction_id]</code>\n\n"
                "<b>Example:</b> <code>/participants 5</code>",
                parse_mode='HTML'
            )
            return
        
        try:
            auction_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("❌ Please provide a valid auction ID.")
            return
        
        auction = bot_instance.get_approved_auction(auction_id)
        if not auction:
            await update.message.reply_text("❌ Auction not found!")
            return
        
        if auction.creator_id != user.id and not bot_instance.is_admin(user.id):
            await update.message.reply_text("❌ Only auction host/admin can view participants!")
            return
        
        message = f"📊 <b>Auction Participants - {auction.name}</b>\n\n"
        
        if auction.approved_captains:
            message += f"👑 <b>Approved Captains ({len(auction.approved_captains)}):</b>\n"
            for captain in auction.approved_captains.values():
                message += f"• {captain.name} - {captain.team_name}\n"
            message += "\n"
        
        pending_captains = [cap for cap in auction.registered_captains.values() if cap.status == "pending"]
        if pending_captains:
            message += f"⏳ <b>Pending Captains ({len(pending_captains)}):</b>\n"
            for captain in pending_captains:
                message += f"• {captain.name} - {captain.team_name}\n"
            message += "\n"
        
        if auction.approved_players:
            message += f"👥 <b>Approved Players ({len(auction.approved_players)}):</b>\n"
            for player in auction.approved_players.values():
                message += f"• {player.name}\n"
            message += "\n"
        
        pending_players = [player for player in auction.registered_players.values() if player.status == "pending"]
        if pending_players:
            message += f"⏳ <b>Pending Players ({len(pending_players)}):</b>\n"
            for player in pending_players:
                message += f"• {player.name}\n"
            message += "\n"
        
        message += f"📊 <b>Summary:</b>\n"
        message += f"✅ Captains: {len(auction.approved_captains)}\n"
        message += f"⏳ Pending Captains: {len(pending_captains)}\n"
        message += f"✅ Players: {len(auction.approved_players)}\n"
        message += f"⏳ Pending Players: {len(pending_players)}\n"
        message += f"📋 Status: {auction.status.title()}"
        
        await update.message.reply_text(message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Error in participants_command: {e}")
        await update.message.reply_text("❌ An error occurred!")

@check_banned
async def delete_auction_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Delete an auction (Admin only)"""
    try:
        user = update.effective_user
        
        if not bot_instance.is_admin(user.id):
            await update.message.reply_text("❌ Only admins can delete auctions!")
            return
        
        if len(context.args) < 1:
            await update.message.reply_text(
                "❌ <b>Usage:</b> <code>/delete_auction [auction_id]</code>\n\n"
                "<b>Example:</b> <code>/delete_auction 5</code>",
                parse_mode='HTML'
            )
            return
        
        try:
            auction_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("❌ Please provide a valid auction ID.")
            return
        
        auction_found = False
        auction_name = ""
        
        if auction_id in bot_instance.approved_auctions:
            auction = bot_instance.approved_auctions[auction_id]
            auction_name = auction.name
            del bot_instance.approved_auctions[auction_id]
            auction_found = True
        
        elif auction_id in bot_instance.auction_proposals:
            proposal = bot_instance.auction_proposals[auction_id]
            auction_name = proposal.name
            del bot_instance.auction_proposals[auction_id]
            auction_found = True
        
        if auction_found:
            await update.message.reply_text(
                f"✅ <b>Auction Deleted!</b>\n\n"
                f"🗑️ <b>Deleted Auction:</b> {auction_name}\n"
                f"🆔 <b>ID:</b> {auction_id}\n\n"
                f"👤 <b>Deleted by Admin:</b> {user.first_name}",
                parse_mode='HTML'
            )
        else:
            await update.message.reply_text(
                f"❌ <b>Auction not found!</b>\n\n"
                f"🔍 No auction found with ID: <b>{auction_id}</b>",
                parse_mode='HTML'
            )
        
    except Exception as e:
        logger.error(f"Error in delete_auction_command: {e}")
        await update.message.reply_text("❌ An error occurred!")

@check_banned
async def list_auctions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all auctions for admin management"""
    try:
        user = update.effective_user
        
        if not bot_instance.is_admin(user.id):
            await update.message.reply_text("❌ Only admins can view auction list!")
            return
        
        message = "📋 <b>All Auctions Management</b>\n\n"
        
        if bot_instance.auction_proposals:
            message += f"📝 <b>Pending Proposals ({len(bot_instance.auction_proposals)}):</b>\n"
            for proposal_id, proposal in bot_instance.auction_proposals.items():
                message += f"• ID: {proposal_id} - {proposal.name} (by {proposal.creator_name})\n"
            message += "\n"
        
        if bot_instance.approved_auctions:
            message += f"✅ <b>Approved Auctions ({len(bot_instance.approved_auctions)}):</b>\n"
            for auction_id, auction in bot_instance.approved_auctions.items():
                status_emoji = {
                    'setup': '🔧',
                    'captain_reg': '👑',
                    'player_reg': '👥',
                    'ready': '⚡',
                    'active': '🔥',
                    'completed': '✅'
                }.get(auction.status, '❓')
                message += f"• ID: {auction_id} - {auction.name} {status_emoji} ({auction.status})\n"
            message += "\n"
        
        if not bot_instance.auction_proposals and not bot_instance.approved_auctions:
            message += "📭 <b>No auctions found</b>\n\n"
        
        message += "🔧 <b>Admin Commands:</b>\n"
        message += "• <code>/delete_auction [id]</code> - Delete any auction\n"
        message += "• <code>/force_end_auction [id]</code> - Force end active auction\n"
        message += "• <code>/pending_auctions</code> - View pending proposals"
        
        await update.message.reply_text(message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Error in list_auctions_command: {e}")
        await update.message.reply_text("❌ An error occurred!")

@check_banned
async def force_end_auction_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Force end an active auction (Admin only)"""
    try:
        user = update.effective_user
        
        if not bot_instance.is_admin(user.id):
            await update.message.reply_text("❌ Only admins can force end auctions!")
            return
        
        if len(context.args) < 1:
            await update.message.reply_text(
                "❌ <b>Usage:</b> <code>/force_end_auction [auction_id]</code>\n\n"
                "<b>Example:</b> <code>/force_end_auction 5</code>",
                parse_mode='HTML'
            )
            return
        
        try:
            auction_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("❌ Please provide a valid auction ID.")
            return
        
        auction = bot_instance.get_approved_auction(auction_id)
        if not auction:
            await update.message.reply_text("❌ Auction not found!")
            return
        
        if auction.status != "active":
            await update.message.reply_text(f"❌ Auction is not active! Current status: {auction.status}")
            return
        
        success = bot_instance.end_auction(auction_id)
        
        if success:
            await update.message.reply_text(
                f"✅ <b>Auction Force Ended!</b>\n\n"
                f"🏆 <b>Auction:</b> {auction.name}\n"
                f"🆔 <b>ID:</b> {auction_id}\n"
                f"👤 <b>Force ended by Admin:</b> {user.first_name}\n\n"
                f"📊 <b>Final Stats:</b>\n"
                f"👑 Teams: {len(auction.approved_captains)}\n"
                f"👥 Players Sold: {len(auction.sold_players)}",
                parse_mode='HTML'
            )
        else:
            await update.message.reply_text("❌ Failed to end auction!")
        
    except Exception as e:
        logger.error(f"Error in force_end_auction_command: {e}")
        await update.message.reply_text("❌ An error occurred!")

@check_banned
async def pending_auctions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """View pending auction proposals (Admin only)"""
    try:
        user = update.effective_user
        
        if not bot_instance.is_admin(user.id):
            await update.message.reply_text("❌ This command is for admins only!")
            return
        
        pending_proposals = []
        for proposal_id, proposal in bot_instance.auction_proposals.items():
            if proposal.status == "pending":
                pending_proposals.append((proposal_id, proposal))
        
        if not pending_proposals:
            await update.message.reply_text(
                "✅ <b>No Pending Auction Proposals</b>\n\n"
                "All proposals have been processed.",
                parse_mode='HTML'
            )
            return
        
        message = f"📋 <b>Pending Auction Proposals ({len(pending_proposals)})</b>\n\n"
        
        for proposal_id, proposal in pending_proposals:
            message += (
                f"🆔 <b>ID:</b> {proposal_id}\n"
                f"🏆 <b>Name:</b> {proposal.name}\n"
                f"👤 <b>Creator:</b> {proposal.creator_name}\n"
                f"👥 <b>Teams:</b> {len(proposal.teams)}\n"
                f"💰 <b>Purse:</b> {proposal.purse}Cr\n"
                f"💎 <b>Base Price:</b> {proposal.base_price}Cr\n"
                f"⏰ <b>Created:</b> {proposal.created_at.strftime('%Y-%m-%d %H:%M')}\n\n"
                f"<b>Actions:</b>\n"
                f"• Approve: <code>/approve_auction {proposal_id}</code>\n"
                f"• Reject: <code>/reject_auction {proposal_id}</code>\n"
                f"• Details: <code>/auction_details {proposal_id}</code>\n\n"
                f"{'─' * 30}\n\n"
            )
        
        await update.message.reply_text(message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Error in pending_auctions_command: {e}")
        await update.message.reply_text("❌ An error occurred!")

@check_banned
async def addpauc_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manually add players to auction by replying to message with usernames (Admin/Host only)"""
    try:
        user = update.effective_user
        
        if len(context.args) < 1:
            await update.message.reply_text(
                "❌ <b>Usage:</b> <code>/addpauc [auction_id]</code>\n\n"
                "📝 <b>Then reply to a message containing usernames</b>\n"
                "💡 <b>Example:</b> <code>/addpauc 1</code>",
                parse_mode='HTML'
            )
            return
        
        try:
            auction_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("❌ Invalid auction ID!")
            return
        
        auction = bot_instance.get_approved_auction(auction_id)
        if not auction:
            await update.message.reply_text("❌ Auction not found!")
            return
        
        if auction.creator_id != user.id and not bot_instance.is_admin(user.id):
            await update.message.reply_text("❌ Only auction host or admin can add players!")
            return
        
        if not update.message.reply_to_message:
            await update.message.reply_text(
                "❌ <b>Please reply to a message containing player names!</b>\n\n"
                "📝 <b>Accepted formats:</b>\n"
                "• @username @username2\n"
                "• Player Name 1, Player Name 2\n"
                "• Mix of both\n\n"
                "💡 <b>Example: Reply to message with '@user1 @user2 John Doe'</b>",
                parse_mode='HTML'
            )
            return
        
        replied_text = update.message.reply_to_message.text or update.message.reply_to_message.caption or ""
        
        import re
        
        usernames_with_at = re.findall(r'@([a-zA-Z][a-zA-Z0-9_]{4,31})', replied_text)
        
        text_without_usernames = re.sub(r'@[a-zA-Z][a-zA-Z0-9_]{4,31}', '', replied_text)
        
        plain_names = re.split(r'[,\n;]+', text_without_usernames)
        
        plain_names = [name.strip() for name in plain_names if name.strip() and len(name.strip()) > 2]
        
        all_players = []
        
        for username in usernames_with_at:
            all_players.append(f"@{username}")
        
        for name in plain_names:
            all_players.append(name)
        
        all_players = list(dict.fromkeys(all_players))
        
        if not all_players:
            await update.message.reply_text(
                "❌ <b>No valid player names found in the message!</b>\n\n"
                "📝 <b>Accepted formats:</b>\n"
                "• @username\n"
                "• Player Name\n"
                "• Multiple names separated by commas or new lines\n\n"
                "💡 <b>Example: '@player1, John Doe, @player2'</b>",
                parse_mode='HTML'
            )
        
        added_players = []
        failed_players = []
        
        for player_name in all_players:
            try:
                player_id = hash(player_name.lower()) % (10 ** 10)  # Use hash as ID
                
                if player_id in auction.approved_players:
                    failed_players.append(f"{player_name} (already added)")
                    continue
                
                new_player = ApprovedPlayer(
                    user_id=player_id,
                    name=player_name,  # Use the player name as-is (with @ if username, or plain name)
                    base_price=auction.base_price,
                    username=player_name.lstrip('@') if player_name.startswith('@') else None
                )
                
                auction.approved_players[player_id] = new_player
                
                if auction.status == "active":
                    auction.player_queue.append(new_player)
                
                added_players.append(player_name)
                
            except Exception as e:
                failed_players.append(f"{player_name} (error: {str(e)[:20]})")
        
        response_parts = []
        
        if added_players:
            response_parts.append(
                f"✅ <b>Successfully Added ({len(added_players)}):</b>\n" +
                "\n".join([f"  {player}" for player in added_players])
            )
        
        if failed_players:
            response_parts.append(
                f"❌ <b>Failed to Add ({len(failed_players)}):</b>\n" +
                "\n".join([f"  {player}" for player in failed_players])
            )
        
        if not added_players and not failed_players:
            response_parts.append("❌ No valid players found to add!")
        
        response = "\n\n".join(response_parts)
        response += f"\n\n📊 <b>Total Players:</b> {len(auction.approved_players)}"
        
        await update.message.reply_text(response, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Error in addpauc_command: {e}")
        await update.message.reply_text("❌ An error occurred!")

@check_banned
async def removepauc_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove players from auction by replying to message with usernames (Admin/Host only)"""
    try:
        user = update.effective_user
        
        if len(context.args) < 1:
            await update.message.reply_text(
                "❌ <b>Usage:</b> <code>/removepauc [auction_id]</code>\n\n"
                "📝 <b>Then reply to a message containing usernames to remove</b>\n"
                "💡 <b>Example:</b> <code>/removepauc 1</code>",
                parse_mode='HTML'
            )
            return
        
        try:
            auction_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("❌ Invalid auction ID!")
            return
        
        auction = bot_instance.get_approved_auction(auction_id)
        if not auction:
            await update.message.reply_text("❌ Auction not found!")
            return
        
        if auction.creator_id != user.id and not bot_instance.is_admin(user.id):
            await update.message.reply_text("❌ Only auction host or admin can remove players!")
            return
        
        if auction.status == "active":
            await update.message.reply_text("❌ Cannot remove players from active auction!")
            return
        
        if not update.message.reply_to_message:
            await update.message.reply_text(
                "❌ <b>Please reply to a message containing usernames!</b>\n\n"
                "📝 <b>The message should contain @username or username</b>\n"
                "💡 <b>Multiple usernames will be removed automatically</b>",
                parse_mode='HTML'
            )
            return
        
        replied_text = update.message.reply_to_message.text or update.message.reply_to_message.caption or ""
        
        import re
        usernames = re.findall(r'@?([a-zA-Z0-9_]{5,32})', replied_text)
        
        if not usernames:
            await update.message.reply_text(
                "❌ <b>No valid usernames found in the replied message!</b>\n\n"
                "📝 <b>Make sure the message contains @username or username</b>",
                parse_mode='HTML'
            )
            return
        
        removed_players = []
        failed_players = []
        
        for username in usernames:
            try:
                player = bot_instance.find_player_by_identifier(username)
                if not player:
                    failed_players.append(f"@{username} (not found in system)")
                    continue
                
                player_id = player['telegram_id']
                
                if player_id in auction.approved_players:
                    del auction.approved_players[player_id]
                    removed_players.append(f"@{username} ({player['display_name']})")
                elif player_id in auction.registered_players:
                    del auction.registered_players[player_id]
                    removed_players.append(f"@{username} ({player['display_name']})")
                else:
                    failed_players.append(f"@{username} (not in auction)")
                
            except Exception as e:
                failed_players.append(f"@{username} (error: {str(e)[:20]})")
        
        response_parts = []
        
        if removed_players:
            response_parts.append(
                f"🗑️ <b>Successfully Removed ({len(removed_players)}):</b>\n" +
                "\n".join([f"  {player}" for player in removed_players])
            )
        
        if failed_players:
            response_parts.append(
                f"❌ <b>Failed to Remove ({len(failed_players)}):</b>\n" +
                "\n".join([f"  {player}" for player in failed_players])
            )
        
        if not removed_players and not failed_players:
            response_parts.append("❌ No valid players found to remove!")
        
        response = "\n\n".join(response_parts)
        response += f"\n\n📊 <b>Remaining Players:</b> {len(auction.approved_players)}"
        
        await update.message.reply_text(response, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Error in removepauc_command: {e}")
        await update.message.reply_text("❌ An error occurred!")

@check_banned
async def clear_all_auctions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear all auction data (Admin only)"""
    try:
        user = update.effective_user
        
        if not bot_instance.is_admin(user.id):
            await update.message.reply_text("❌ Only admins can clear auction data!")
            return
        
        if len(context.args) == 0 or context.args[0] != 'CONFIRM':
            await update.message.reply_text(
                "⚠️ <b>WARNING: Clear All Auctions</b>\n\n"
                "🚨 This will permanently delete:\n"
                "• All pending auction proposals\n"
                "• All approved auctions\n"
                "• All auction registrations\n"
                "• All auction data\n\n"
                "❌ <b>This action cannot be undone!</b>\n\n"
                "To confirm, use:\n"
                "<code>/clear_all_auctions CONFIRM</code>",
                parse_mode='HTML'
            )
            return
        
        pending_count = len(bot_instance.auction_proposals)
        approved_count = len(bot_instance.approved_auctions)
        
        bot_instance.auction_proposals.clear()
        bot_instance.approved_auctions.clear()
        bot_instance.registration_states.clear()
        bot_instance.next_auction_id = 1
        
        success_message = (
            f"🗑️ <b>All Auction Data Cleared!</b>\n\n"
            f"✅ <b>Cleared Data:</b>\n"
            f"• Pending Proposals: {pending_count}\n"
            f"• Approved Auctions: {approved_count}\n"
            f"• Registration States: Cleared\n"
            f"• Next Auction ID: Reset to 1\n\n"
            f"💾 Data has been saved to file.\n"
            f"🔄 Fresh start ready!"
        )
        
        await update.message.reply_text(success_message, parse_mode='HTML')
        
        logger.info(f"Admin {user.id} ({user.first_name}) cleared all auction data")
        
    except Exception as e:
        logger.error(f"Error in clear_all_auctions_command: {e}")
        await update.message.reply_text("❌ An error occurred while clearing auction data!")

# ====================================
# ====================================

def process_current_player(auction: ApprovedAuction) -> str:
    """Process the current player being auctioned"""
    player_ids = list(auction.approved_players.keys())
    if not player_ids or auction.current_player_index >= len(player_ids):
        return "🏁 **AUCTION COMPLETED!** All players have been auctioned."
    
    current_player_id = player_ids[auction.current_player_index]
    current_player = auction.approved_players[current_player_id]
    
    auction.current_bid = auction.base_price
    auction.current_bidder = None
    auction.countdown_active = False
    
    return (
        f"🎯 **CURRENT PLAYER: {current_player.name}**\n"
        f"💰 Base Price: {auction.base_price} shards\n"
        f"📊 Player {auction.current_player_index + 1}/{len(player_ids)}\n\n"
        f"💸 Current Bid: {auction.current_bid} shards\n"
        f"👑 Bidder: None\n\n"
        f"Use `/bid {auction.auction_id} [amount]` to place bids!"
    )

def next_player(auction: ApprovedAuction) -> str:
    """Move to next player and return status"""
    player_ids = list(auction.approved_players.keys())
    if auction.current_player_index < len(player_ids):
        if auction.current_bidder:
            captain = next((c for c in auction.approved_captains if c.captain_id == auction.current_bidder), None)
            if captain:
                current_player_id = player_ids[auction.current_player_index]
                current_player = auction.approved_players[current_player_id]
                captain.team_players.append(current_player.name)
                captain.budget -= auction.current_bid
                
                result = (
                    f"🏆 **SOLD!** {current_player.name} to {captain.team_name}\n"
                    f"💰 Price: {auction.current_bid} shards\n"
                    f"💳 Remaining budget: {captain.budget} shards\n\n"
                )
            else:
                current_player_id = player_ids[auction.current_player_index]
                result = f"❌ Error: Bidder not found for {auction.approved_players[current_player_id].name}\n\n"
        else:
            current_player_id = player_ids[auction.current_player_index]
            result = f"📤 **UNSOLD** {auction.approved_players[current_player_id].name}\n\n"
        
        auction.current_player_index += 1
        
        if auction.current_player_index < len(auction.approved_players):
            result += process_current_player(auction)
        else:
            result += "🏁 **AUCTION COMPLETED!** All players have been auctioned."
        
        return result
    
    return "🏁 **AUCTION COMPLETED!** All players have been auctioned."

# ====================================
# ====================================

def format_amount(amount):
    """Format amount in crore format (e.g. 5 -> 5Cr, 100 -> 100Cr)"""
    if isinstance(amount, float) and amount.is_integer():
        amount = int(amount)
    
    return f"{amount}Cr"

def parse_bid_amount(bid_text):
    """Parse bid amount from text (e.g. '5' -> 5, '10.5' -> 10.5)"""
    if not bid_text or not isinstance(bid_text, str):
        return None
    
    bid_text = bid_text.strip()
    
    if not bid_text:
        return None
    
    if bid_text.lower().endswith('cr'):
        bid_text = bid_text[:-2].strip()
    
    if not bid_text:
        return None
    
    try:
        amount = float(bid_text)
        if 0.1 <= amount <= 1000:
            return amount
        else:
            return None
    except (ValueError, TypeError):
        return None

def get_auction_info_message(auction: ApprovedAuction) -> str:
    """Generate auction info message"""
    status_emoji = {
        'setup': '🔧',
        'captain_reg': '👑',
        'player_reg': '👥', 
        'ready': '⚡',
        'active': '🔥',
        'completed': '✅'
    }.get(auction.status, '❓')
    
    message = (
        f"📊 <b>Auction Information</b>\n\n"
        f"🏆 <b>Name:</b> {auction.name}\n"
        f"🆔 <b>ID:</b> {auction.id}\n"
        f"{status_emoji} <b>Status:</b> {auction.status.title()}\n"
        f"👤 <b>Creator:</b> {auction.creator_name}\n\n"
        f"💰 <b>Team Purse:</b> {format_amount(auction.purse)}\n"
        f"💎 <b>Base Price:</b> {format_amount(auction.base_price)}\n\n"
        f"🏏 <b>Teams ({len(auction.teams)}):</b>\n"
    )
    
    for team in auction.teams:
        captain_assigned = any(c.team_name == team for c in auction.approved_captains.values())
        emoji = "✅" if captain_assigned else "⏳"
        message += f"  {emoji} {team}\n"
    
    message += (
        f"\n📊 <b>Registration Stats:</b>\n"
        f"👑 <b>Captains:</b> {len(auction.approved_captains)}/{len(auction.teams)}\n"
        f"👥 <b>Players:</b> {len(auction.approved_players)}\n"
    )
    
    if auction.status == "active" and auction.current_player:
        message += (
            f"\n🎯 <b>Current Auction:</b>\n"
            f"👤 <b>Player:</b> {auction.current_player.name}\n"
            f"💰 <b>Highest Bid:</b> "
        )
        if auction.current_bids:
            highest = max(auction.current_bids.values(), key=lambda x: x['amount'])
            message += f"{highest['amount']}Cr ({highest['captain'].team_name})"
        else:
            message += f"{auction.base_price}Cr (Base Price)"
    
    return message

# ====================================
# ====================================

async def auction_callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route auction-related callbacks with timeout handling"""
    query = update.callback_query
    
    try:
        await query.answer()
        
        data = query.data
        
        if data.startswith(("approve_auction_", "reject_auction_")):
            await handle_admin_auction_approval(update, context)
        elif data.startswith("host_"):
            await handle_host_panel_callbacks(update, context)
        elif data.startswith(("approve_player_", "reject_player_")):
            await handle_player_approval_callbacks(update, context)
        elif data.startswith(("approve_captain_", "reject_captain_")):
            await handle_captain_approval_callbacks(update, context)
        elif data.startswith(("confirm_sale_", "continue_bid_")):
            await handle_auction_sale_callbacks(update, context)
        elif data.startswith("start_bidding_"):
            auction_id = int(data.split('_')[-1])
        else:
            await query.edit_message_text(
                f"❓ Unknown callback: {data}\n\nPlease try again or contact support.",
                parse_mode='HTML'
            )
            
    except telegram_error.BadRequest as e:
        if "query is too old" in str(e).lower() or "timeout" in str(e).lower():
            logger.warning(f"Callback query timeout: {e}")
            return
        else:
            logger.error(f"BadRequest in auction callback router: {e}")
            try:
                await query.edit_message_text("❌ Request error. Please try again.")
            except:
                pass
    except Exception as e:
        logger.error(f"Error in auction callback router: {e}")
        try:
            await query.edit_message_text("❌ An error occurred! Please try again.")
        except:
            pass

def register_commands(application):
    """Register all bot commands"""
    
    # ====================================
    # ====================================
    application.add_handler(CommandHandler("register", register_auction_command))
    application.add_handler(CommandHandler("hostpanel", hostpanel_command))
    application.add_handler(CommandHandler("regcap", registercaptain_command))
    application.add_handler(CommandHandler("regplay", registerplayer_command))
    application.add_handler(CommandHandler("rebid", rebid_command))
    application.add_handler(CommandHandler("sell", sell_command))
    application.add_handler(CommandHandler("myteam", myteam_command))
    application.add_handler(CommandHandler("purse", purse_command))
    application.add_handler(CommandHandler("transfercap", transfercap_command))
    application.add_handler(CommandHandler("pending", pending_auctions_command))
    application.add_handler(CommandHandler("clearauc", clear_all_auctions_command))
    application.add_handler(CommandHandler("auctionhelp", auctionhelp_command))
    application.add_handler(CommandHandler("auction", auction_command))
    application.add_handler(CommandHandler("status", auction_status_command))
    application.add_handler(CommandHandler("ping", ping_command))
    application.add_handler(CommandHandler("setgc", setgc_command))
    application.add_handler(CommandHandler("participants", participants_command))
    application.add_handler(CommandHandler("delauc", delete_auction_command))
    application.add_handler(CommandHandler("listauc", list_auctions_command))
    application.add_handler(CommandHandler("endauc", force_end_auction_command))
    application.add_handler(CommandHandler("addpauc", addpauc_command))
    application.add_handler(CommandHandler("removepauc", removepauc_command))
    application.add_handler(CommandHandler("unsold", unsold_command))
    application.add_handler(CommandHandler("addpt", addpt_command))
    application.add_handler(CommandHandler("pauseauc", pauseauc_command))
    
    # ====================================
    # ====================================
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("commands", commands_command))
    application.add_handler(CommandHandler("shards", shards_command))
    application.add_handler(CommandHandler("daily", daily_command))
    application.add_handler(CommandHandler("nightmare", nightmare_command))
    application.add_handler(CommandHandler("chase", chase_command))
    application.add_handler(CommandHandler("guess", guess_command))
    application.add_handler(CommandHandler("quit", quit_command))
    application.add_handler(CommandHandler("profile", profile_command))
    application.add_handler(CommandHandler("achievements", achievements_command))
    
    # ====================================
    # ====================================
    application.add_handler(CommandHandler("leaderboard", leaderboard_command))
    application.add_handler(CommandHandler("shardlb", shardlb_command))
    application.add_handler(CommandHandler("dailylb", dailylb_command))
    application.add_handler(CommandHandler("chasestats", chase_stats_command))
    application.add_handler(CommandHandler("guessstats", guess_stats_command))
    application.add_handler(CommandHandler("guesslb", guess_leaderboard_command))
    
    # ====================================
    # ====================================
    application.add_handler(CommandHandler("dailyguess", daily_guess_command))
    application.add_handler(CommandHandler("goat", goat_command))
    application.add_handler(CommandHandler("myroast", my_roast_command))
    
    # ====================================
    # ====================================
    application.add_handler(CommandHandler("giveshards", give_shards_command))
    application.add_handler(CommandHandler("removeshards", remove_shards_command))
    application.add_handler(CommandHandler("addadmin", add_admin_command))
    application.add_handler(CommandHandler("removeadmin", remove_admin_command))
    application.add_handler(CommandHandler("listadmins", list_admins_command))
    application.add_handler(CommandHandler("banuser", banuser_command))
    application.add_handler(CommandHandler("unbanuser", unbanuser_command))
    application.add_handler(CommandHandler("finduser", finduser_command))
    application.add_handler(CommandHandler("broadcast", broadcast_command))
    application.add_handler(CommandHandler("bulkaward", bulk_award_command))
    application.add_handler(CommandHandler("resetplayer", reset_player_command))
    application.add_handler(CommandHandler("addach", add_achievement_command))
    application.add_handler(CommandHandler("remach", remove_achievement_command))
    application.add_handler(CommandHandler("settitle", set_title_command))
    application.add_handler(CommandHandler("removetitle", remove_title_command))
    application.add_handler(CommandHandler("confach", confirm_achievement_command))
    application.add_handler(CommandHandler("pending_conf", list_pending_confirmations_command))
    application.add_handler(CommandHandler("ddrlb", distribute_daily_rewards_command))
    application.add_handler(CommandHandler("resetdlb", reset_daily_leaderboard_command))
    application.add_handler(CommandHandler("dlbstats", daily_leaderboard_stats_command))
    application.add_handler(CommandHandler("cleanupchase", cleanup_chase_command))
    application.add_handler(CommandHandler("cleanupguess", cleanup_guess_command))
    application.add_handler(CommandHandler("adminstatus", admin_status_command))
    application.add_handler(CommandHandler("restart", restart_command))
    application.add_handler(CommandHandler("backup", backup_command))
    application.add_handler(CommandHandler("cleancache", cleancache_command))
    application.add_handler(CommandHandler("cstatus", cache_status_command))
    application.add_handler(CommandHandler("tstatus", thread_safety_status_command))
    application.add_handler(CommandHandler("transactions", transactions_command))
    application.add_handler(CommandHandler("resetall", reset_all_command))
    application.add_handler(CommandHandler("listplay", list_players_command))
    application.add_handler(CommandHandler("botstatus", bot_status_command))
    application.add_handler(CommandHandler("update", update_command))
    application.add_handler(CommandHandler("emojis", emojis_command))
    
    # ====================================
    # ====================================
    application.add_handler(CallbackQueryHandler(nightmare_callback, pattern="^nightmare_"))
    application.add_handler(CallbackQueryHandler(dailylb_callback, pattern="^dailylb_"))
    application.add_handler(CallbackQueryHandler(update_callback, pattern="^update_"))
    application.add_handler(CallbackQueryHandler(help_callback, pattern="^help_"))
    application.add_handler(CallbackQueryHandler(commands_callback, pattern="^cmd_"))
    application.add_handler(CallbackQueryHandler(chase_callback, pattern="^chase:"))
    application.add_handler(CallbackQueryHandler(broadcast_callback, pattern="^broadcast_"))
    application.add_handler(CallbackQueryHandler(guess_callback, pattern="^guess_"))
    application.add_handler(CallbackQueryHandler(auction_callback_router, pattern="^(approve_auction_|reject_auction_|host_|approve_player_|reject_player_|approve_captain_|reject_captain_|confirm_sale_|continue_bid_|start_bidding_)"))
    
    # ====================================
    # ====================================
    from telegram.ext import MessageHandler, filters
    
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_auction_registration), group=1)
    
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_manual_auction_input), group=2)
    
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_guess_input), group=3)
    
    # ====================================
    # ====================================
    application.add_error_handler(error_handler)

TOKEN = os.getenv('BOT_TOKEN')
if not TOKEN:
    raise ValueError("BOT_TOKEN environment variable is required")

PORT = int(os.environ.get("PORT", 8080))
BIND_ADDRESS = str(os.environ.get("WEB_SERVER_BIND_ADDRESS", "0.0.0.0"))

try:
    from web import web_server, keep_alive
    WEB_SERVER = True
except ImportError:
    WEB_SERVER = False
    logger.warning("Web server module not found. Running without keep-alive.")

async def start_bot():
    """Start the bot with web server"""
    try:
        application = Application.builder().token(TOKEN).build()
        
        register_commands(application)
        
        await application.initialize()
        await application.start()
        
        await application.updater.start_polling()
        
        logger.info("✅ Bot started successfully!")
        
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"Bot error: {e}")
        raise
    finally:
        if application.updater.running:
            await application.updater.stop()
        await application.stop()
        await application.shutdown()

async def main():
    """Main function with web server integration"""
    try:
        if WEB_SERVER:
            app = web_server()
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, BIND_ADDRESS, PORT)
            await site.start()
            logger.info(f"🌐 Web server started on {BIND_ADDRESS}:{PORT}")
            
            asyncio.create_task(keep_alive())
            logger.info("⏰ Keep-alive pinger started")
        
        await start_bot()
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        
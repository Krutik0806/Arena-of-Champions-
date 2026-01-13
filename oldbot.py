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
from typing import Optional, Dict, List, Tuple, Union
from contextlib import contextmanager


# ====================================
# UTILITY FUNCTIONS
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
    
    # Split by lines to preserve formatting
    lines = text.split('\n')
    
    for line in lines:
        # If adding this line would exceed max length
        if len(current_page + line + '\n') > max_length:
            if current_page:
                pages.append(current_page.rstrip())
                current_page = line + '\n'
            else:
                # Line itself is too long, force break
                while len(line) > max_length:
                    pages.append(line[:max_length])
                    line = line[max_length:]
                current_page = line + '\n'
        else:
            current_page += line + '\n'
    
    # Add remaining content
    if current_page:
        pages.append(current_page.rstrip())
    
    return pages

from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup, error as telegram_error
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
import psycopg2
from psycopg2 import sql, pool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def send_paginated_message(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, parse_mode: str = 'HTML', max_length: int = 4000):
    """Send a long message as multiple paginated messages"""
    pages = paginate_text(text, max_length)
    
    for i, page in enumerate(pages):
        if len(pages) > 1:
            # Add page indicator
            page_indicator = f"\n\n📄 <i>Page {i+1} of {len(pages)}</i>"
            if len(page + page_indicator) <= max_length:
                page += page_indicator
        
        await safe_send(update.message.reply_text, page, parse_mode=parse_mode)
        
        # Small delay between messages to avoid rate limits
        if i < len(pages) - 1:
            await asyncio.sleep(0.5)

# Input validation utilities
def validate_username(username: str) -> bool:
    """Validate username format according to Telegram standards.
    
    Args:
        username (str): The username to validate
        
    Returns:
        bool: True if username is valid, False otherwise
        
    Note:
        Allows alphanumeric characters and underscores, 1-32 chars length
    """
    if not username:
        return False
    # Allow alphanumeric, underscore, and length between 5-32 chars
    return bool(re.match(r'^[a-zA-Z0-9_]{1,32}$', username))

def validate_display_name(name: str) -> bool:
    """Validate display name for reasonable length and content.
    
    Args:
        name (str): The display name to validate
        
    Returns:
        bool: True if display name is valid, False otherwise
        
    Note:
        Ensures non-empty name with reasonable length limit (64 chars)
    """
    if not name or len(name.strip()) < 1:
        return False
    # Allow reasonable length and basic characters
    return len(name.strip()) <= 64

def validate_achievement_name(name: str) -> bool:
    """Validate achievement name"""
    if not name:
        return False
    # Allow alphanumeric, spaces, basic punctuation
    return bool(re.match(r'^[a-zA-Z0-9\s\-_\.]{1,100}$', name))

# Security and validation utilities
def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Sanitize user input to prevent security issues and ensure data integrity.
    
    Args:
        text (str): The input text to sanitize
        max_length (int, optional): Maximum allowed length. Defaults to 1000.
        
    Returns:
        str: Sanitized and HTML-escaped text
        
    Note:
        Removes control characters, limits length, and HTML-escapes for safety
    """
    if not text:
        return ""
    
    # Remove null bytes and control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Limit length
    text = text[:max_length]
    
    # HTML escape for safety
    text = html.escape(text)
    
    return text.strip()

def validate_telegram_id(telegram_id: Union[int, str]) -> bool:
    """Validate telegram ID format"""
    try:
        tid = int(telegram_id)
        return 0 < tid < 10**12  # Reasonable range for Telegram IDs
    except (ValueError, TypeError):
        return False

def validate_chat_id(chat_id: Union[int, str]) -> bool:
    """Validate chat ID format"""
    try:
        cid = int(chat_id)
        return -10**15 < cid < 10**15  # Range for Telegram chat IDs
    except (ValueError, TypeError):
        return False

@contextmanager
def safe_db_transaction(connection):
    """Context manager for safe database transactions"""
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise

def split_message_safely(message: str, max_length: int = 4090) -> List[str]:
    """
    Split long messages at safe boundaries to prevent HTML parsing errors.
    Ensures HTML tags and emojis aren't broken mid-way.
    """
    if len(message) <= max_length:
        return [message]
    
    messages = []
    current_msg = ""
    
    # Split by lines first to preserve structure
    lines = message.split('\n')
    
    for line in lines:
        # Check if adding this line would exceed limit
        test_msg = current_msg + ('\n' if current_msg else '') + line
        
        if len(test_msg) > max_length:
            # If current message has content, save it
            if current_msg:
                # Add continuation marker
                current_msg += "\n\n<i>📄 Continued...</i>"
                messages.append(current_msg)
                current_msg = ""
            
            # If single line is too long, split it carefully
            if len(line) > max_length:
                # Split at word boundaries, preserving HTML tags
                words = line.split(' ')
                for word in words:
                    test_word = current_msg + (' ' if current_msg else '') + word
                    
                    if len(test_word) > max_length - 50:  # Leave buffer
                        if current_msg:
                            current_msg += "\n\n<i>📄 Continued...</i>"
                            messages.append(current_msg)
                            current_msg = word
                        else:
                            # Even single word too long, force split (rare case)
                            current_msg = word[:max_length-50] + "..."
                            messages.append(current_msg)
                            current_msg = "..." + word[max_length-50:]
                    else:
                        current_msg = test_word
            else:
                current_msg = line
        else:
            current_msg = test_msg
    
    # Add final message if there's content
    if current_msg:
        messages.append(current_msg)
    
    return messages

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# HTML-safe helper function for user content
def H(s: str | None) -> str:
    """HTML-escape helper for user-supplied content"""
    return html.escape(s or "")

# Rate-limiting wrapper for Telegram API calls
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
    
    # Final attempt
    return await call(*args, **kwargs)

# Central error logging helper
def log_exception(context: str, e: Exception, user_id: int = None):
    """Centralized exception logging with context"""
    error_msg = f"[{context}] Error"
    if user_id:
        error_msg += f" for user {user_id}"
    error_msg += f": {type(e).__name__}: {e}"
    logger.error(error_msg, exc_info=True)

# User-friendly error messages
ERROR_MESSAGES = {
    'database': "❌ <b>Database temporarily unavailable!</b>\n\nPlease try again in a moment.",
    'permission': "❌ <b>ACCESS DENIED!</b>\n\n🛡️ You don't have permission for this action.",
    'invalid_input': "❌ <b>Invalid input!</b>\n\nPlease check your command and try again.", 
    'rate_limit': "⏳ <b>Please slow down!</b>\n\nToo many requests. Try again in a moment.",
    'generic': "❌ <b>Something went wrong!</b>\n\nPlease try again later."
}

class BotWatchdog:
    def __init__(self, url: str, interval: int = 60):
        self.url = url
        self.interval = interval
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _run(self):
        """Enhanced watchdog with error recovery and restart capability"""
        consecutive_failures = 0
        max_failures = 3
        
        while self.running:
            try:
                response = requests.get(self.url, timeout=10)
                if response.status_code != 200:
                    consecutive_failures += 1
                    logger.error(f"Health check failed with status code {response.status_code} (failure {consecutive_failures}/{max_failures})")
                    
                    if consecutive_failures >= max_failures:
                        logger.critical("Health check failed multiple times. Triggering restart.")
                        os._exit(1)
                else:
                    # Reset failure counter on success
                    if consecutive_failures > 0:
                        logger.info("Health check recovered")
                        consecutive_failures = 0
                        
            except requests.RequestException as e:
                consecutive_failures += 1
                logger.error(f"Health check request failed: {e} (failure {consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    logger.critical("Health check failed multiple times. Triggering restart.")
                    os._exit(1)
                    
            except Exception as e:
                # Catch any other unexpected errors
                consecutive_failures += 1
                logger.error(f"Unexpected error in watchdog: {e} (failure {consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    logger.critical("Watchdog encountered critical errors. Triggering restart.")
                    os._exit(1)
            
            # Progressive backoff on failures
            if consecutive_failures > 0:
                sleep_time = self.interval * (1 + consecutive_failures * 0.5)
                time.sleep(min(sleep_time, 300))  # Cap at 5 minutes
            else:
                time.sleep(self.interval)

class SPLAchievementBot:
    def __init__(self):
        # Initialize roast cache
        self.roast_cache = {
            'lines': [],
            'last_updated': 0,
            'cache_duration': 300,  # 5 minutes cache
            'usage_counts': {}
        }
        
        # Initialize leaderboard cache
        self.leaderboard_cache = {
            'data': [],
            'last_updated': 0,
            'cache_duration': 60  # 1 minute cache for leaderboard
        }
        
        # Initialize profile cache
        self.profile_cache = {
            'data': {},  # {user_id: profile_data}
            'last_updated': {},  # {user_id: timestamp}
            'cache_duration': 120  # 2 minutes cache for profiles
        }
        
        # Initialize GOAT cache
        self.goat_cache = {
            'data': None,
            'date': None,
            'last_updated': 0,
            'cache_duration': 3600  # 1 hour cache for GOAT
        }
        
        # Bot configuration
        self.bot_token = os.getenv('BOT_TOKEN')
        self.db_url = os.getenv('DATABASE_URL')
        self.super_admin_id = int(os.getenv('SUPER_ADMIN_ID', '0'))  # Creator/Super Admin
        self.admin_ids = [int(x.strip()) for x in os.getenv('ADMIN_IDS', '').split(',') if x.strip()]
        
        # Logs configuration
        self.logs_bot_token = "8209000379:AAF0yBBwp_RdlyWXNKlfKdE9-6r2cD2PS88"
        self.logs_chat_id = -1002957624302
        self.logs_enabled = True
        
        # Initialize connection pool
        try:
            self.db_pool = psycopg2.pool.ThreadedConnectionPool(
                1, 10,  # Much smaller pool to avoid connection issues
                self.db_url,
                # Additional connection parameters for stability
                connect_timeout=10,
                application_name="SPL_Achievement_Bot"
            )
            logger.info("Database connection pool initialized")
            
            # Initialize required database tables
            self._initialize_database_tables()
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            self.db_pool = None
        
        # Achievement emojis mapping - 4 main categories
        self.achievement_emojis = {
            'winner': '🏆',
            'orange cap': '🟧',
            'purple cap': '🟪', 
            'mvp': '🏅'
        }
        
        # Initialize guess game sessions storage
        self.guess_games = {}  # {user_id: game_state}
        
        # Guess game configuration
        self.guess_difficulties = {
            'beginner': {'emoji': '🟢', 'range': (1, 20), 'attempts': 6, 'time_limit': 30, 'multiplier': 1.0},
            'easy': {'emoji': '🔵', 'range': (1, 50), 'attempts': 8, 'time_limit': 60, 'multiplier': 1.2},
            'medium': {'emoji': '🟡', 'range': (1, 100), 'attempts': 7, 'time_limit': 60, 'multiplier': 1.5},
            'hard': {'emoji': '🟠', 'range': (1, 200), 'attempts': 8, 'time_limit': 90, 'multiplier': 2.0},
            'expert': {'emoji': '🔴', 'range': (1, 500), 'attempts': 10, 'time_limit': 90, 'multiplier': 3.0}
        }
        
    def _initialize_database_tables(self):
        """Initialize required database tables that might be missing"""
        try:
            conn = self.get_db_connection()
            if conn:
                cursor = conn.cursor()
                
                # Create nightmare_games table if it doesn't exist
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
                
                # Add result column if it doesn't exist (for existing tables)
                try:
                    cursor.execute("ALTER TABLE nightmare_games ADD COLUMN IF NOT EXISTS result VARCHAR(50)")
                except Exception:
                    pass  # Column might already exist
                
                # Create indexes for nightmare games
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_nightmare_player_telegram_id ON nightmare_games(player_telegram_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_nightmare_completed ON nightmare_games(is_completed)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_nightmare_created_at ON nightmare_games(created_at)")
                
                # Create comprehensive daily leaderboard system
                # Main daily leaderboard entries table
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
                
                # Create daily leaderboard rewards tracking table
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
                
                # Create daily leaderboard status table (for tracking when rewards were distributed)
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
                
                # Create indexes for daily leaderboard system
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_lb_date ON daily_leaderboard_entries(leaderboard_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_lb_player ON daily_leaderboard_entries(player_telegram_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_lb_game_type ON daily_leaderboard_entries(game_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_lb_chase_score ON daily_leaderboard_entries(chase_total_score DESC)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_lb_guess_score ON daily_leaderboard_entries(guess_total_score DESC)")
                
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_rewards_date ON daily_leaderboard_rewards(reward_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_rewards_player ON daily_leaderboard_rewards(player_telegram_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_rewards_game_type ON daily_leaderboard_rewards(game_type)")
                
                # Create the update_daily_leaderboard_entry function
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
                                
                        ELSIF p_game_type = 'nightmare' THEN
                            INSERT INTO daily_leaderboard_entries (
                                player_id, player_telegram_id, player_name, leaderboard_date, game_type,
                                nightmare_games_played, nightmare_games_won
                            ) VALUES (
                                COALESCE(player_record.id, NULL), p_telegram_id, p_player_name, today_date, p_game_type,
                                1, CASE WHEN p_won THEN 1 ELSE 0 END
                            )
                            ON CONFLICT (player_telegram_id, leaderboard_date, game_type) DO UPDATE SET
                                nightmare_games_played = daily_leaderboard_entries.nightmare_games_played + 1,
                                nightmare_games_won = daily_leaderboard_entries.nightmare_games_won + CASE WHEN p_won THEN 1 ELSE 0 END,
                                updated_at = CURRENT_TIMESTAMP;
                        END IF;
                        
                        RETURN TRUE;
                    END;
                    $$ LANGUAGE plpgsql
                """)
                
                # Create views for daily leaderboards (these create the virtual tables the bot expects)
                # Drop existing views first to avoid column name conflicts
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
                
                conn.commit()
                cursor.close()
                self.return_db_connection(conn)
                logger.info("Database tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database tables: {e}")
        
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
        
    def get_db_connection(self):
        """Create database connection from pool with better error handling"""
        try:
            if self.db_pool:
                # Log pool status for debugging
                try:
                    pool_size = self.db_pool.maxconn - len(self.db_pool._pool)
                    if pool_size > 35:  # Warn if pool usage is high (>87% of 40)
                        logger.warning(f"High pool usage: {pool_size}/{self.db_pool.maxconn} connections in use")
                        # Schedule admin log for high pool usage (thread-safe)
                        try:
                            import threading
                            def send_log_async():
                                import asyncio
                                try:
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    loop.run_until_complete(self.send_admin_log(
                                        'system_event',
                                        f"High database pool usage: {pool_size}/{self.db_pool.maxconn} connections in use ({pool_size/self.db_pool.maxconn*100:.1f}%)",
                                        None,
                                        "System"
                                    ))
                                    loop.close()
                                except Exception as e:
                                    logger.debug(f"Could not send pool usage log: {e}")
                            
                            threading.Thread(target=send_log_async, daemon=True).start()
                        except Exception as e:
                            logger.debug(f"Could not schedule pool usage log: {e}")
                except (AttributeError, TypeError) as e:
                    logger.debug(f"Could not retrieve pool status: {e}")
                
                # Try to get connection with timeout
                conn = self.db_pool.getconn()
                if conn:
                    # Test the connection
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()
                    return conn
                else:
                    logger.error("Database connection pool exhausted: connection pool exhausted")
                    # Schedule admin log for pool exhaustion (thread-safe)
                    try:
                        import threading
                        def send_exhaustion_log():
                            import asyncio
                            try:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                loop.run_until_complete(self.send_admin_log(
                                    'error',
                                    "Database connection pool EXHAUSTED - all connections in use",
                                    None,
                                    "System"
                                ))
                                loop.close()
                            except Exception as e:
                                logger.debug(f"Could not send pool exhaustion log: {e}")
                        
                        threading.Thread(target=send_exhaustion_log, daemon=True).start()
                    except Exception as e:
                        logger.debug(f"Could not schedule pool exhaustion log: {e}")
                    # Try to reset the pool once
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
                # Fallback to direct connection if pool failed
                return psycopg2.connect(
                    self.db_url,
                    connect_timeout=10,
                    application_name="SPL_Bot_Fallback"
                )
        except psycopg2.pool.PoolError as e:
            logger.error(f"Database connection pool exhausted: {e}")
            # Try to reset the pool and retry once
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
            
            # Reinitialize the pool
            self.db_pool = psycopg2.pool.ThreadedConnectionPool(
                3, 40,
                self.db_url,
                connect_timeout=10,
                application_name="SPL_Achievement_Bot_Reset"
            )
            logger.info("Connection pool reset successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reset connection pool: {e}")
            return False
    
    def get_connection_pool_status(self):
        """Get current connection pool status for monitoring"""
        if not self.db_pool:
            return "Pool not initialized"
        try:
            # Get pool stats (if available)
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
        
        # Skip complex initialization for now - tables should be created manually
        logger.info("Using simplified database initialization...")
        logger.info("Please run complete_database_schema.sql manually for full setup")
        
        # Just test the connection and create basic tables
        conn = self.get_db_connection()
        if not conn:
            logger.error("Failed to get database connection")
            return False
            
        try:
            logger.info("Testing database connection...")
            cursor = conn.cursor()
            
            # Simple connection test
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            logger.info(f"Database connection successful: {version[:50]}...")
            
            # Create only the most essential table if it doesn't exist
            logger.info("Ensuring players table exists...")
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
            
            conn.commit()
            logger.info("Essential tables verified successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            try:
                conn.rollback()
            except:
                pass
            return False
        finally:
            self.return_db_connection(conn)
    
    def is_admin(self, user_id: int) -> bool:
        """Check if user is admin (includes super admin and database admins)"""
        if user_id == self.super_admin_id:
            return True
        if user_id in self.admin_ids:
            return True
        
        # Check database admins
        with self.get_db_connection_ctx() as conn:
            if not conn:
                return False
            
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT telegram_id FROM admins WHERE telegram_id = %s", (user_id,))
                result = cursor.fetchone() is not None
                cursor.close()
                return result
            except Exception as e:
                logger.error(f"Error checking admin status: {e}")
                return False
    
    def is_super_admin(self, user_id: int) -> bool:
        """Check if user is super admin (creator)"""
        return user_id == self.super_admin_id

    def ban_user(self, player_id: int, reason: str, banned_by: int) -> bool:
        """Ban a user from the bot"""
        conn = self.get_db_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Add ban record to database
            cursor.execute("""
                INSERT INTO user_bans (player_id, reason, banned_by, banned_at) 
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (player_id) DO UPDATE SET
                    reason = EXCLUDED.reason,
                    banned_by = EXCLUDED.banned_by,
                    banned_at = NOW(),
                    unbanned_at = NULL
            """, (player_id, reason, banned_by))
            
            logger.info(f"User {player_id} banned by {banned_by} for: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error banning user {player_id}: {e}")
            return False
        finally:
            self.return_db_connection(conn)
    
    def unban_user(self, player_id: int, unbanned_by: int) -> bool:
        """Unban a user"""
        conn = self.get_db_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Update ban record
            cursor.execute("""
                UPDATE user_bans SET unbanned_at = NOW(), unbanned_by = %s 
                WHERE player_id = %s AND unbanned_at IS NULL
            """, (unbanned_by, player_id))
            
            if cursor.rowcount > 0:
                logger.info(f"User {player_id} unbanned by {unbanned_by}")
                return True
            else:
                logger.warning(f"Attempted to unban user {player_id} but no active ban found")
                return False
                
        except Exception as e:
            logger.error(f"Error unbanning user {player_id}: {e}")
            return False
        finally:
            self.return_db_connection(conn)
    
    def is_banned(self, user_id: int) -> bool:
        """Check if user is currently banned"""
        conn = self.get_db_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 1 FROM user_bans 
                WHERE player_id = %s AND unbanned_at IS NULL
            """, (user_id,))
            
            return cursor.fetchone() is not None
            
        except Exception as e:
            logger.error(f"Error checking ban status for user {user_id}: {e}")
            return False
        finally:
            self.return_db_connection(conn)

    def track_chat(self, chat_id: int, chat_type: str, title: str = None, username: str = None) -> bool:
        """Track chat information for broadcast purposes"""
        conn = self.get_db_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Update or insert chat information
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
            
            # Check if identifier is a user ID (numeric)
            if identifier.isdigit():
                cursor.execute(
                    "SELECT * FROM players WHERE telegram_id = %s",
                    (int(identifier),)
                )
            # Check if identifier is a username (starts with @)
            elif identifier.startswith('@'):
                username = identifier[1:]  # Remove @
                cursor.execute(
                    "SELECT * FROM players WHERE username ILIKE %s",
                    (username,)
                )
            # Otherwise, search by display name
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
        
        # Validate inputs
        if not validate_display_name(display_name):
            logger.warning(f"Invalid display name: {display_name}")
            return False, False
        
        if username and not validate_username(username):
            logger.warning(f"Invalid username: {username}")
            username = None  # Set to None if invalid
        
        with self.get_db_cursor() as cursor_result:
            if cursor_result is None:
                return False, False
            cursor, conn = cursor_result
            
            # Check if user already exists
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
            
            # Get player name for verification
            cursor.execute("SELECT display_name FROM players WHERE id = %s", (player_id,))
            player_data = cursor.fetchone()
            if not player_data:
                # Player doesn't exist - return False to indicate user not registered
                logger.info(f"Player {player_id} not found - user not registered")
                return False
            
            player_name = player_data[0]
            
            # Check if achievement already exists for this player
            cursor.execute(
                "SELECT count FROM achievements WHERE player_id = %s AND achievement_name = %s",
                (player_id, achievement.lower())
            )
            
            existing = cursor.fetchone()
            
            is_new_achievement = not existing
            
            if existing:
                # Increment count
                cursor.execute(
                    "UPDATE achievements SET count = count + 1, updated_at = CURRENT_TIMESTAMP WHERE player_id = %s AND achievement_name = %s",
                    (player_id, achievement.lower())
                )
            else:
                # Create new achievement
                cursor.execute(
                    "INSERT INTO achievements (player_id, achievement_name) VALUES (%s, %s)",
                    (player_id, achievement.lower())
                )
            
            conn.commit()
            
            # Award shards for new achievements
            if is_new_achievement:
                try:
                    # Get player's telegram_id for shard award
                    cursor.execute("SELECT telegram_id FROM players WHERE id = %s", (player_id,))
                    telegram_result = cursor.fetchone()
                    
                    if telegram_result:
                        telegram_id = telegram_result[0]
                        shard_reward = 50  # Reduced from 100 - Base reward for achievements
                        
                        # Bonus shards for special achievements - reduced
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
                            logger.info(f"Awarded {shard_reward} shards for new achievement '{achievement}' to player {player_id}")
                        
                except Exception as e:
                    logger.error(f"Error awarding achievement shards: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding achievement: {e}")
            conn.rollback()
            return False
        finally:
            self.return_db_connection(conn)

    def add_or_increment_achievement(self, player_id: int, achievement: str, inc: int = 1) -> bool:
        """Safely add or increment achievement using upsert (requires UNIQUE constraint)"""
        conn = self.get_db_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Verify player exists
            cursor.execute("SELECT id FROM players WHERE id = %s", (player_id,))
            if not cursor.fetchone():
                logger.info(f"Player {player_id} not found - user not registered")
                return False
            
            # Upsert achievement (requires UNIQUE constraint on player_id, achievement_name)
            cursor.execute("""
                INSERT INTO achievements (player_id, achievement_name, count)
                VALUES (%s, %s, %s)
                ON CONFLICT (player_id, achievement_name)
                DO UPDATE SET count = achievements.count + EXCLUDED.count,
                              updated_at = CURRENT_TIMESTAMP
            """, (player_id, achievement.lower(), inc))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error in add_or_increment_achievement: {e}")
            conn.rollback()
            return False
        finally:
            self.return_db_connection(conn)

    def remove_achievement(self, player_id: int, achievement: str) -> bool:
        """Remove one instance of achievement from player"""
        conn = self.get_db_connection()
        if not conn:
            return False
            
        try:
            cursor = conn.cursor()
            
            # Check current count
            cursor.execute(
                "SELECT count FROM achievements WHERE player_id = %s AND achievement_name = %s",
                (player_id, achievement.lower())
            )
            
            result = cursor.fetchone()
            
            if not result:
                return False  # Achievement doesn't exist
            
            current_count = result[0]
            
            if current_count > 1:
                # Decrement count
                cursor.execute(
                    "UPDATE achievements SET count = count - 1, updated_at = CURRENT_TIMESTAMP WHERE player_id = %s AND achievement_name = %s",
                    (player_id, achievement.lower())
                )
            else:
                # Remove achievement completely
                cursor.execute(
                    "DELETE FROM achievements WHERE player_id = %s AND achievement_name = %s",
                    (player_id, achievement.lower())
                )
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error removing achievement: {e}")
            conn.rollback()
            return False
        finally:
            self.return_db_connection(conn)
    
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
    
    def get_player_achievements(self, player_id: int) -> List[Tuple[str, int]]:
        """Get all achievements for a player"""
        conn = self.get_db_connection()
        if not conn:
            return []
            
        try:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT achievement_name, count FROM achievements WHERE player_id = %s ORDER BY achievement_name",
                (player_id,)
            )
            
            return cursor.fetchall()
            
        except Exception as e:
            logger.error(f"Error getting achievements: {e}")
            return []
        finally:
            self.return_db_connection(conn)
    
    def get_achievement_emoji(self, achievement: str) -> str:
        """Get emojis for achievement - supports multiple keywords and returns all matching emojis"""
        achievement_lower = achievement.lower().strip()
        emojis = []
        
        # Check for each keyword and collect all matching emojis
        if 'winner' in achievement_lower:
            emojis.append('🏆')
        
        if 'orange' in achievement_lower:
            emojis.append('🟧')
            
        if 'purple' in achievement_lower:
            emojis.append('🟪')
            
        if 'mvp' in achievement_lower:
            emojis.append('🏅')
        
        # Return space-separated emojis if any found, empty string if none
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
            
            # Get player_id if player is registered
            player_id = None
            cursor.execute("SELECT id FROM players WHERE telegram_id = %s", (telegram_id,))
            player_data = cursor.fetchone()
            if player_data:
                player_id = player_data[0]
                logger.info(f"Found registered player ID: {player_id}")
            else:
                logger.info(f"Player {telegram_id} not registered, using telegram_id")
            
            # Insert chase game record
            cursor.execute("""
                INSERT INTO chase_games (
                    player_id, telegram_id, player_name, chat_id, 
                    final_score, max_level, game_outcome, game_duration, completed_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """, (player_id, telegram_id, player_name, chat_id, final_score, max_level, game_outcome, game_duration))
            
            conn.commit()
            logger.info(f"Successfully recorded chase game for player {telegram_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording chase game: {e}")
            conn.rollback()
            return False
        finally:
            self.return_db_connection(conn)

    def get_chase_game_stats(self) -> dict:
        """Get overall chase game statistics"""
        conn = self.get_db_connection()
        if not conn:
            return {'total_games': 0, 'total_players': 0, 'avg_score': 0, 'high_score': 0}
        
        try:
            cursor = conn.cursor()
            
            # Total games played
            cursor.execute("SELECT COUNT(*) FROM chase_games")
            total_games = cursor.fetchone()[0] or 0
            
            # Total unique players
            cursor.execute("SELECT COUNT(DISTINCT telegram_id) FROM chase_games")
            total_players = cursor.fetchone()[0] or 0
            
            # Average score
            cursor.execute("SELECT AVG(final_score) FROM chase_games WHERE final_score > 0")
            avg_score_result = cursor.fetchone()
            avg_score = round(avg_score_result[0]) if avg_score_result[0] else 0
            
            # High score
            cursor.execute("SELECT MAX(final_score) FROM chase_games")
            high_score = cursor.fetchone()[0] or 0
            
            # Games in last 24 hours
            cursor.execute("""
                SELECT COUNT(*) FROM chase_games 
                WHERE created_at >= NOW() - INTERVAL '24 hours'
            """)
            games_24h = cursor.fetchone()[0] or 0
            
            # Games in last 7 days
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
            
        except Exception as e:
            logger.error(f"Error getting chase game stats: {e}")
            return {'total_games': 0, 'total_players': 0, 'avg_score': 0, 'high_score': 0, 'games_24h': 0, 'games_7d': 0}
        finally:
            self.return_db_connection(conn)

    def get_chase_leaderboard(self, limit: int = 10) -> List[Dict]:
        """
        Get chase game leaderboard based on best performance per player.
        Ranking: Highest Level > Most Runs at that Level > Fewer Balls Used
        """
        # Check cache first
        current_time = time.time()
        if (current_time - self.leaderboard_cache['last_updated'] < self.leaderboard_cache['cache_duration'] 
            and len(self.leaderboard_cache['data']) > 0):
            return self.leaderboard_cache['data'][:limit]
        
        with self.get_db_cursor() as cursor_result:
            if cursor_result is None:
                return []
            cursor, conn = cursor_result
            
            # Get top 10 best performances across all games with strict sorting
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
                # Ensure all values are properly validated
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
            
            # Update cache
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
                
                # Get player's telegram_id first
                cursor.execute("SELECT telegram_id FROM players WHERE id = %s", (player_id,))
                player_result = cursor.fetchone()
                if not player_result:
                    logger.warning(f"Player with ID {player_id} not found")
                    return default_stats
                
                telegram_id = player_result[0]
                
                # Total games played
                cursor.execute("SELECT COUNT(*) FROM chase_games WHERE telegram_id = %s", (telegram_id,))
                games_played = cursor.fetchone()[0] or 0
                
                # Highest level reached
                cursor.execute("SELECT MAX(max_level) FROM chase_games WHERE telegram_id = %s", (telegram_id,))
                highest_level = cursor.fetchone()[0] or 0
                
                # Highest score
                cursor.execute("SELECT MAX(final_score) FROM chase_games WHERE telegram_id = %s", (telegram_id,))
                highest_score = cursor.fetchone()[0] or 0
                
                # Best Strike Rate (calculated as score per level)
                cursor.execute("""
                    SELECT MAX(CASE WHEN max_level > 0 THEN final_score::float / max_level ELSE 0 END) 
                    FROM chase_games WHERE telegram_id = %s
                """, (telegram_id,))
                best_sr_result = cursor.fetchone()
                best_sr = round(best_sr_result[0], 1) if best_sr_result[0] else 0.0
                
                # Get player's rank in leaderboard (simplified query)
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
                
                # Get all roast lines with usage counts
                cursor.execute("""
                    SELECT roast_line, usage_count, last_used_date
                    FROM roast_rotation 
                    ORDER BY usage_count ASC, RANDOM()
                """)
                
                roast_data = cursor.fetchall()
                
                # Update cache
                self.roast_cache['lines'] = [row[0] for row in roast_data]
                self.roast_cache['usage_counts'] = {row[0]: row[1] for row in roast_data}
                self.roast_cache['last_updated'] = time.time()
                
                cursor.close()
                logger.info(f"Roast cache refreshed with {len(roast_data)} lines")
                return True
                
            except Exception as e:
                logger.error(f"Error refreshing roast cache: {e}")
                return False
    
    def get_cached_roast_line(self) -> str:
        """Get next roast line using cache for better performance"""
        current_time = time.time()
        
        # Check if cache needs refresh
        if (current_time - self.roast_cache['last_updated'] > self.roast_cache['cache_duration'] or 
            not self.roast_cache['lines']):
            if not self.refresh_roast_cache():
                # Fallback to random if cache refresh fails
                return random.choice(GOAT_ROAST_LINES)
        
        # Get least used roast from cache
        lines = self.roast_cache['lines']
        usage_counts = self.roast_cache['usage_counts']
        
        if not lines:
            return random.choice(GOAT_ROAST_LINES)
        
        # Find minimum usage count
        min_usage = min(usage_counts.values()) if usage_counts else 0
        
        # Get all lines with minimum usage
        available_lines = [line for line in lines if usage_counts.get(line, 0) == min_usage]
        
        if not available_lines:
            available_lines = lines
        
        # Return random choice from available lines
        return random.choice(available_lines)
    
    def get_cached_profile_data(self, user_id: int) -> dict:
        """Get cached profile data or fetch from database"""
        current_time = time.time()
        
        # Check if we have cached data for this user
        if (user_id in self.profile_cache['data'] and 
            user_id in self.profile_cache['last_updated'] and
            current_time - self.profile_cache['last_updated'][user_id] < self.profile_cache['cache_duration']):
            return self.profile_cache['data'][user_id]
        
        # Fetch fresh data
        try:
            # Find player
            player = self.find_player_by_identifier(str(user_id))
            if not player:
                return None
            
            # Get achievements
            achievements = self.get_player_achievements(player['id'])
            
            # Get chase stats
            chase_stats = self.get_player_chase_stats(player['id'])
            
            # Cache the data
            profile_data = {
                'player': player,
                'achievements': achievements,
                'chase_stats': chase_stats
            }
            
            self.profile_cache['data'][user_id] = profile_data
            self.profile_cache['last_updated'][user_id] = current_time
            
            return profile_data
            
        except Exception as e:
            logger.error(f"Error getting profile data for user {user_id}: {e}")
            return None
    
    def update_roast_usage_async(self, roast_line: str):
        """Update roast usage asynchronously to avoid blocking"""
        import threading
        
        def update_usage():
            try:
                conn = self.get_db_connection()
                if not conn:
                    return
                
                cursor = conn.cursor()
                
                # Update usage count
                cursor.execute("""
                    UPDATE roast_rotation 
                    SET usage_count = usage_count + 1, 
                        last_used_date = CURRENT_DATE 
                    WHERE roast_line = %s
                """, (roast_line,))
                
                conn.commit()
                self.return_db_connection(conn)
                
                # Update local cache
                if roast_line in self.roast_cache['usage_counts']:
                    self.roast_cache['usage_counts'][roast_line] += 1
                
            except Exception as e:
                logger.error(f"Error updating roast usage: {e}")
        
        # Run in background thread
        threading.Thread(target=update_usage, daemon=True).start()

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
        
        # Store in active games
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
        
        # Calculate final score and time
        time_taken = int(time.time() - game['start_time'])
        final_score = self.calculate_guess_score(game, outcome, time_taken)
        
        # Record in database
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
        
        # Update player scores and unlock levels if won
        if outcome == 'won' and final_score > 0:
            self.update_player_scores(user_id, final_score)
            unlocked_level = self.unlock_next_level(user_id, game['difficulty'])
            if unlocked_level:
                # Store unlocked level info for display
                game['new_level_unlocked'] = unlocked_level
        
        # Award shards for game completion
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
                        # Store shard reward in game for display
                        game['shard_reward'] = shard_reward
                    else:
                        logger.warning(f"Failed to award shards to player {user_id}")
            
            except Exception as e:
                logger.error(f"Error awarding guess game shards: {e}")
            
            # Update daily leaderboard
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
        
        # Remove from active games
        del self.guess_games[user_id]
        
        return success
    
    def calculate_guess_score(self, game: dict, outcome: str, time_taken: int) -> int:
        """Calculate final score for guess game"""
        if outcome != 'won':
            return 0
            
        config = self.guess_difficulties[game['difficulty']]
        base_score = int(100 * config['multiplier'])
        
        # Speed bonus (max 50 points)
        speed_bonus = max(0, 50 - (time_taken // 4))
        
        # Attempt bonus (10 points per unused attempt)
        attempt_bonus = max(0, (game['max_attempts'] - game['attempts_used']) * 10)
        
        # Perfect guess bonus
        perfect_bonus = 25 if game['attempts_used'] == 1 else 0

        if game['hint_used']:
            final_score = (base_score + speed_bonus + attempt_bonus + perfect_bonus) // 2
        else:
            final_score = base_score + speed_bonus + attempt_bonus + perfect_bonus
        
        # Daily challenge bonus (50% extra score)
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
            # Get current scores
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
            
            # Get player_id if needed
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
                # Player-specific stats
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
                # Global stats
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
        
        # Narrow down based on previous guesses
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
            # Give a narrower range
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
                
            # Check if game has exceeded time limit
            elapsed = current_time - game['start_time']
            if elapsed > game['time_limit']:
                self.end_guess_game(user_id, 'timeout')
                expired_users.append(user_id)
        
        # Remove expired games
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
            
            # Get player info for backup
            cursor.execute("SELECT display_name, title FROM players WHERE id = %s", (player_id,))
            player_data = cursor.fetchone()
            if not player_data:
                return False
            
            player_name, current_title = player_data
            
            # Backup current achievements
            cursor.execute("SELECT achievement_name, count FROM achievements WHERE player_id = %s", (player_id,))
            achievements = cursor.fetchall()
            
            for achievement, count in achievements:
                self.backup_achievement_action(player_id, player_name, achievement, count, 'RESET', performed_by)
            
            # Delete achievements
            cursor.execute("DELETE FROM achievements WHERE player_id = %s", (player_id,))
            
            # Reset title
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
        
        # Start with the header format
        message = f"🏏 <b>SPL ACHIEVEMENTS</b>\n<b>━━━━━━━━━━━━━━━</b>\n👤 <b>Player:</b> <b>{name}</b>\n"
        
        # Add title if exists (in bold)
        if player.get("title"):
            message += f"👑 <b>Title:</b> <b>{player['title']}</b>\n"
        
        message += "━━━━━━━━━━━━━━━\n\n"
        
        if not achievements:
            message += "🚫 <b>No achievements yet!</b>\n💪 <b>Keep playing to earn your first award!</b> 🎯"
            return message
        
        # Achievement list header
        message += "🏆 <b>ACHIEVEMENT LIST:</b>\n\n"
        
        # List achievements with numbers and bold names
        total_awards = 0
        for index, (achievement, count) in enumerate(achievements, 1):
            emoji = self.get_achievement_emoji(achievement)
            emoji_display = f"{emoji} " if emoji else "🎖️ "
            count_display = f" <b>(×{count})</b>" if count > 1 else ""
            message += f"<b>{index}.</b> {emoji_display}<b>{achievement}</b>{count_display}\n"
            total_awards += count
        
        # Footer with total count
        message += f"\n<b>━━━━━━━━━━━━━━━</b>\n📊 <b>Total Awards:</b> <b>{total_awards}</b> 🎖"
        
        return message

    # ====================================
    # GAME STATE MANAGEMENT
    # ====================================
    
    def get_all_active_games(self, telegram_id: int) -> dict:
        """Get all active games for a user across all game modes"""
        active_games = {
            'guess': None,
            'nightmare': None,
            'chase': []  # Can have multiple chase games
        }
        
        # Check guess game
        guess_game = self.get_guess_game(telegram_id)
        if guess_game and guess_game.get('game_active'):
            active_games['guess'] = {
                'type': 'Guess Game',
                'difficulty': guess_game.get('difficulty', '').title(),
                'attempts_left': guess_game.get('max_attempts', 0) - guess_game.get('attempts_used', 0),
                'time_left': max(0, guess_game.get('time_limit', 0) - int(time.time() - guess_game.get('start_time', 0))),
                'range': f"{guess_game.get('range_min', 1)}-{guess_game.get('range_max', 100)}"
            }
        
        # Check nightmare game  
        nightmare_game = self.get_nightmare_game(telegram_id)
        if nightmare_game:
            active_games['nightmare'] = {
                'type': 'Nightmare Mode',
                'attempts_left': nightmare_game.get('max_attempts', 0) - nightmare_game.get('attempts_used', 0),
                'hint': nightmare_game.get('decoded_hint', 'Mystery number awaits...')
            }
        
        # Check chase games
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
        
        # Count total active games
        total_active = 0
        if active_games['guess']:
            total_active += 1
        if active_games['nightmare']:
            total_active += 1
        total_active += len(active_games['chase'])
        
        if total_active == 0:
            return "", False  # No conflicts
        
        # Create informative message
        message = f"🎮 <b>Active Games Status</b>\n\n"
        message += f"You currently have <b>{total_active}</b> active game{'s' if total_active > 1 else ''}:\n\n"
        
        # List all active games with details
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
            message += f"   • Hint: {n['hint'][:30]}{'...' if len(n['hint']) > 30 else ''}\n\n"
        
        for i, chase in enumerate(active_games['chase']):
            message += f"🏏 <b>{chase['type']}</b> #{i+1}\n"
            message += f"   • Level {chase['level']}\n"
            message += f"   • Score: {chase['score']}\n"
            message += f"   • {chase['balls_left']} balls, {chase['wickets_left']} wickets left\n\n"
        
        # Add helpful suggestion
        message += f"💡 <b>Options:</b>\n"
        message += f"• Continue any active game\n"
        message += f"• Use /quit to end current games\n"
        message += f"• Or start {new_game_type} anyway (games auto-cleanup after timeout)\n\n"
        message += f"🎯 <b>All games earn shards - no progress is lost!</b>"
        
        return message, True
    
    # ====================================
    # SHARDS CURRENCY SYSTEM
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
            
            # Get current balance, player name, and ID
            cursor.execute("SELECT shards, total_shards_earned, display_name, id FROM players WHERE telegram_id = %s", (telegram_id,))
            result = cursor.fetchone()
            
            if not result:
                # Player doesn't exist - create them first with default name
                logger.warning(f"Player {telegram_id} not found during shard award - auto-creating")
                cursor.execute("""
                    INSERT INTO players (telegram_id, username, display_name)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (telegram_id) DO NOTHING
                """, (telegram_id, None, source_details or f"User {telegram_id}"))
                
                # Try to get player data again
                cursor.execute("SELECT shards, total_shards_earned, display_name, id FROM players WHERE telegram_id = %s", (telegram_id,))
                result = cursor.fetchone()
                
                if not result:
                    logger.error(f"Failed to create player {telegram_id} for shard award")
                    return False
                
            current_shards, total_earned, player_name, player_id = result
            new_balance = (current_shards or 0) + amount
            new_total = (total_earned or 0) + amount
            
            # Get the performed_by player ID if specified
            performed_by_id = None
            if performed_by:
                cursor.execute("SELECT id FROM players WHERE telegram_id = %s", (performed_by,))
                admin_result = cursor.fetchone()
                performed_by_id = admin_result[0] if admin_result else None
            
            # Update player shards
            cursor.execute("""
                UPDATE players 
                SET shards = %s, total_shards_earned = %s 
                WHERE telegram_id = %s
            """, (new_balance, new_total, telegram_id))
            
            # Record transaction with correct player IDs
            cursor.execute("""
                INSERT INTO shard_transactions 
                (player_id, player_telegram_id, player_name, transaction_type, amount, source, source_details, 
                 balance_before, balance_after, performed_by, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (player_id, telegram_id, player_name or f"User {telegram_id}", 'EARN', amount, source, source_details,
                  current_shards or 0, new_balance, performed_by_id, notes))
            
            conn.commit()
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
            
            # Get current balance, player name, and ID
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
            
            # Get the performed_by player ID if specified
            performed_by_id = None
            if performed_by:
                cursor.execute("SELECT id FROM players WHERE telegram_id = %s", (performed_by,))
                admin_result = cursor.fetchone()
                performed_by_id = admin_result[0] if admin_result else None
            
            # Update player shards
            cursor.execute("""
                UPDATE players 
                SET shards = %s 
                WHERE telegram_id = %s
            """, (new_balance, telegram_id))
            
            # Record transaction with correct player IDs
            cursor.execute("""
                INSERT INTO shard_transactions 
                (player_id, player_telegram_id, player_name, transaction_type, amount, source, source_details, 
                 balance_before, balance_after, performed_by, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (player_id, telegram_id, player_name or f"User {telegram_id}", 'ADMIN_REMOVE', -amount, source, source_details,
                  current_shards, new_balance, performed_by_id, notes))
            
            conn.commit()
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
            
        # Prevent integer overflow - limit to reasonable amounts
        MAX_SHARD_AMOUNT = 2147483647  # PostgreSQL INTEGER max value
        if amount > MAX_SHARD_AMOUNT:
            logger.error(f"Amount {amount} too large, max allowed: {MAX_SHARD_AMOUNT}")
            return False
            
        conn = self.get_db_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Get current balance, player name, and ID
            cursor.execute("SELECT shards, total_shards_earned, display_name, id FROM players WHERE telegram_id = %s", (telegram_id,))
            result = cursor.fetchone()
            
            if not result:
                logger.error(f"Player {telegram_id} not found for shard award")
                return False
                
            current_shards, total_earned, player_name, player_id = result
            new_balance = current_shards + amount
            new_total = (total_earned or 0) + amount
            
            # Prevent overflow in balances
            if new_balance > MAX_SHARD_AMOUNT or new_total > MAX_SHARD_AMOUNT:
                logger.error(f"Balance would overflow: new_balance={new_balance}, new_total={new_total}")
                return False
            
            # Get the admin's player ID for performed_by field
            cursor.execute("SELECT id FROM players WHERE telegram_id = %s", (performed_by,))
            admin_result = cursor.fetchone()
            admin_player_id = admin_result[0] if admin_result else None
            
            # Update player shards
            cursor.execute("""
                UPDATE players 
                SET shards = %s, total_shards_earned = %s 
                WHERE telegram_id = %s
            """, (new_balance, new_total, telegram_id))
            
            # Record transaction with correct player IDs
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
            
            # Check if already claimed today
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
            
            # Calculate streak
            yesterday = today.replace(day=today.day - 1) if today.day > 1 else today.replace(month=today.month - 1 if today.month > 1 else 12, day=31 if today.month > 1 else 30)
            
            streak_days = 1
            if last_claim == yesterday:
                # Get current streak from most recent daily bonus
                cursor.execute("""
                    SELECT streak_days FROM daily_shard_bonuses 
                    WHERE player_id = (SELECT id FROM players WHERE telegram_id = %s)
                    ORDER BY claim_date DESC LIMIT 1
                """, (telegram_id,))
                
                streak_result = cursor.fetchone()
                if streak_result:
                    streak_days = streak_result[0] + 1
            
            # Calculate bonus amount (base + streak bonus)
            base_amount = 50  # Base daily bonus
            streak_bonus = min((streak_days - 1) * 10, 200)  # Up to 20 days = +200 bonus
            total_amount = base_amount + streak_bonus
            
            # Award the shards
            success = self.award_shards(
                telegram_id, 
                total_amount, 
                'daily_bonus', 
                f'Day {streak_days} streak bonus',
                None,
                f'Base: {base_amount}, Streak bonus: {streak_bonus}'
            )
            
            if success:
                # Update last claim date
                cursor.execute("""
                    UPDATE players SET last_daily_claim = %s WHERE telegram_id = %s
                """, (today, telegram_id))
                
                # Record daily bonus
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
            
            # Total shards in circulation (current balances)
            cursor.execute("SELECT COALESCE(SUM(current_shards), 0) FROM players")
            total_circulation = cursor.fetchone()[0]
            
            # Total shards ever earned
            cursor.execute("SELECT COALESCE(SUM(total_shards_earned), 0) FROM players")
            total_earned = cursor.fetchone()[0]
            
            # Total transactions
            cursor.execute("SELECT COUNT(*) FROM shard_transactions")
            total_transactions = cursor.fetchone()[0]
            
            # Breakdown by source
            cursor.execute("""
                SELECT source, COUNT(*), COALESCE(SUM(amount), 0)
                FROM shard_transactions 
                WHERE transaction_type = 'earned'
                GROUP BY source
                ORDER BY SUM(amount) DESC
            """)
            earn_sources = cursor.fetchall()
            
            # Top shard holders
            cursor.execute("""
                SELECT display_name, current_shards, total_shards_earned
                FROM players 
                WHERE current_shards > 0
                ORDER BY current_shards DESC 
                LIMIT 10
            """)
            top_holders = cursor.fetchall()
            
            # Daily activity (last 7 days)
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
            # Small consolation prize
            return max(2, base_reward // 5)  # Reduced consolation
        
        # Bonus calculations - reduced bonuses
        score_bonus = min(score // 150, 25)  # Reduced from 100->50 to 150->25
        level_bonus = min((level - 1) * 3, 15)  # Reduced from 5->30 to 3->15
        
        total_reward = base_reward + score_bonus + level_bonus
        
        # Cap the reward - reduced multiplier
        max_reward = base_reward * 2  # Reduced from 3x to 2x
        return min(total_reward, max_reward)

    # ====================================
    # DAILY LEADERBOARD SYSTEM
    # ====================================
    
    def update_daily_leaderboard(self, telegram_id: int, player_name: str, game_type: str, score: int, level: int = 1, won: bool = True) -> bool:
        """Update player's daily leaderboard entry"""
        conn = self.get_db_connection()
        if not conn:
            logger.error("Failed to get database connection for daily leaderboard update")
            return False
        
        try:
            cursor = conn.cursor()
            
            # Clean player name for Unicode issues
            clean_player_name = player_name.encode('utf-8', 'ignore').decode('utf-8', 'ignore')[:100]
            
            # Use the database function for safe updates
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
            
            # Manual reward distribution with correct reward structure
            reward_count = 0
            target_date = reward_date if reward_date else datetime.now().date()
            
            # Standard reward amounts: [100,100,100,80,80,80,60,60,40,20]
            rewards = [100, 100, 100, 80, 80, 80, 60, 60, 40, 20]
            
            # Get top players for the game type using the correct view
            leaderboard = self.get_daily_leaderboard(game_type, 10)
            
            if not leaderboard:
                logger.info(f"No players found for {game_type} leaderboard on {target_date}")
                return 0
                
            # Award rewards to top players
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
                
                # Update status table to prevent duplicate distributions
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
    # ADMIN LOGS SYSTEM
    # ====================================
    
    async def send_admin_log(self, log_type: str, message: str, user_id: int = None, username: str = None) -> bool:
        """Send log messages to admin logs group chat"""
        if not self.logs_enabled:
            return True
            
        try:
            # Format the log message with timestamp and icons
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Choose emoji based on log type
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
            
            # Format user info if provided
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
            
            # Send using the logs bot
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
    # NIGHTMARE MODE METHODS
    # ====================================
    
    def generate_nightmare_hint(self, number: int) -> dict:
        """Generate strategic mathematical hints - challenging but not too direct!"""
        import random
        
        number_str = str(number)
        digits = [int(d) for d in number_str]
        
        # Create accurate mathematical hints for 1-10000 range
        hint_variants = []
        
        # Basic info hints (always accurate)
        hint_variants.append(f"🔢 This number has {len(number_str)} digits")
        hint_variants.append(f"➕ The sum of all digits is {sum(digits)}")
        
        # Mathematical properties (always accurate)
        if number % 2 == 0:
            hint_variants.append("⚡ This number is EVEN")
        else:
            hint_variants.append("⚡ This number is ODD")
        
        # Strategic comparison hints (not too revealing)
        if number > 5000:
            hint_variants.append("� Greater than 5000")
        elif number < 1000:
            hint_variants.append("� Less than 1000")
        
        if number > 2500 and number < 7500:
            hint_variants.append("🎯 Between 2500 and 7500")
        
        # Digit-specific hints (always accurate)
        if len(digits) > 1:
            # Combined first and last digit hint
            hint_variants.append(f"🎯� First digit: {digits[0]}, Last digit: {digits[-1]}")
            hint_variants.append(f"�🎲 Contains the digit {random.choice(digits)}")
            
            # Special digit patterns
            if digits[0] == digits[-1]:
                hint_variants.append("🔄 First and last digits are the same")
            
            if len(set(digits)) == 1:
                hint_variants.append("🎭 All digits are the same")
            elif len(set(digits)) == len(digits):
                hint_variants.append("🌟 All digits are unique")
        
        # Divisibility hints (mathematical challenge)
        if number % 3 == 0:
            hint_variants.append("� Divisible by 3")
        if number % 4 == 0:
            hint_variants.append("🎳 Divisible by 4")
        if number % 5 == 0:
            hint_variants.append("🔮 Divisible by 5")
        if number % 6 == 0:
            hint_variants.append("⚡ Divisible by 6")
        if number % 8 == 0:
            hint_variants.append("� Divisible by 8")
        if number % 9 == 0:
            hint_variants.append("🌟 Divisible by 9")
        if number % 11 == 0:
            hint_variants.append("� Divisible by 11")
        if number % 25 == 0:
            hint_variants.append("💎 Divisible by 25")
        
        # Last digit patterns (strategic)
        if number % 10 == 0:
            hint_variants.append("🎯 Ends in zero")
        elif str(number)[-1] in ['1', '3', '7', '9']:
            hint_variants.append("🔥 Ends in a prime digit")
        elif str(number)[-1] in ['2', '4', '6', '8']:
            hint_variants.append("📐 Ends in an even digit")
        elif str(number)[-1] == '5':
            hint_variants.append("⭐ Ends in five")
        
        # Pick a random hint
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
            
            # Check for existing active game
            cursor.execute("""
                SELECT game_id FROM nightmare_games 
                WHERE player_telegram_id = %s AND NOT is_completed
            """, (telegram_id,))
            
            if cursor.fetchone():
                # Create smooth transition message for nightmare conflicts
                other_games = self.get_all_active_games(telegram_id)
                if other_games['guess'] or other_games['chase']:
                    transition_msg, _ = self.create_game_switch_message(telegram_id, "Nightmare Mode")
                    return {'error': transition_msg}
                else:
                    return {'error': 'You have an active nightmare mode game! Use /nightmare to continue it.'}
            
            # Generate game parameters
            range_min = 1
            range_max = 10000
            target_number = random.randint(range_min, range_max)
            max_attempts = 3  # Match schema default
            
            # Generate hint using Python function
            hint_data = self.generate_nightmare_hint(target_number)
            
            # Insert new game record directly without any function calls
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
            
            # Provide more specific error messages
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
            
            # Get current game state
            cursor.execute("""
                SELECT * FROM nightmare_games 
                WHERE game_id = %s AND NOT is_completed
            """, (game_id,))
            game = cursor.fetchone()
            
            if not game:
                return {'error': 'Game not found or already completed'}
            
            # Check if guess is correct
            is_correct = (guess == game['current_number'])
            attempts_used = game['attempts_used'] + 1
            attempts_remaining = game['max_attempts'] - attempts_used
            game_over = is_correct or attempts_remaining <= 0
            won = is_correct
            
            # Update game state
            if game_over:
                cursor.execute("""
                    UPDATE nightmare_games 
                    SET attempts_used = %s, is_completed = TRUE, completed_at = NOW()
                    WHERE game_id = %s
                """, (attempts_used, game_id))
                
                if won:
                    # Award shards and update stats for win
                    self.award_shards(game['player_telegram_id'], 10000, 'Nightmare Mode Victory', 
                                    game['player_name'])
                    
                    # Log nightmare victory (thread-safe)
                    try:
                        import threading
                        def send_victory_log():
                            import asyncio
                            try:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                loop.run_until_complete(self.send_admin_log(
                                    'nightmare_win',
                                    f"NIGHTMARE MODE VICTORY! Player: {game['player_name']} | Target: {game['current_number']:,} | Guess: {guess:,} | Attempts: {attempts_used}/3 | Reward: 10,000 shards",
                                    game['player_telegram_id'],
                                    None
                                ))
                                loop.close()
                            except Exception as e:
                                logger.debug(f"Could not send nightmare victory log: {e}")
                        
                        threading.Thread(target=send_victory_log, daemon=True).start()
                    except Exception as e:
                        logger.debug(f"Could not schedule nightmare victory log: {e}")
                    
                    # Update player stats
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
                    # Update stats for loss
                    cursor.execute("""
                        UPDATE players SET 
                            nightmare_games_played = nightmare_games_played + 1,
                            nightmare_total_attempts = nightmare_total_attempts + %s,
                            nightmare_last_played = NOW()
                        WHERE telegram_id = %s
                    """, (attempts_used, game['player_telegram_id']))
            else:
                # Just update attempts if game continues
                cursor.execute("""
                    UPDATE nightmare_games 
                    SET attempts_used = %s
                    WHERE game_id = %s
                """, (attempts_used, game_id))
                
                # Shift the number for next guess (predictable nightmare mode)
                # Make shifting more transparent and educational
                guess_direction = "high" if guess > game['current_number'] else "low"
                
                # Predictable shift based on guess accuracy (adjusted for 10k range)
                if abs(guess - game['current_number']) <= 500:
                    # Close guess - small shift
                    shift_amount = random.randint(10, 100)
                    shift_info = "🎯 You were close! Small shift applied."
                elif abs(guess - game['current_number']) <= 2000:
                    # Medium distance - medium shift
                    shift_amount = random.randint(100, 500)
                    shift_info = "🎲 Moderate distance. Medium shift applied."
                else:
                    # Far guess - larger shift
                    shift_amount = random.randint(500, 1500)
                    shift_info = "🌪️ Way off target! Large shift applied."
                
                # Apply shift in a logical direction
                if guess_direction == "high":
                    new_target = (game['current_number'] - shift_amount) % (game['game_range_max'] - game['game_range_min'] + 1) + game['game_range_min']
                else:
                    new_target = (game['current_number'] + shift_amount) % (game['game_range_max'] - game['game_range_min'] + 1) + game['game_range_min']
                
                # Generate new hint for shifted number
                hint_data = self.generate_nightmare_hint(new_target)
                
                cursor.execute("""
                    UPDATE nightmare_games 
                    SET current_number = %s, encoded_hint = %s, hint_type = %s, decoded_hint = %s
                    WHERE game_id = %s
                """, (new_target, hint_data['encoded_hint'], hint_data['hint_type'], hint_data['encoded_hint'], game_id))
            
            conn.commit()
            
            # Generate response message
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
                # No decoding needed - hint is already clear and accurate
                return encoded_hint
            elif hint_type.upper() == 'ROT13':
                # ROT13 decode
                import codecs
                return codecs.decode(encoded_hint, 'rot13')
            elif hint_type.upper() == 'BASE64':
                # Base64 decode
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
            
            # Check if player already has the achievement
            cursor.execute("""
                SELECT has_shard_mastermind FROM players 
                WHERE telegram_id = %s
            """, (telegram_id,))
            
            result = cursor.fetchone()
            if result and result[0]:
                return True  # Already has achievement
            
            # Get player info
            player = self.find_player_by_telegram_id(telegram_id)
            if not player:
                return False
            
            # Request admin confirmation
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
            
            # Notify admins about the achievement
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
            
            # Send to all admins
            for admin_id in admin_list:
                try:
                    asyncio.create_task(
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
            
            # Total players
            cursor.execute("SELECT COUNT(*) FROM players")
            total_players = cursor.fetchone()[0] or 0
            
            # Active players (7 days)
            cursor.execute("""
                SELECT COUNT(DISTINCT telegram_id) 
                FROM shard_transactions 
                WHERE performed_at >= CURRENT_DATE - INTERVAL '7 days'
            """)
            active_players = cursor.fetchone()[0] or 0
            
            # Total chase games
            cursor.execute("SELECT COUNT(*) FROM chase_games")
            chase_games = cursor.fetchone()[0] or 0
            
            # Total guess games
            cursor.execute("SELECT COUNT(*) FROM guess_games")
            guess_games = cursor.fetchone()[0] or 0
            
            # Total nightmare games
            cursor.execute("SELECT COUNT(*) FROM nightmare_games")
            nightmare_games = cursor.fetchone()[0] or 0
            
            # Total shards in circulation
            cursor.execute("SELECT COALESCE(SUM(current_shards), 0) FROM players")
            total_shards = cursor.fetchone()[0] or 0
            
            # Average balance
            cursor.execute("SELECT COALESCE(AVG(current_shards), 0) FROM players WHERE current_shards > 0")
            avg_balance = cursor.fetchone()[0] or 0
            
            # Total achievements
            cursor.execute("SELECT COUNT(*) FROM achievements")
            total_achievements = cursor.fetchone()[0] or 0
            
            # Unique achievements
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
# SHARDS CURRENCY COMMANDS
# ====================================

async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show user's current shard balance"""
    try:
        user_id = update.effective_user.id
        user = update.effective_user
        
        # Ensure user is registered
        username = user.username or ""
        display_name = user.full_name or user.first_name or f"User{user.id}"
        bot_instance.create_or_update_player(user_id, username, display_name)
        
        # Get current balance and total earned
        current_shards, total_earned = bot_instance.get_player_shards(user_id)
        
        # Calculate actual player rank
        conn = bot_instance.get_db_connection()
        rank = 1  # Default fallback
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) + 1 as rank
                    FROM players 
                    WHERE shards > %s AND (shards > 0 OR total_shards_earned > 0)
                """, (current_shards,))
                rank = cursor.fetchone()[0]
            except Exception as e:
                logger.error(f"Error calculating rank: {e}")
            finally:
                bot_instance.return_db_connection(conn)
        
        message = (
            f"💠 <b>YOUR SHARD BALANCE</b> 💠\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💎 <b>Current Balance:</b> {current_shards:,} shards\n"
            f"✨ <b>Total Earned:</b> {total_earned:,} shards\n"
            f"🏅 <b>Global Rank:</b> #{rank}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💡 <b>How to Earn More Shards:</b>\n"
            f"🎮 Play games: 15-30 per game\n"
            f"📅 Daily bonus: 50-250 per day\n"
            f"🏆 Achievements: 100-200 each\n"
            f"🐐 Daily GOAT: 300 bonus\n\n"
            f"Use /shardslb to see top holders!"
        )
        
        await safe_send(update.message.reply_text, message, parse_mode='HTML')
        
    except Exception as e:
        log_exception("balance_command", e, update.effective_user.id)
        await safe_send(update.message.reply_text, 
                       "❌ Error retrieving balance. Please try again.", 
                       parse_mode='HTML')

async def daily_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Claim daily shard bonus"""
    try:
        user_id = update.effective_user.id
        user = update.effective_user
        
        # Ensure user is registered
        username = user.username or ""
        display_name = user.full_name or user.first_name or f"User{user.id}"
        bot_instance.create_or_update_player(user_id, username, display_name)
        
        # Attempt to claim daily bonus
        success, amount, streak_days, message = bot_instance.claim_daily_shards(user_id)
        
        if success:
            # Get updated balance
            current_shards, total_earned = bot_instance.get_player_shards(user_id)
            
            response = (
                f"🎉 <b>DAILY BONUS CLAIMED!</b> 🎉\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"👤 <b>{H(display_name)}</b>\n\n"
                f"💠 <b>Earned:</b> +{amount:,} shards\n"
                f"🔥 <b>Streak:</b> {streak_days} day{'s' if streak_days > 1 else ''}\n"
                f"💰 <b>New Balance:</b> {current_shards:,} shards\n"
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

async def shards_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show detailed shard balance and last 5 transactions"""
    try:
        user_id = update.effective_user.id
        user = update.effective_user
        
        # Parse optional target user argument
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
        
        # Ensure target user is registered
        if target_user_id == user_id:
            username = user.username or ""
            display_name = user.full_name or user.first_name or f"User{user.id}"
            bot_instance.create_or_update_player(user_id, username, display_name)
            target_player = {'display_name': display_name, 'telegram_id': user_id}
        
        # Get user's current stats
        current_shards, total_earned = bot_instance.get_player_shards(target_user_id)
        
        # Get last 5 transactions
        transactions = bot_instance.get_shard_transactions(target_user_id, 5)
        
        # Get user's rank
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
            
            # Build message
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
                    # Format transaction type
                    trans_type = transaction.get('transaction_type', 'UNKNOWN')
                    amount = transaction.get('amount', 0)
                    source = transaction.get('source', 'Unknown')
                    created = transaction.get('created_at')
                    
                    # Format date
                    if created:
                        try:
                            if isinstance(created, str):
                                created = datetime.fromisoformat(created.replace('Z', '+00:00'))
                            date_str = created.strftime("%m/%d %H:%M")
                        except:
                            date_str = "Recent"
                    else:
                        date_str = "Recent"
                    
                    # Format amount and type
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
            message += f"Use /shardslb to see top holders!"
            
            await safe_send(update.message.reply_text, message, parse_mode='HTML')
            
        finally:
            bot_instance.return_db_connection(conn)
        
    except Exception as e:
        log_exception("shards_command", e, update.effective_user.id)
        await safe_send(update.message.reply_text, 
                       "❌ Error retrieving shard balance. Please try again.", 
                       parse_mode='HTML')

async def shard_leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show top shard holders leaderboard"""
    try:
        user_id = update.effective_user.id
        user = update.effective_user
        
        # Ensure user is registered
        username = user.username or ""
        display_name = user.full_name or user.first_name or f"User{user.id}"
        bot_instance.create_or_update_player(user_id, username, display_name)
        
        # Get user's current stats
        current_shards, total_earned = bot_instance.get_player_shards(user_id)
        
        # Get top shard holders
        conn = bot_instance.get_db_connection()
        if not conn:
            await safe_send(update.message.reply_text, "❌ Database connection failed")
            return
            
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT display_name, shards, total_shards_earned, 
                       ROW_NUMBER() OVER (ORDER BY shards DESC) as rank
                FROM players 
                WHERE shards > 0 OR total_shards_earned > 0
                ORDER BY shards DESC 
                LIMIT 20
            """)
            
            leaderboard = cursor.fetchall()
            
            # Find user's rank
            cursor.execute("""
                SELECT COUNT(*) + 1 as rank
                FROM players 
                WHERE shards > %s AND (shards > 0 OR total_shards_earned > 0)
            """, (current_shards,))
            
            user_rank = cursor.fetchone()[0]
            
            message = f"🏆 <b>SHARD LEADERBOARD</b> 🏆\n"
            message += f"━━━━━━━━━━━━━━━━━━━━\n"
            message += f"👤 <b>Your Rank:</b> #{user_rank} ({current_shards:,} 💠)\n\n"
            message += f"🔝 <b>Top 20 Shard Holders:</b>\n"
            
            for i, (name, balance, earned, rank) in enumerate(leaderboard, 1):
                if i <= 3:
                    medals = ["🥇", "🥈", "🥉"]
                    medal = medals[i-1]
                elif i == user_rank:
                    medal = f"👤{i}."
                else:
                    medal = f"{i}."
                
                message += f"{medal} <b>{H(name[:15])}</b> - {balance:,} 💠\n"
            
            message += f"\n━━━━━━━━━━━━━━━━━━━━\n"
            message += f"💡 <b>Climb the ranks by earning more shards!</b>\n"
            message += f"Use /shards to see your balance & recent activity"
            
            await send_paginated_message(update, context, message, parse_mode='HTML')
            
        finally:
            bot_instance.return_db_connection(conn)
        
    except Exception as e:
        log_exception("shard_leaderboard_command", e, update.effective_user.id)
        await safe_send(update.message.reply_text, 
                       "❌ Error retrieving shard leaderboard. Please try again.", 
                       parse_mode='HTML')

async def give_shards_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin command to give shards to a player"""
    if not bot_instance.is_admin(update.effective_user.id):
        await safe_send(update.message.reply_text, 
                       "❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can give shards!", 
                       parse_mode='HTML')
        return
    
    try:
        if len(context.args) < 2:
            await safe_send(update.message.reply_text,
                           "📝 <b>Usage:</b>\n"
                           "/giveshards &lt;player&gt; &lt;amount&gt; [reason]\n\n"
                           "<b>Examples:</b>\n"
                           "• /giveshards @user 500\n"
                           "• /giveshards 123456789 1000 Tournament winner\n"
                           "• /giveshards PlayerName 250 Event participation",
                           parse_mode='HTML')
            return
        
        # Parse arguments
        player_identifier = context.args[0]
        try:
            amount = int(context.args[1])
            if amount <= 0:
                raise ValueError("Amount must be positive")
            # Prevent excessive amounts
            if amount > 2000000000:  # 2 billion limit
                raise ValueError("Amount too large")
        except ValueError as e:
            if "too large" in str(e):
                await safe_send(update.message.reply_text, "❌ Amount too large. Maximum allowed is 2,000,000,000 shards.")
            else:
                await safe_send(update.message.reply_text, "❌ Invalid amount. Must be a positive number.")
            return
        
        reason = " ".join(context.args[2:]) if len(context.args) > 2 else "Admin award"
        
        # Find player
        player = bot_instance.find_player_by_identifier(player_identifier)
        if not player:
            await safe_send(update.message.reply_text, 
                           f"❌ Player '{H(player_identifier)}' not found.\n\n"
                           "💡 Make sure they've used the bot before!")
            return
        
        # Give shards
        success = bot_instance.give_admin_shards(
            player['telegram_id'], 
            amount, 
            update.effective_user.id, 
            reason
        )
        
        if success:
            # Get updated balance
            current_shards, total_earned = bot_instance.get_player_shards(player['telegram_id'])
            
            # Log admin action
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
        if len(context.args) < 2:
            await safe_send(update.message.reply_text,
                           "📝 <b>Usage:</b>\n"
                           "/removeshards &lt;player&gt; &lt;amount&gt; [reason]\n\n"
                           "<b>Examples:</b>\n"
                           "• /removeshards @user 500\n"
                           "• /removeshards 123456789 1000 Rule violation\n"
                           "• /removeshards PlayerName 250 Correction",
                           parse_mode='HTML')
            return
        
        # Parse arguments
        player_identifier = context.args[0]
        try:
            amount = int(context.args[1])
            if amount <= 0:
                raise ValueError("Amount must be positive")
        except ValueError:
            await safe_send(update.message.reply_text, "❌ Invalid amount. Must be a positive number.")
            return
        
        reason = " ".join(context.args[2:]) if len(context.args) > 2 else "Admin removal"
        
        # Find player
        player = bot_instance.find_player_by_identifier(player_identifier)
        if not player:
            await safe_send(update.message.reply_text, 
                           f"❌ Player '{H(player_identifier)}' not found.\n\n"
                           "💡 Make sure they've used the bot before!")
            return
        
        # Check current balance
        current_shards, _ = bot_instance.get_player_shards(player['telegram_id'])
        
        # Remove shards
        success = bot_instance.remove_shards(
            player['telegram_id'], 
            amount, 
            'admin_removal',
            reason,
            update.effective_user.id, 
            reason
        )
        
        if success:
            # Get updated balance
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
    
    # Check admin permissions
    if not bot_instance.is_admin(user_id):
        await safe_send(update.message.reply_text, 
                       "❌ <b>Admin Only Command</b>\n\nThis command is restricted to administrators only.", 
                       parse_mode='HTML')
        return
    
    try:
        # Get current daily leaderboards (top 10 from /dailylb)
        chase_lb = bot_instance.get_daily_leaderboard('chase', 10)
        guess_lb = bot_instance.get_daily_leaderboard('guess', 10)
        
        if not chase_lb and not guess_lb:
            await safe_send(update.message.reply_text, 
                           "📭 <b>No Leaderboard Data</b>\n\nNo players found in today's daily leaderboards.\n\nUse /dailylb to check current standings.", 
                           parse_mode='HTML')
            return
            
        # Check if rewards already distributed today
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
        
        # Show processing message
        await safe_send(update.message.reply_text, 
                       "⏳ <b>Distributing Daily Rewards...</b>\n\n💠 Processing top 10 players from both Chase and Guess leaderboards...",
                       parse_mode='HTML')
        
        # Distribute rewards manually
        chase_count = 0
        guess_count = 0
        
        if chase_lb:
            chase_count = bot_instance.distribute_daily_leaderboard_rewards('chase')
        
        if guess_lb:
            guess_count = bot_instance.distribute_daily_leaderboard_rewards('guess')
        
        # Send detailed confirmation
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
    
    # Check admin permissions
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
        
        # Reset daily leaderboard
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
    
    # Check admin permissions
    if not bot_instance.is_admin(user_id):
        await safe_send(update.message.reply_text, 
                       "❌ This command is only available to admins.", 
                       parse_mode='HTML')
        return
    
    try:
        from datetime import datetime
        today = datetime.now().strftime("%B %d, %Y")
        
        # Get leaderboard data
        chase_lb = bot_instance.get_daily_leaderboard('chase', 50)  # Get more for stats
        guess_lb = bot_instance.get_daily_leaderboard('guess', 50)
        
        # Calculate statistics
        total_chase_players = len(chase_lb)
        total_guess_players = len(guess_lb)
        total_chase_games = sum(p.get('chase_games_played', 0) for p in chase_lb)
        total_guess_games = sum(p.get('guess_games_played', 0) for p in guess_lb)
        total_chase_score = sum(p.get('chase_total_score', 0) for p in chase_lb)
        total_guess_score = sum(p.get('guess_total_score', 0) for p in guess_lb)
        
        # Top performers
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
        message += f"🏆 Use /distributedailyrewards to trigger manually"
        
        await safe_send(update.message.reply_text, message, parse_mode='HTML')
        
    except Exception as e:
        log_exception("daily_leaderboard_stats_command", e, update.effective_user.id)
        await safe_send(update.message.reply_text, 
                       "❌ Error retrieving daily leaderboard stats. Please try again.", 
                       parse_mode='HTML')

async def confirm_achievement_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Confirm special achievement (Admin only)"""
    user_id = update.effective_user.id
    
    # Check admin permissions
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
        
        # Confirm the achievement
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
    
    # Check admin permissions
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
# NIGHTMARE MODE COMMANDS  
# ====================================

async def nightmare_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start or continue nightmare mode game"""
    try:
        user_id = update.effective_user.id
        user = update.effective_user
        player_name = user.first_name or "Unknown"
        
        # Register player if not exists
        bot_instance.create_or_update_player(user_id, user.username or "", player_name)
        
        # Check for active game
        active_game = bot_instance.get_nightmare_game(user_id)
        
        if active_game:
            # Continue existing game
            attempts_left = active_game['max_attempts'] - active_game['attempts_used']
            
            message = f"🧩 <b>Nightmare Mode</b>\n"
            message += f"━━━━━━━━━━━━━━━━━━━━\n\n"
            message += f"🎯 <b>Range:</b> 1 to 10,000\n"
            message += f"🎯 <b>Attempts left:</b> {attempts_left}\n\n"
            message += f"💡 <b>Your Hint:</b>\n"
            message += f"{active_game['encoded_hint']}\n\n"
            message += f"🎮 <b>Just type your guess!</b>"
            
            # Simple buttons
            keyboard = [
                [InlineKeyboardButton("❓ Need Help?", callback_data="nightmare_help")],
                [InlineKeyboardButton("❌ Quit", callback_data=f"nightmare_quit_{active_game['game_id']}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
        else:
            # Start new game
            game_data = bot_instance.start_nightmare_game(user_id, player_name)
            
            if 'error' in game_data:
                await safe_send(update.message.reply_text, 
                               f"❌ {game_data['error']}", 
                               parse_mode='HTML')
                return
            
            message = f"🧩 <b>Nightmare Mode Started!</b>\n"
            message += f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            message += f"🎯 <b>Find the secret number (1 to 10,000)</b>\n"
            message += f"💫 <b>You have 3 attempts</b>\n"
            message += f"🏆 <b>Win 10,000 shards!</b>\n\n"
            message += f"💡 <b>Your Hint:</b>\n"
            
            # Display the clear hint
            hint = game_data.get('encoded_hint', 'No hint available')
            message += f"{hint}\n\n"
            message += f"⚠️ <b>Challenge:</b> The number changes slightly after each wrong guess!\n"
            message += f"💡 <b>Strategy:</b> Use the hint to narrow down the range, then guess smart!\n\n"
            message += f"🎮 <b>Type your first guess now!</b>"
            
            # Simple buttons
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
    await query.answer()
    
    try:
        data = query.data
        user_id = query.from_user.id
        
        if data.startswith('nightmare_decode_'):
            # No longer needed with clear hints
            message = f"🎉 <b>Good News!</b>\n"
            message += f"━━━━━━━━━━━━━━━━━━\n\n"
            message += f"🎯 <b>Your hint is already clear and readable!</b>\n"
            message += f"✨ <b>No decoding needed anymore</b>\n\n"
            message += f"🧠 <b>Just use the mathematical clues to find the number</b>\n"
            message += f"🎮 <b>Type your guess when ready!</b>"
            
            await query.edit_message_text(message, parse_mode='HTML')
            
        elif data == 'nightmare_stats':
            # Show player's nightmare stats
            stats = bot_instance.get_nightmare_stats(user_id)
            
            if not stats or stats.get('nightmare_games_played', 0) == 0:
                message = f"💀 <b>NIGHTMARE MODE STATS</b>\n"
                message += f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                message += f"🎮 You haven't played Nightmare Mode yet!\n"
                message += f"🌟 Ready to face the ultimate challenge?\n\n"
                message += f"Use /nightmare to begin your first attempt!"
            else:
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
            
            await query.edit_message_text(message, parse_mode='HTML')
            
        elif data == 'nightmare_leaderboard':
            # Show nightmare mode leaderboard
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
            
            await query.edit_message_text(message, parse_mode='HTML')
            
        elif data.startswith('nightmare_quit_'):
            # Quit active game
            game_id = int(data.split('_')[2])
            
            # Confirm quit
            keyboard = [
                [InlineKeyboardButton("✅ Yes, Quit", callback_data=f"nightmare_quit_confirm_{game_id}")],
                [InlineKeyboardButton("❌ Cancel", callback_data="nightmare_cancel")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            message = f"⚠️ <b>QUIT NIGHTMARE GAME?</b>\n\n"
            message += f"🚨 This will count as a loss!\n"
            message += f"💀 Are you sure you want to quit?"
            
            await query.edit_message_text(message, parse_mode='HTML', reply_markup=reply_markup)
            
        elif data.startswith('nightmare_quit_confirm_'):
            # Actually quit the game
            game_id = int(data.split('_')[3])
            
            # Mark game as completed but not won
            conn = bot_instance.get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE nightmare_games 
                        SET is_completed = TRUE, is_won = FALSE, completed_at = CURRENT_TIMESTAMP
                        WHERE game_id = %s AND player_telegram_id = %s
                    """, (game_id, user_id))
                    
                    # Update player stats
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
                    
                    await query.edit_message_text(message, parse_mode='HTML')
                    
                except Exception as e:
                    logger.error(f"Error quitting nightmare game: {e}")
                    await query.edit_message_text("❌ Error quitting game.", parse_mode='HTML')
                finally:
                    bot_instance.return_db_connection(conn)
            
        elif data == 'nightmare_cancel':
            await query.edit_message_text("🎮 Nightmare mode continues! Send your guess.", parse_mode='HTML')
            
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
            
            await query.edit_message_text(message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Error in nightmare_callback: {e}")
        await safe_send(query.message.reply_text, "❌ Error processing request.", parse_mode='HTML')

async def handle_nightmare_guess(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Handle nightmare mode guesses. Returns True if message was a nightmare guess."""
    try:
        user_id = update.effective_user.id
        text = update.message.text.strip()
        
        # Check if user has active nightmare game
        active_game = bot_instance.get_nightmare_game(user_id)
        if not active_game:
            return False  # Not a nightmare guess
        
        # Check if text is a valid number
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
        
        # Make the guess
        result = bot_instance.make_nightmare_guess(active_game['game_id'], guess)
        
        if 'error' in result:
            await safe_send(update.message.reply_text, 
                           f"❌ {result['error']}", 
                           parse_mode='HTML')
            return True
        
        if result['is_correct']:
            # WINNER!
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
            
            # Update daily leaderboard if applicable
            try:
                bot_instance.update_daily_leaderboard(user_id, 'nightmare', 10000, True, 3)  # Max score for nightmare
            except:
                pass  # Don't fail on daily leaderboard update
                
        elif result['game_over']:
            # Game over - failed
            message = f"💀 <b>Game Over</b> 💀\n"
            message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            message += f"😔 <b>You ran out of attempts</b>\n\n"
            message += f"❌ <b>Your guess:</b> {guess:,}\n"
            message += f"🎯 <b>The final number was:</b> {result['current_number']:,}\n"
            message += f"❌ <b>Used all 3 attempts</b>\n\n"
            message += f"💡 <b>Tip:</b> The shifting made it extra challenging!\n"
            message += f"🎮 <b>Want to try again?</b>"
            
            keyboard = [
                [InlineKeyboardButton("🔄 Try Again", callback_data="nightmare_retry")],
                [InlineKeyboardButton("📊 My Stats", callback_data="nightmare_stats")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await safe_send(update.message.reply_text, message, parse_mode='HTML', reply_markup=reply_markup)
            
        else:
            # Wrong guess but game continues
            attempts_left = result['attempts_remaining']
            
            message = f"❌ <b>Wrong Guess</b>\n"
            message += f"━━━━━━━━━━━━━━━━━━━━\n\n"
            message += f"🎯 <b>Your guess:</b> {guess:,}\n"
            
            # Show shifting info if available
            if result.get('shift_info'):
                message += f"{result['shift_info']}\n"
            else:
                message += f"🌪️ <b>Number has shifted!</b>\n"
                
            message += f"🎯 <b>Attempts left:</b> {attempts_left}\n\n"
            
            # Show new hint if available
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

async def dailylb_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show daily leaderboard with toggle buttons for Chase/Guess"""
    try:
        # Default to chase leaderboard
        game_type = 'chase'
        
        # Get both leaderboards with more entries
        chase_lb = bot_instance.get_daily_leaderboard('chase', 20)
        guess_lb = bot_instance.get_daily_leaderboard('guess', 20)
        
        # Create inline keyboard for switching
        keyboard = [
            [
                InlineKeyboardButton("🏏 Chase", callback_data="dailylb_chase"),
                InlineKeyboardButton("🎲 Guess", callback_data="dailylb_guess")
            ],
            [InlineKeyboardButton("🔄 Refresh", callback_data="dailylb_refresh")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Generate chase leaderboard message
        message = generate_daily_leaderboard_message('chase', chase_lb, guess_lb)
        
        # Send message with reply markup
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
            
            # Get both leaderboards
            chase_lb = bot_instance.get_daily_leaderboard('chase', 10)
            guess_lb = bot_instance.get_daily_leaderboard('guess', 10)
            
            # Create keyboard
            keyboard = [
                [
                    InlineKeyboardButton("🏏 Chase", callback_data="dailylb_chase"),
                    InlineKeyboardButton("🎲 Guess", callback_data="dailylb_guess")
                ],
                [InlineKeyboardButton("🔄 Refresh", callback_data="dailylb_refresh")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Generate appropriate message
            if action == 'chase' or action == 'refresh':
                message = generate_daily_leaderboard_message('chase', chase_lb, guess_lb)
            elif action == 'guess':
                message = generate_daily_leaderboard_message('guess', chase_lb, guess_lb)
            else:
                return
            
            try:
                await query.edit_message_text(message, parse_mode='HTML', reply_markup=reply_markup)
            except Exception as e:
                # If edit fails, send new message
                await safe_send(query.message.reply_text, message, parse_mode='HTML', reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"Error in dailylb_callback: {e}")
        await safe_send(query.message.reply_text, "❌ Error processing request.", parse_mode='HTML')

def generate_daily_leaderboard_message(game_type: str, chase_lb: list, guess_lb: list) -> str:
    """Generate daily leaderboard message for specified game type"""
    from datetime import datetime
    today = datetime.now().strftime("%B %d, %Y")
    
    if game_type == 'chase':
        leaderboard = chase_lb
        title = "🏏 DAILY CHASE LEADERBOARD"
        empty_msg = "🏏 No chase games played today!\n🎮 Be the first to play with /chase"
        
        # Reward info
        rewards_msg = "🏆 <b>Daily Rewards (Manual Distribution):</b>\n"
        rewards_msg += "🥇 1st-3rd: <b>100 💠 each</b>\n"
        rewards_msg += "🏅 4th-6th: <b>80 💠 each</b>\n"
        rewards_msg += "🎖️ 7th-8th: <b>60 💠 each</b>\n"
        rewards_msg += "🏵️ 9th: <b>40 💠</b> • 🎗️ 10th: <b>20 💠</b>\n\n"
        
    else:  # guess
        leaderboard = guess_lb
        title = "🎲 DAILY GUESS LEADERBOARD"  
        empty_msg = "🎲 No guess games played today!\n🎮 Be the first to play with /guess"
        
        # Reward info
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
        # Rank emojis
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
    
    # Show counts for both game types
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
        
        # Reward structure: [100,100,100,80,80,80,60,60,40,20]
        rewards = [100, 100, 100, 80, 80, 80, 60, 60, 40, 20]
        
        # Get today's leaderboards
        chase_lb = bot_instance.get_daily_leaderboard('chase', 10)
        guess_lb = bot_instance.get_daily_leaderboard('guess', 10)
        
        total_distributed = 0
        notifications = []
        
        # Distribute chase rewards
        for i, player in enumerate(chase_lb[:10]):  # Top 10 only
            reward_amount = rewards[i]
            player_id = player.get('player_telegram_id') or player.get('player_id')
            player_name = player.get('player_name', 'Unknown Player')
            rank = i + 1
            
            if not player_id:
                logger.error(f"No player ID found for chase rank {rank}: {player}")
                continue
            
            # Award shards
            success = bot_instance.award_shards(
                player_id, 
                reward_amount, 
                f"Daily Chase Leaderboard - Rank #{rank}"
            )
            
            if success:
                total_distributed += reward_amount
                # Add to notification list
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
        
        # Distribute guess rewards  
        for i, player in enumerate(guess_lb[:10]):  # Top 10 only
            reward_amount = rewards[i]
            player_id = player.get('player_telegram_id') or player.get('player_id')
            player_name = player.get('player_name', 'Unknown Player')
            rank = i + 1
            
            if not player_id:
                logger.error(f"No player ID found for guess rank {rank}: {player}")
                continue
            
            # Award shards
            success = bot_instance.award_shards(
                player_id,
                reward_amount,
                f"Daily Guess Leaderboard - Rank #{rank}"
            )
            
            if success:
                total_distributed += reward_amount
                # Add to notification list
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
        
        # Send notifications to winners
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
                
                # Try to send notification
                await bot_instance.application.bot.send_message(
                    chat_id=notification['player_id'],
                    text=message,
                    parse_mode='HTML'
                )
                
            except Exception as e:
                logger.error(f"Failed to send reward notification to {notification['player_name']}: {e}")
        
        # Log summary
        chase_winners = len([n for n in notifications if 'Chase' in n['game_type']])
        guess_winners = len([n for n in notifications if 'Guess' in n['game_type']])
        
        logger.info(f"Daily leaderboard rewards distributed successfully!")
        logger.info(f"Total shards distributed: {total_distributed}")
        logger.info(f"Chase winners: {chase_winners}, Guess winners: {guess_winners}")
        logger.info(f"Total notifications sent: {len(notifications)}")
        
        # Send admin log
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
    # Automatic daily reward distribution is disabled
    # Use /distributedailyrewards command to manually distribute rewards
    logger.info("Automatic daily reward distribution is disabled")
    logger.info("Use /distributedailyrewards command to manually distribute rewards to top 10 /dailylb users")
    
    # Keep the function running but do nothing
    import asyncio
    while True:
        await asyncio.sleep(86400)  # Sleep for 24 hours and do nothing

def check_banned(func):
    """Decorator to check if user is banned before executing command"""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        
        # Check if user is banned
        if bot_instance.is_banned(user_id):
            await update.message.reply_text(
                "🚫 <b>ACCESS RESTRICTED</b>\n\n"
                "Your access to this bot has been restricted.\n"
                "Contact administrators if you believe this is an error.",
                parse_mode='HTML'
            )
            return
        
        # User is not banned, proceed with command
        return await func(update, context, *args, **kwargs)
    
    return wrapper

# ====================================
# BOT COMMAND HANDLERS
# ====================================

# Bot command handlers
bot_instance = SPLAchievementBot()

@check_banned
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    
    # Add user to database when they start the bot - HTML escape user inputs
    username = H(user.username or "")
    display_name = H(user.full_name or user.first_name or f"User{user.id}")
    raw_display_name = user.full_name or user.first_name or f"User{user.id}"  # For DB storage
    
    success, is_new_user = bot_instance.create_or_update_player(user.id, user.username or "", raw_display_name)
    
    if success:
        logger.info(f"User {'registered' if is_new_user else 'updated'}: {raw_display_name} (@{user.username or ''}) - ID: {user.id}")
    
    # Different messages for new vs returning users
    if is_new_user:
        welcome_message = f"""
╭─────────────────────╮
│   🏏 SPL CRICKET BOT 🏆   │
╰─────────────────────╯

✨ <b>Welcome, {display_name}!</b> ✨

━━━━━━━━━━━━━━━━━━━━━━━━
🎉 <b>REGISTRATION COMPLETE</b> 🎊
━━━━━━━━━━━━━━━━━━━━━━━━

🎯 Ready to track your cricket achievements!
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

🚀 <b>Ready to dominate the cricket world?</b> 🚀
"""
    else:
        welcome_message = f"""
╭─────────────────────╮
│   🏏 SPL CRICKET BOT 🏆   │
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
💠 <b>/shardslb</b> - Rich list
📊 <b>/leaderboard</b> - Top players

🔥 <b>Time to reclaim your throne!</b> 🔥
"""
    
    welcome_message += """
━━━━━━━━━━━━━━━━━━━━━━━━
❤️ <b>Crafted with passion for cricket lovers</b> ❤️
━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    await update.message.reply_text(welcome_message, parse_mode='HTML')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show paginated help information based on user role."""
    user_id = update.effective_user.id
    is_super_admin = bot_instance.is_super_admin(user_id)
    is_admin = bot_instance.is_admin(user_id)
    
    # Create main help menu with categories
    keyboard = [
        [InlineKeyboardButton("🎮 Games", callback_data="help_games"),
         InlineKeyboardButton("💠 Shards", callback_data="help_shards")],
        [InlineKeyboardButton("🏆 Achievements", callback_data="help_achievements"),
         InlineKeyboardButton("📊 Stats", callback_data="help_stats")]
    ]
    
    # Add admin sections for admins
    if is_admin:
        keyboard.append([InlineKeyboardButton("👨‍💼 Admin Commands", callback_data="help_admin")])
    
    if is_super_admin:
        keyboard.append([InlineKeyboardButton("👑 Super Admin", callback_data="help_superadmin")])
    
    # Add quick command reference
    keyboard.append([InlineKeyboardButton("⚡ Quick Commands", callback_data="help_quick")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_message = f"""🏏 <b>SPL Achievement Bot Help</b> 🏆

Welcome to the ultimate cricket gaming experience! 
Choose a category below to explore:

🎮 <b>Games:</b> Chase, Guess, Nightmare Mode
💠 <b>Shards:</b> Currency system & rewards
🏆 <b>Achievements:</b> Unlock titles & bonuses
📊 <b>Stats:</b> Leaderboards & tracking
⚡ <b>Quick Commands:</b> Essential commands

👇 <b>Select a category to get started!</b>"""

    await safe_send(update.message.reply_text, welcome_message, parse_mode='HTML', reply_markup=reply_markup)

async def update_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show bot update information with user and admin sections"""
    user_id = update.effective_user.id
    is_admin = bot_instance.is_admin(user_id)
    
    # Check if command is used in a group chat
    if update.effective_chat.type != 'private':
        await safe_send(update.message.reply_text, 
                       "📱 <b>Please use this command in my DM!</b>\n\n"
                       "🔒 The /update command is only available in private chat for better readability.\n\n"
                       "👉 Click here to start: @SPLAchievementBot", 
                       parse_mode='HTML')
        return
    
    # Create keyboard with user and admin buttons
    keyboard = [
        [InlineKeyboardButton("👤 User Updates", callback_data="update_user")]
    ]
    
    # Add admin button if user is admin
    if is_admin:
        keyboard.append([InlineKeyboardButton("👨‍💼 Admin Updates", callback_data="update_admin")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Dynamic date
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    
    welcome_message = f"""🎉 <b>SPL BOT - LATEST UPDATE!</b> 🚀

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
    
    # Back button
    back_keyboard = [
        [InlineKeyboardButton("👤 User Updates", callback_data="update_user")]
    ]
    if is_admin:
        back_keyboard.append([InlineKeyboardButton("👨‍💼 Admin Updates", callback_data="update_admin")])
    back_keyboard.append([InlineKeyboardButton("🔙 Back to Menu", callback_data="update_main")])
    
    if data == "update_main":
        # Return to main update menu
        keyboard = [
            [InlineKeyboardButton("👤 User Updates", callback_data="update_user")]
        ]
        if is_admin:
            keyboard.append([InlineKeyboardButton("👨‍💼 Admin Updates", callback_data="update_admin")])
        
        # Dynamic date
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")
        
        message = f"""🎉 <b>SPL BOT - LATEST UPDATE!</b> 🚀

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 <b>Update Date:</b> {current_date}
🔄 <b>Version:</b> 3.1.0 - Enhanced Experience

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
<b>Your new digital wallet for the SPL ecosystem!</b>

<b>💰 How to Earn Shards:</b>
• 🏏 <b>Chase Games:</b> 15-30 shards per game
• 🎯 <b>Guess Games:</b> 12-24 shards per game  
• 🌙 <b>Nightmare Mode:</b> 10,000 shards for victory!
• 🎁 <b>Daily Bonus:</b> 50 base + streak bonuses
• 🏆 <b>Achievements:</b> Various shard bonuses

<b>💸 Shard Commands:</b>
• <code>/balance</code> - Check your shard wallet
• <code>/dailyreward</code> - Claim daily bonus (50+ shards)
• <code>/shardslb</code> - View top shard holders
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

async def help_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle help menu button callbacks"""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    is_super_admin = bot_instance.is_super_admin(user_id)
    is_admin = bot_instance.is_admin(user_id)
    
    data = query.data
    
    # Back button for all pages
    back_button = [InlineKeyboardButton("🔙 Back to Menu", callback_data="help_main")]
    
    if data == "help_main":
        # Return to main menu
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
        
        message = f"""🏏 <b>SPL Achievement Bot Help</b> 🏆

Welcome to the ultimate cricket gaming experience! 
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
• /leaderboard - View top 10 chase players
• Hand cricket rules: same number = wicket!
• 🍀 15% luck factor to escape wickets
• ⚡ 10 second cooldown between games
• 💠 Earn 30-90 shards per game

<b>🎲 GUESS GAME:</b>
• /guess - Play Guess the Number game
• /guessleaderboard - View guess game rankings
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
• /achievements - View your achievements
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
• Build your cricket reputation
• Track your journey in /profile"""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup([back_button]))
    
    elif data == "help_stats":
        message = f"""📊 <b>Statistics & Leaderboards</b> 📈

<b>🏏 CHASE STATISTICS:</b>
• /leaderboard - Top 10 chase players
• /chasestats - Your personal & global chase stats
• Track your best scores and levels

<b>🎲 GUESS STATISTICS:</b>
• /guessleaderboard - Guess game rankings
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
• /chase - Cricket game
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
• /achievements - Your achievements
• /profile - Complete profile
• /start - Register/update profile

<b>📊 LEADERBOARDS:</b>
• /leaderboard - Chase top 10
• /guessleaderboard - Guess rankings
• /dailylb - Daily competitions

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
• /addachievement @user Achievement
• /removeachievement @user Achievement
• /bulkward "Achievement" @user1 @user2
• /settitle @user "Title"
• /emojis - Achievement emoji guide

<b>💠 SHARD MANAGEMENT:</b>
• /giveshards @user amount [reason]
• /removeshards @user amount [reason]

<b>🎮 GAME ADMINISTRATION:</b>
• /cleanupchase - Force cleanup chase games
• /cleanupguess - Force cleanup guess games

<b>📊 DAILY LEADERBOARDS:</b>
• /distributedailyrewards - Trigger daily rewards
• /resetdailylb CONFIRM - Reset daily leaderboard
• /dailylbstats - Daily leaderboard statistics

<b>🎯 SPECIAL FEATURES:</b>
• /confirmachievement [ID] [notes]
• /listpending - Pending confirmations
• /broadcast message - Broadcast to all

<b>📝 EXAMPLES:</b>
• /addachievement @john Winner
• /giveshards @player 500 Tournament winner
• /settitle @mike "Captain\""""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup([back_button]))
    
    elif data == "help_superadmin" and is_super_admin:
        message = f"""👑 <b>Super Admin Commands</b> ⚡

<b>👥 ADMIN MANAGEMENT:</b>
• /addadmin @user - Add new admin
• /rmadmin @user - Remove admin
• /aadmins - List all admins

<b>🗄️ DATABASE MANAGEMENT:</b>
• /resetall CONFIRM - Reset entire database
• /resetplayer @user - Reset player data
• /listplayers - View all players
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

# GOAT Command Cricket Roast Lines - Cricket Themed Roasts
GOAT_ROAST_LINES = [
    # 🤬 Pure Gali + Savage Roast 🚨

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

# Global game state management
ACTIVE_CHASE_GAMES = {}  # user_id -> game_data
MAX_CONCURRENT_GAMES = 3  # Max games per user
GAME_TIMEOUT = 1800  # 30 minutes timeout

# Cache the keyboard to avoid recreating it
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

    # Simple format
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
    
    # Progress bar
    progress = (score / target) if target > 0 else 0
    filled = int(progress * 10)
    bar = "▓" * filled + "░" * (10 - filled)
    message += f"📈 <b>Progress:</b> {bar} {int(progress * 100)}%\n"
    
    if last_event:
        message += f"\n⚡ <b>Last ball:</b> {last_event}\n"
    
    message += f"\n🎯 <b>Choose:</b> 1 • 2 • 3 • 4 • 6 • OUT"
    
    return message

def _reset_level_state(user_id: int, user_name: str, level: int) -> dict:
    # Get wickets from LEVELS configuration
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
        
        # Create roast_rotation table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS roast_rotation (
                id SERIAL PRIMARY KEY,
                roast_line TEXT NOT NULL UNIQUE,
                last_used_date DATE,
                usage_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create roast_usage table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS roast_usage (
                id SERIAL PRIMARY KEY,
                roast_line TEXT NOT NULL,
                player_id INTEGER REFERENCES players(id),
                used_date DATE NOT NULL DEFAULT CURRENT_DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert all roast lines into rotation table (ignore duplicates)
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
        
        # Find roast lines that haven't been used yet
        cursor.execute("""
            SELECT roast_line FROM roast_rotation 
            WHERE usage_count = 0 
            ORDER BY RANDOM() 
            LIMIT 1
        """)
        
        result = cursor.fetchone()
        if result:
            return result[0]
        
        # If all lines have been used, find the least used ones
        cursor.execute("""
            SELECT roast_line FROM roast_rotation 
            WHERE usage_count = (SELECT MIN(usage_count) FROM roast_rotation)
            ORDER BY RANDOM() 
            LIMIT 1
        """)
        
        result = cursor.fetchone()
        if result:
            return result[0]
        
        # Fallback to random choice (shouldn't happen)
        return random.choice(GOAT_ROAST_LINES)
        
    except Exception as e:
        logger.error(f"Error getting next roast line: {e}")
        return random.choice(GOAT_ROAST_LINES)


def update_roast_usage(conn, roast_line, player_id):
    """Update roast usage statistics."""
    try:
        cursor = conn.cursor()
        
        # Update rotation table
        cursor.execute("""
            UPDATE roast_rotation 
            SET usage_count = usage_count + 1, last_used_date = CURRENT_DATE 
            WHERE roast_line = %s
        """, (roast_line,))
        
        # Insert into usage history
        cursor.execute("""
            INSERT INTO roast_usage (roast_line, player_id, used_date) 
            VALUES (%s, %s, CURRENT_DATE)
        """, (roast_line, player_id))
        
        conn.commit()
        
    except Exception as e:
        logger.error(f"Error updating roast usage: {e}")
        conn.rollback()


async def quit_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Quit all active games and show current status"""
    user = update.effective_user
    user_id = user.id
    
    # Get all active games
    active_games = bot_instance.get_all_active_games(user_id)
    
    # Count total games
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
    
    # Show current status before quitting
    message = f"🚪 <b>Quitting {total_active} Active Game{'s' if total_active > 1 else ''}</b>\n\n"
    
    games_quit = []
    
    # End guess game
    if active_games['guess']:
        bot_instance.end_guess_game(user_id, 'quit')
        g = active_games['guess']
        games_quit.append(f"🎯 <b>Guess Game</b> ({g['difficulty']}) - {g['attempts_left']} attempts left")
    
    # End nightmare game
    if active_games['nightmare']:
        # Find nightmare game and end it
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
    
    # End chase games
    for i, chase in enumerate(active_games['chase']):
        # Find and end chase game
        for key, game_state in list(ACTIVE_CHASE_GAMES.items()):
            if game_state.get('player_id') == user_id:
                ACTIVE_CHASE_GAMES.pop(key, None)
                break
        
        games_quit.append(f"🏏 <b>Chase Game #{i+1}</b> - Level {chase['level']}, {chase['score']}")
    
    # Build message
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


async def goat_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Select and announce the GOAT (Greatest Of All Time) player of the day."""
    today = date.today()
    
    conn = bot_instance.get_db_connection()
    if not conn:
        await update.message.reply_text("❌ <b>Database error!</b>\n\nPlease try again later.", parse_mode='HTML')
        return
    
    try:
        cursor = conn.cursor()
        
        # Ensure daily_goat table exists with correct schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_goat (
                id SERIAL PRIMARY KEY,
                date DATE UNIQUE NOT NULL,
                player_id INTEGER REFERENCES players(id) ON DELETE CASCADE,
                roast_line TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Check if roast_line column exists, add it if missing
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'daily_goat' AND column_name = 'roast_line'
        """)
        
        if not cursor.fetchone():
            # Add roast_line column if it doesn't exist
            cursor.execute("ALTER TABLE daily_goat ADD COLUMN roast_line TEXT")
            logger.info("Added missing roast_line column to daily_goat table")
        
        # Check if today's GOAT already picked
        cursor.execute("SELECT player_id, roast_line FROM daily_goat WHERE date = %s", (today,))
        result = cursor.fetchone()
        
        if result:
            # GOAT already selected for today
            player_id, stored_roast = result
            # Ensure we have a valid roast line
            if not stored_roast:
                stored_roast = bot_instance.get_cached_roast_line()
                # Update the database with the new roast line
                cursor.execute(
                    "UPDATE daily_goat SET roast_line = %s WHERE date = %s",
                    (stored_roast, today)
                )
        else:
            # Pick a random player (avoid yesterday's GOAT if possible)
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
                # If all players were yesterday's GOAT, just pick any random player
                cursor.execute("SELECT id FROM players ORDER BY RANDOM() LIMIT 1")
                player_result = cursor.fetchone()
                
            if not player_result:
                await update.message.reply_text("❌ <b>No players found!</b>\n\nPlease register some players first.", parse_mode='HTML')
                return
                
            player_id = player_result[0]
            
            # Initialize roast rotation system
            initialize_roast_rotation(conn)
            
            # Select next roast line using cached system for better performance
            stored_roast = bot_instance.get_cached_roast_line()
            
            # Update roast usage asynchronously to avoid blocking
            bot_instance.update_roast_usage_async(stored_roast)
            
            # Also update the database synchronously for the usage table
            update_roast_usage(conn, stored_roast, player_id)
            
            # Store in daily_goat table
            cursor.execute(
                "INSERT INTO daily_goat (date, player_id, roast_line) VALUES (%s, %s, %s)",
                (today, player_id, stored_roast)
            )
            conn.commit()
            
            # Award GOAT bonus shards
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
        
        # Get player details
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
        
        # Get player achievements
        achievements = bot_instance.get_player_achievements(player_id)
        
        # Build the enhanced GOAT message with better UI
        date_str = today.strftime("%B %d, %Y")
        
        # Create user mention if possible
        player_mention = H(player_dict['display_name'])
        if player_dict.get('telegram_id'):
            try:
                player_mention = f'<a href="tg://user?id={player_dict["telegram_id"]}">{H(player_dict["display_name"])}</a>'
            except:
                pass
        
        if achievements and len(achievements) > 0:
            # Player has achievements - show them as GOAT with enhanced UI
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
            # Player has title but no achievements - show title instead of roast
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
            # No achievements - use cricket roast with enhanced UI
            # Ensure we have a valid roast line
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

        
        # Send the message (removed pin functionality)
        await safe_send(update.message.reply_text, message, parse_mode='HTML')
        logger.info(f"GOAT announcement sent for {player_dict['display_name']}")
    
    except Exception as e:
        log_exception("goat_command", e, update.effective_user.id)
        await safe_send(update.message.reply_text, 
                       ERROR_MESSAGES['generic'], 
                       parse_mode='HTML')
    finally:
        bot_instance.return_db_connection(conn)

async def my_roast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show all roast lines used for the calling player with dates in a table format."""
    user_id = update.effective_user.id
    
    conn = bot_instance.get_db_connection()
    if not conn:
        await update.message.reply_text("❌ <b>Database error!</b>\n\nPlease try again later.", parse_mode='HTML')
        return
    
    try:
        cursor = conn.cursor()
        
        # Get player info
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
        
        # Get all roast history for this player
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
        
        # Create formatted table
        message = f"🎭 <b>{display_name}'s Roast Collection</b> 🎭\n\n"
        message += "📋 <b>Your Complete Roast History:</b>\n"
        message += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        # Add roasts with numbers and dates
        for idx, (roast_line, used_date) in enumerate(roast_history, 1):
            # Format the roast line with player name
            formatted_roast = roast_line.format(name=display_name)
            date_str = used_date.strftime("%d/%m/%Y") if used_date else "Unknown"
            
            message += f"<b>#{idx}</b> 📅 <code>{date_str}</code>\n"
            message += f"🎯 {formatted_roast}\n\n"
        
        # Add statistics
        message += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        message += f"📊 <b>Statistics:</b>\n"
        message += f"🎭 Total Roasts: <b>{len(roast_history)}</b>\n"
        
        if roast_history:
            latest_date = roast_history[0][1].strftime("%d/%m/%Y") if roast_history[0][1] else "Unknown"
            message += f"📅 Latest Roast: <b>{latest_date}</b>\n"
        
        # Enhanced message splitting with safe boundaries and HTML preservation
        MAX_MESSAGE_LENGTH = 4090  # Leave buffer for safety
        
        if len(message) > MAX_MESSAGE_LENGTH:
            messages = split_message_safely(message, MAX_MESSAGE_LENGTH)
            
            # Send all messages with delays
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

# Global state with automatic cleanup
GAME_EXPIRY_TIME = 1800  # 30 minutes

def cleanup_expired_games() -> int:
    """Clean up expired/inactive chase games to prevent memory leaks."""
    now = time.time()
    expired = []
    
    for key, state in list(ACTIVE_CHASE_GAMES.items()):
        # Don't cleanup if game is not active
        if not state.get("active", True):
            expired.append(key)
            continue
            
        # Don't cleanup if game was recently accessed
        last_action = state.get("last_action_time", 0)
        if now - last_action > GAME_TIMEOUT:
            # Only cleanup if game is truly stale (no recent activity)
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

    # Check for too many concurrent games
    active_games = get_user_active_games(user.id)
    if active_games >= MAX_CONCURRENT_GAMES:
        # Create detailed game status message
        transition_msg, _ = bot_instance.create_game_switch_message(user.id, "Chase Game")
        enhanced_msg = f"⚠️ <b>Maximum Games Reached!</b>\n\n{transition_msg}\n\n⏰ Games auto-expire after 10 minutes of inactivity."
        await update.message.reply_text(enhanced_msg, parse_mode="HTML")
        return

    # Force cleanup of any existing game for this user in this chat
    game_key = f"{user.id}_{update.effective_chat.id}"
    
    # More aggressive cleanup - check both global and user data
    existing_global_game = ACTIVE_CHASE_GAMES.get(game_key)
    existing_user_game = context.user_data.get("chase")
    
    if existing_global_game or existing_user_game:
        # Properly end the existing game before starting a new one
        existing_state = existing_global_game or existing_user_game
        if existing_state and existing_state.get('active'):
            logger.info(f"Ending existing active game for user {user.id} before starting new one")
            end_chase_game(existing_state, context, 'abandoned')
        else:
            # Clean up both locations if game wasn't active
            ACTIVE_CHASE_GAMES.pop(game_key, None)
            context.user_data.pop("chase", None)
            logger.info(f"Force cleaned up inactive game data for user {user.id}")

    # Start at Level 1 - use raw_name for storage, name for display
    state = _reset_level_state(user.id, raw_name, level=1)
    state["last_action_time"] = current_time
    context.user_data["chase"] = state
    
    # Register in global state tracking
    ACTIVE_CHASE_GAMES[game_key] = state

    # Send first card with async optimization
    try:
        text = _format_chase_card(state, last_event=None)
        sent = await update.message.reply_text(
            text, parse_mode="HTML", reply_markup=_chase_keyboard()
        )

        # Store message/chat info
        state["message_id"] = sent.message_id
        state["chat_id"] = sent.chat_id
        
        logger.info(f"Started chase game for user {user.id} at level 1")
        
    except Exception as e:
        # Cleanup on failure
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
            
        # Record game in database
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
                
                # Award shards for game completion
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
                            # Store shard reward in state for display
                            state['shard_reward'] = shard_reward
                        else:
                            logger.warning(f"Failed to award shards to player {state['player_id']}")
                
                except Exception as e:
                    logger.error(f"Error awarding chase game shards: {e}")
                
                # Update daily leaderboard
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
        
        # Clean up global tracking
        game_key = f"{state['player_id']}_{state.get('chat_id', 0)}"
        ACTIVE_CHASE_GAMES.pop(game_key, None)
        
        # Clean up user data
        context.user_data.pop("chase", None)
        
        # Mark as inactive
        state["active"] = False
        
        logger.info(f"Chase game ended: {game_outcome} for player {state.get('player_id')}")
        return True
        
    except Exception as e:
        logger.error(f"Error ending chase game: {e}")
        # Force cleanup even if recording fails
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
    
    # Immediate callback answer to prevent timeout
    await query.answer()
    
    # Safe game state check with proper error handling
    state = context.user_data.get("chase")
    if not state or not state.get("active"):
        # Check if there's a stale game in global tracking
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

    # Restrict actions to the original message  
    if query.message.message_id != state.get("message_id"):
        return

    # Owner-only guard with better error message
    if query.from_user.id != state["player_id"]:
        await query.answer("❌ This is not your game!", show_alert=True)
        return

    # Enhanced rate limiting with exponential backoff
    last_action = state.get("last_action_time", 0)
    time_diff = current_time - last_action
    
    # More aggressive rate limiting
    if time_diff < 1.5:  # Increased from 0.5 to 1.5 seconds
        return  # Silent ignore rapid clicks
    
    state["last_action_time"] = current_time

    # Parse player choice with error handling
    try:
        _, num_str = query.data.split(":")
        player_num = int(num_str)
        if player_num not in {1, 2, 3, 4, 5, 6}:
            return
    except Exception:
        return

    # Bot's random pick with LUCK FACTOR for easier gameplay
    bot_num = random.randint(1, 6)
    
    # 🍀 LUCK FACTOR: 15% chance to avoid wicket (make game easier)
    luck_roll = random.randint(1, 100)
    is_lucky = luck_roll <= 15  # 15% luck chance
    
    # If player and bot pick same number
    if player_num == bot_num:
        if is_lucky:
            # Lucky escape! Treat as normal run
            last = (
                f"⚫ <b>Ball {state['balls_used']}:</b> "
                f"You <b>{player_num}</b> | Bot <b>{bot_num}</b>\n"
                f"🍀 <b>LUCKY ESCAPE!</b> +{player_num} runs (saved by luck!)"
            )
            state["score"] += player_num
        else:
            # Normal wicket
            state["wickets_left"] -= 1
            last = (
                f"⚫ <b>Ball {state['balls_used']}:</b> "
                f"You <b>{player_num}</b> | Bot <b>{bot_num}</b>\n"
                f"💀 <b>WICKET!</b> Wickets left: {state['wickets_left']}"
            )
    else:
        # Different numbers = runs scored normally
        state["score"] += player_num
        last = (
            f"⚫ <b>Ball {state['balls_used']}:</b> "
            f"You <b>{player_num}</b> | Bot <b>{bot_num}</b>\n"
            f"✅ <b>+{player_num} runs</b>"
        )

    # Ball consumed
    state["balls_left"] -= 1
    state["balls_used"] += 1

    # Check if all wickets are lost (only if it was a real wicket, not lucky escape)
    if player_num == bot_num and not is_lucky and state["wickets_left"] <= 0:
        state["active"] = False
        
        # Calculate shard reward before ending game
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
        
        # Add shard reward info
        if shard_reward > 0:
            end += f"\n💠 <b>Shards Earned:</b> +{shard_reward}"
        
        end += f"\n━━━━━━━━━━━━━━━━━━━━\n💪 <i>Try again with /chase to improve!</i>"
        
        text = f"{_format_chase_card(state, last)}{end}"
        try:
            await query.edit_message_text(text, parse_mode="HTML")
        except Exception as e:
            logger.warning(f"Failed to edit message on game end: {e}")
        
        # Properly end the game to record it
        end_chase_game(state, context, 'lost')
        return
    
    # Check if wicket fell but still have wickets left
    if player_num == bot_num and not is_lucky and state["wickets_left"] > 0:
        # Still have wickets left, continue playing
        try:
            text = _format_chase_card(state, last)
            await query.edit_message_text(text, parse_mode="HTML", reply_markup=_chase_keyboard())
        except telegram_error.BadRequest as e:
            if "message is not modified" not in str(e).lower():
                logger.warning(f"Chase callback edit error: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error in chase callback: {e}")
        return

    # Win level? (check if target reached)
    if state["score"] >= state["target"]:
        if state["level"] == 10:
            # Calculate shard reward for complete victory
            score = state.get('score', 0)
            level = state.get('level', 1)
            won = True
            shard_reward = bot_instance.calculate_game_shard_reward('chase_game', score, level, won)
            
            # Use unified end function for complete victory
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

        # Advance to next level
        next_level = state["level"] + 1
        
        # Calculate shard reward for level completion
        score = state.get('score', 0)
        level = state.get('level', 1)
        won = True
        shard_reward = bot_instance.calculate_game_shard_reward('chase_game', score, level, won)
        
        # Get shard reward info
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

        # Reset for next level but keep same message/chat
        new_state = _reset_level_state(state["player_id"], state["player_name"], next_level)
        new_state["message_id"] = query.message.message_id
        new_state["chat_id"] = query.message.chat_id
        new_state["last_action_time"] = current_time
        new_state["start_time"] = state.get("start_time", time.time())  # Preserve original start time
        
        # Update both storage locations
        context.user_data["chase"] = new_state
        game_key = f"{state['player_id']}_{query.message.chat_id}"
        ACTIVE_CHASE_GAMES[game_key] = new_state

        try:
            await query.edit_message_text(text, parse_mode="HTML", reply_markup=_chase_keyboard())
        except Exception as e:
            logger.warning(f"Failed to edit message on level advance: {e}")
        return

    # Balls over & target not reached → game over
    if state["balls_left"] == 0 and state["score"] < state["target"]:
        # Calculate shard reward before ending game
        score = state.get('score', 0)
        level = state.get('level', 1)
        won = False  # Target not reached
        shard_reward = bot_instance.calculate_game_shard_reward('chase_game', score, level, won)
        
        # Use unified end function for game loss
        end_chase_game(state, context, 'lost')
        
        end = (
            f"\n\n💀 <b>BALLS FINISHED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Final Score:</b> {state['score']}/{state['target']}\n"
            f"🏏 <b>Balls Used:</b> {state['balls_used']}\n"
            f"🎯 <b>Target:</b> {state['target']}\n"
            f"📈 <b>Level Reached:</b> {state['level']}"
        )
        
        # Add shard reward info
        if shard_reward > 0:
            end += f"\n💠 <b>Shards Earned:</b> +{shard_reward}"
        
        end += f"\n━━━━━━━━━━━━━━━━━━━━\n💪 <i>Try again with /chase!</i>"
        
        try:
            await query.edit_message_text(f"{_format_chase_card(state, last)}{end}", parse_mode="HTML")
        except Exception as e:
            logger.warning(f"Failed to edit message on game over: {e}")
        return

    # Otherwise continue same level
    try:
        text = _format_chase_card(state, last)
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=_chase_keyboard())
    except telegram_error.BadRequest as e:
        # Handle message edit conflicts gracefully
        if "message is not modified" not in str(e).lower():
            logger.warning(f"Chase callback edit error: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error updating chase game: {e}")
        # Use unified end function for error cleanup
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

async def chase_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show chase game statistics - Public command with personal stats"""
    user = update.effective_user
    
    # Clean up expired games first
    cleanup_expired_games()
    
    # Get player's personal statistics
    player = bot_instance.get_player_by_telegram_id(user.id)
    if not player:
        await update.message.reply_text("❌ <b>Player not found!</b>\n\n� Use /start to register first.", parse_mode='HTML')
        return
    
    # Get personal chase stats  
    personal_stats = bot_instance.get_player_chase_stats(player['id'])
    
    # Get overall public statistics from database  
    overall_stats = bot_instance.get_chase_game_stats()
    
    message = (
        f"🏏 <b>CHASE GAME STATISTICS</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"👤 <b>YOUR STATS</b>\n"
        f"🎮 <b>Games Played:</b> {personal_stats.get('total_games', 0)}\n"
        f"🏆 <b>Games Won:</b> {personal_stats.get('games_won', 0)}\n"
        f"📈 <b>Win Rate:</b> {personal_stats.get('win_rate', 0):.1f}%\n"
        f"� <b>Best Score:</b> {personal_stats.get('best_score', 0)}\n"
        f"⚡ <b>Average Score:</b> {personal_stats.get('avg_score', 0):.1f}\n\n"
        f"� <b>GLOBAL STATS</b>\n"
        f"📊 <b>Total Games:</b> {overall_stats.get('total_games', 0)}\n"
        f"� <b>Total Players:</b> {overall_stats.get('total_players', 0)}\n"
        f"🏆 <b>High Score:</b> {overall_stats.get('high_score', 0)}\n"
        f"📈 <b>Average Score:</b> {overall_stats.get('avg_score', 0):.1f}\n"
        f"� <b>Games (24h):</b> {overall_stats.get('games_24h', 0)}\n"
        f"📅 <b>Games (7d):</b> {overall_stats.get('games_7d', 0)}\n\n"
        f"🎯 <b>Want to play?</b> Use /chase to start!"
    )
    
    await update.message.reply_text(message, parse_mode="HTML")

async def leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display the Chase Game Leaderboard - Modern Block Style"""
    try:
        # Fetch leaderboard data
        leaderboard = bot_instance.get_chase_leaderboard(10)

        if not leaderboard:
            await update.message.reply_text(
                "🏏 <b>CHASE LEADERBOARD</b> 🏏\n\n"
                "📊 No chase games played yet!\n"
                "🎮 Be the first to start with /chase",
                parse_mode='HTML'
            )
            return

        # Start message
        message = "🏏 <b>CHASE LEADERBOARD</b> 🏏\n"
        message += "━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        # Rank emojis for top 3, then numbers
        rank_emojis = {1: "🥇", 2: "🥈", 3: "🥉"}

        for i, player in enumerate(leaderboard, 1):
            if i in rank_emojis:
                rank_display = rank_emojis[i]
            else:
                rank_display = f"{i}."  # Simple number format
            
            # Sanitize player name to prevent HTML parsing issues
            name = html.escape(player.get('display_name', 'Unknown'))
            level = player.get('highest_level', 0) or 0
            runs = player.get('runs_scored', 0) or 0
            balls = player.get('balls_faced', 0) or 0  # Ensure not None
            strike_rate = player.get('strike_rate', 0.0) or 0.0

            # Only show strike rate if it's meaningful (> 0 and balls > 0)
            sr_display = f" | <b>SR:</b> {strike_rate:.0f}" if strike_rate > 0 and balls > 0 else ""

            # Build modern block for each player
            message += (
                f"{rank_display} <b>{name}</b>\n"
                f"   ┗ <b>Level:</b> {level} | "
                f"<b>Runs:</b> {runs}({balls}){sr_display}\n\n"
            )

        # Footer with rules
        message += "━━━━━━━━━━━━━━━━━━━━━━━\n"
        message += "📋 <b>Ranking Rules:</b>\n"
        message += "1️⃣ Highest Level\n"
        message += "2️⃣ Highest Runs at that Level\n"
        message += "3️⃣ Fewer Balls Faced\n"
        message += "━━━━━━━━━━━━━━━━━━━━━━━\n"
        message += "🎯 Play <b>/chase</b> & climb the ranks!"

        # Send the message
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
    
    # First, properly end all active games to record them
    games_to_end = list(ACTIVE_CHASE_GAMES.items())  # Create a copy to iterate over
    for game_key, game_state in games_to_end:
        try:
            if game_state and game_state.get('active'):
                # Create a mock context for the end_chase_game function
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
    
    # Clear any remaining game data
    ACTIVE_CHASE_GAMES.clear()
    
    # Enhanced feedback message
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

async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin command to broadcast messages to all registered users"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can broadcast messages!", parse_mode='HTML')
        return
    
    # Check if replying to a message
    if update.message.reply_to_message:
        # Use the replied message content
        reply_msg = update.message.reply_to_message
        
        # Handle different message types
        if reply_msg.text:
            broadcast_text = reply_msg.text
        elif reply_msg.caption:
            broadcast_text = reply_msg.caption
        else:
            await update.message.reply_text("❌ Can only broadcast text messages or media with captions!")
            return
            
        # Include media if present
        broadcast_media = None
        if reply_msg.photo:
            broadcast_media = {'type': 'photo', 'media': reply_msg.photo[-1].file_id}
        elif reply_msg.video:
            broadcast_media = {'type': 'video', 'media': reply_msg.video.file_id}
        elif reply_msg.document:
            broadcast_media = {'type': 'document', 'media': reply_msg.document.file_id}
            
    else:
        # Use command arguments
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
    
    # Get all registered users and active group chats
    try:
        with bot_instance.get_db_cursor() as cursor_result:
            if cursor_result is None:
                await update.message.reply_text("❌ Database error occurred!")
                return
            cursor, conn = cursor_result
            
            # Get private chat users (players)
            cursor.execute("SELECT telegram_id FROM players")
            users = cursor.fetchall()
            
            # Get active group chats (groups and supergroups)
            cursor.execute("""
                SELECT chat_id FROM chats 
                WHERE chat_type IN ('group', 'supergroup') 
                AND is_active = TRUE
            """)
            groups = cursor.fetchall()
            
            # Combine users and groups
            all_recipients = [(row[0], 'user') for row in users] + [(row[0], 'group') for row in groups]
            
            if not all_recipients:
                await update.message.reply_text("❌ No recipients found! (No users or groups)")
                return
            
    except Exception as e:
        logger.error(f"Database error in broadcast: {e}")
        await update.message.reply_text("❌ Database error occurred!")
        return
    
    # Send confirmation before broadcasting
    user_count = len(users)
    group_count = len(groups)
    total_recipients = len(all_recipients)
    preview_text = broadcast_text[:100] + "..." if len(broadcast_text) > 100 else broadcast_text
    
    # Create confirmation keyboard
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    keyboard = [
        [
            InlineKeyboardButton("✅ CONFIRM BROADCAST", callback_data=f"broadcast_confirm_{total_recipients}"),
            InlineKeyboardButton("❌ CANCEL", callback_data="broadcast_cancel")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Prepare media text outside f-string to avoid backslash issues
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
    
    # Store broadcast data temporarily for confirmation callback
    context.user_data['pending_broadcast'] = {
        'text': broadcast_text,
        'media': broadcast_media,
        'recipients': all_recipients,
        'user_count': user_count,
        'group_count': group_count,
        'confirm_msg_id': confirm_msg.message_id
    }
    
    return  # Wait for confirmation

async def draftbroadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Draft a broadcast message for later sending"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can draft broadcast messages!", parse_mode='HTML')
        return
    
    if not context.args:
        await update.message.reply_text(
            "📝 <b>DRAFT BROADCAST MESSAGE</b>\n\n"
            "<b>Usage:</b> /draftbroadcast &lt;message&gt;\n\n"
            "<b>Example:</b> /draftbroadcast Server maintenance scheduled for tomorrow\n\n"
            "💡 This will save your message as a draft without sending it.",
            parse_mode='HTML'
        )
        return
    
    draft_text = ' '.join(context.args)
    
    # Store draft message in user data
    context.user_data['draft_broadcast'] = {
        'text': draft_text,
        'created_at': update.message.date,
        'author': update.effective_user.full_name or update.effective_user.username
    }
    
    preview_text = draft_text[:100] + "..." if len(draft_text) > 100 else draft_text
    
    await update.message.reply_text(
        f"📝 <b>BROADCAST DRAFT SAVED!</b>\n\n"
        f"✅ Your message has been saved as a draft.\n\n"
        f"📄 <b>Draft Preview:</b>\n<code>{preview_text}</code>\n\n"
        f"💡 <b>Next Steps:</b>\n"
        f"• Use /testbroadcast to send to admins only\n"
        f"• Use /broadcast {draft_text} to send to all users\n"
        f"• Or draft a new message to replace this one",
        parse_mode='HTML'
    )

async def testbroadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send test broadcast to admins only"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can test broadcast messages!", parse_mode='HTML')
        return
    
    # Get message to test - either from args or draft
    test_message = None
    
    if context.args:
        test_message = ' '.join(context.args)
    elif 'draft_broadcast' in context.user_data:
        test_message = context.user_data['draft_broadcast']['text']
    else:
        await update.message.reply_text(
            "📝 <b>TEST BROADCAST</b>\n\n"
            "<b>Usage:</b>\n"
            "• /testbroadcast &lt;message&gt; - Test specific message\n"
            "• /testbroadcast - Test your saved draft\n\n"
            "<b>Example:</b> /testbroadcast This is a test announcement\n\n"
            "💡 This will send the message to all admins only.",
            parse_mode='HTML'
        )
        return
    
    try:
        # Get all admins
        admins = bot_instance.get_all_admins()
        admin_ids = [admin['telegram_id'] for admin in admins]
        
        # Add super admin
        admin_ids.append(bot_instance.super_admin_id)
        
        # Remove duplicates
        admin_ids = list(set(admin_ids))
        
        if not admin_ids:
            await update.message.reply_text("❌ No admins found to send test to!")
            return
        
        success_count = 0
        failed_count = 0
        
        # Send test message to all admins
        for admin_id in admin_ids:
            try:
                await update.get_bot().send_message(
                    chat_id=admin_id,
                    text=f"🧪 <b>TEST BROADCAST</b>\n\n{test_message}\n\n"
                         f"<i>This is a test message sent to admins only.</i>",
                    parse_mode='HTML'
                )
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to send test broadcast to admin {admin_id}: {e}")
                failed_count += 1
        
        preview_text = test_message[:50] + "..." if len(test_message) > 50 else test_message
        
        await update.message.reply_text(
            f"🧪 <b>TEST BROADCAST COMPLETE!</b>\n\n"
            f"✅ <b>Sent to:</b> {success_count} admins\n"
            f"❌ <b>Failed:</b> {failed_count}\n"
            f"📊 <b>Total Admins:</b> {len(admin_ids)}\n\n"
            f"📝 <b>Message:</b> {preview_text}\n\n"
            f"💡 If test looks good, use /broadcast to send to all users!",
            parse_mode='HTML'
        )
        
    except Exception as e:
        logger.error(f"Error in test broadcast: {e}")
        await update.message.reply_text("❌ Error sending test broadcast. Please try again.")

# Add broadcast confirmation callback handler
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
        # Get stored broadcast data
        broadcast_data = context.user_data.get('pending_broadcast')
        if not broadcast_data:
            await query.edit_message_text("❌ Broadcast data not found. Please try again.")
            return
        
        broadcast_text = broadcast_data['text']
        broadcast_media = broadcast_data['media']
        all_recipients = broadcast_data['recipients']
        user_count = broadcast_data['user_count']
        group_count = broadcast_data['group_count']
        
        # Update message to show broadcasting
        await query.edit_message_text(
            f"📢 <b>BROADCASTING...</b>\n\n"
            f"👥 Users: {user_count} | 👨‍👩‍👧‍👦 Groups: {group_count}\n"
            f"📊 Total: {len(all_recipients)} recipients\n\n"
            f"🔄 <i>Sending messages...</i>",
            parse_mode='HTML'
        )
    
    # Broadcast to all recipients (users and groups)
    success_count = 0
    failed_count = 0
    blocked_count = 0
    user_success = 0
    group_success = 0
    error_details = {}
    
    for chat_id, chat_type in all_recipients:
        try:
            if broadcast_media:
                # Send media with caption
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
                # Send text message
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
            # User blocked the bot or chat not found
            if "blocked" in str(e).lower() or "forbidden" in str(e).lower():
                blocked_count += 1
                logger.info(f"Chat {chat_id} blocked the bot")
            else:
                failed_count += 1
                error_type = "Forbidden Access"
                error_details[error_type] = error_details.get(error_type, 0) + 1
                
        except telegram_error.BadRequest as e:
            # Invalid user ID or chat not found
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
            # Enhanced rate limiting handling
            failed_count += 1
            error_type = "Rate Limited"
            error_details[error_type] = error_details.get(error_type, 0) + 1
            logger.warning(f"Rate limited, retry after {e.retry_after}s")
            
            # Respect Telegram's retry_after and add buffer
            await asyncio.sleep(e.retry_after + 1)
            
        except Exception as e:
            # Other errors
            failed_count += 1
            error_type = f"Other: {type(e).__name__}"
            error_details[error_type] = error_details.get(error_type, 0) + 1
            logger.warning(f"Failed to broadcast to chat {chat_id}: {e}")
            
        # Enhanced throttling based on batch size
        batch_size = 20  # Messages per batch
        if (success_count + failed_count) % batch_size == 0:
            # Longer pause every 20 messages
            await asyncio.sleep(1.0)
        else:
            # Small delay between individual messages
            await asyncio.sleep(0.1)
    
    # Update the confirmation message with detailed results
    total_recipients = len(all_recipients)
    preview_text = broadcast_text[:50] + "..." if len(broadcast_text) > 50 else broadcast_text
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
    
    # Clean up stored data
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
    
    # Find the user/player
    player = bot_instance.find_player_by_identifier(target_identifier)
    
    if not player:
        await update.message.reply_text(
            f"❌ <b>USER NOT FOUND!</b>\n\n"
            f"🔍 Could not find: {target_identifier}\n\n"
            f"💡 User must have interacted with the bot first!",
            parse_mode='HTML'
        )
        return
    
    # Add as admin
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
    
    # Find the admin
    player = bot_instance.find_player_by_identifier(target_identifier)
    
    if not player:
        await update.message.reply_text(f"❌ <b>USER NOT FOUND!</b> {target_identifier}", parse_mode='HTML')
        return
    
    # Check if trying to remove super admin
    if player['telegram_id'] == bot_instance.super_admin_id:
        await update.message.reply_text("❌ <b>CANNOT REMOVE SUPER ADMIN!</b> 👑", parse_mode='HTML')
        return
    
    # Remove admin
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
    
    message = "🛡️ <b>SPL BOT ADMIN LIST</b> 👑\n\n"
    message += "━━━━━━━━━━━━━━━━━━━━\n"
    
    # Show Super Admin first
    message += f"👑 <b>SUPER ADMIN (Creator)</b>\n"
    message += f"🆔 ID: {bot_instance.super_admin_id}\n\n"
    
    # Show database admins
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
    
    # Parse the message text to handle quoted achievement names
    message_text = update.message.text
    command_parts = message_text.split(maxsplit=1)  # Split only once to get command and rest
    
    if len(command_parts) < 2:
        await update.message.reply_text("❌ <b>No arguments provided!</b>", parse_mode='HTML')
        return
    
    args_text = command_parts[1]  # Everything after /bulkward
    
    # Check if achievement is quoted
    if args_text.startswith('"'):
        # Find the closing quote
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
        # No quotes - use first word as achievement, rest as players
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
        # Handle both @username and user ID formats
        player = bot_instance.find_player_by_identifier(identifier)
        if player and bot_instance.add_achievement(player['id'], achievement, update.effective_user.id, player.get('username')):
            successful.append(player['display_name'])
        elif not player:
            # User not registered
            not_registered.append(identifier)
        else:
            # Achievement add failed for registered user
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
    
    # Find player
    player = bot_instance.find_player_by_identifier(player_identifier)
    
    if not player:
        # Player not found - show user not registered message
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
    
    # Add achievement to registered player
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
    
    # Find player
    player = bot_instance.find_player_by_identifier(player_identifier)
    
    if not player:
        await update.message.reply_text(
            f"❌ <b>PLAYER NOT FOUND!</b>\n\n"
            f"🔍 Could not find: {player_identifier}",
            parse_mode='HTML'
        )
        return
    
    # Get current achievement count before removing
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
    
    # Remove achievement
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
    
    # Find player
    player = bot_instance.find_player_by_identifier(player_identifier)
    
    if not player:
        await update.message.reply_text(
            f"❌ **PLAYER NOT FOUND!**\n\n"
            f"🔍 Could not find: `{player_identifier}`",
            parse_mode='Markdown'
        )
        return
    
    # Set title
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
        await update.message.reply_text(
            "❌ <b>ACCESS DENIED!</b>\n\n"
            "🛡️ This command requires administrator privileges.",
            parse_mode='HTML'
        )
        return

    if len(context.args) < 1:
        await update.message.reply_text(
            "❌ <b>USAGE ERROR!</b>\n\n"
            "<b>Usage:</b> /rmtitle @username\n"
            "<b>Example:</b> /rmtitle @player123\n\n"
            "💡 This will remove the custom title from the specified player.",
            parse_mode='HTML'
        )
        return

    player_identifier = context.args[0]
    
    # Find player
    player = bot_instance.find_player_by_identifier(player_identifier)
    
    if not player:
        await update.message.reply_text(
            f"❌ <b>PLAYER NOT FOUND!</b>\n\n"
            f"🔍 Player <b>{player_identifier}</b> is not registered.\n"
            f"💡 Players must use the bot at least once to be registered.",
            parse_mode='HTML'
        )
        return

    # Remove title (set to empty string)
    if bot_instance.set_player_title(player['id'], ""):
        await update.message.reply_text(
            f"🗑️ <b>TITLE REMOVED SUCCESSFULLY!</b> 🗑️\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"👤 <b>Player:</b> {player['display_name']}\n"
            f"🔄 <b>Action:</b> Title removed\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"✅ <b>Player no longer has a custom title!</b>",
            parse_mode='HTML'
        )
    else:
        await update.message.reply_text("❌ <b>FAILED!</b> Unable to remove title. Please try again.", parse_mode='HTML')

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
    
    # Find player
    player = bot_instance.find_player_by_identifier(player_identifier)
    
    if not player:
        await update.message.reply_text(
            f"❌ <b>Player not found!</b>\n\n"
            f"🔍 Could not find player: <code>{player_identifier}</code>\n\n"
            f"💡 Make sure they have started the bot first!",
            parse_mode='HTML'
        )
        return

    # Remove title (set to empty string)
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
    
    # Find player
    player = bot_instance.find_player_by_identifier(user_identifier)
    
    if not player:
        await update.message.reply_text(
            f"❌ <b>User not found!</b>\n\n"
            f"🔍 Could not find user: <code>{user_identifier}</code>\n\n"
            f"💡 Make sure they have started the bot first!",
            parse_mode='HTML'
        )
        return

    # Get additional info
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
    """Ban user (Admin only) - Placeholder implementation"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can ban users!", parse_mode='HTML')
        return

    await update.message.reply_text(
        "🚫 <b>USER BAN SYSTEM</b>\n\n"
        "⚠️ <b>Feature Coming Soon!</b>\n\n"
        "This feature will allow admins to:\n"
        "• Ban users from using the bot\n"
        "• Set ban reasons and durations\n"
        "• Track ban history\n\n"
        "💡 Currently in development for safety.",
        parse_mode='HTML'
    )

async def unbanuser_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Unban user (Admin only) - Placeholder implementation"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can unban users!", parse_mode='HTML')
        return

    await update.message.reply_text(
        "✅ <b>USER UNBAN SYSTEM</b>\n\n"
        "⚠️ <b>Feature Coming Soon!</b>\n\n"
        "This feature will allow admins to:\n"
        "• Remove user bans\n"
        "• Restore user access\n"
        "• View ban history\n\n"
        "💡 Currently in development for safety.",
        parse_mode='HTML'
    )

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
        # Clear various caches
        cache_cleared = 0
        
        # Profile cache
        if hasattr(bot_instance, 'profile_cache') and bot_instance.profile_cache['data']:
            profile_count = len(bot_instance.profile_cache['data'])
            bot_instance.profile_cache['data'].clear()
            bot_instance.profile_cache['last_updated'].clear()
            cache_cleared += profile_count
        
        # Roast cache
        if hasattr(bot_instance, 'roast_cache') and bot_instance.roast_cache:
            roast_count = len(bot_instance.roast_cache)
            bot_instance.roast_cache.clear()
            bot_instance.roast_rotation_index = 0
            cache_cleared += roast_count
        
        # GOAT cache (if exists)
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
        
        # Check if viewing specific user transactions
        if context.args and len(context.args) > 0:
            user_identifier = context.args[0]
            player = bot_instance.find_player_by_identifier(user_identifier)
            
            if not player:
                await update.message.reply_text(f"❌ User not found: {user_identifier}")
                return
            
            # Get user-specific transactions
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
            # Get recent global transactions
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
        
        # Format transactions
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


async def achievements_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """View your own achievements"""
    user = update.effective_user
    
    # Create or update player record
    if not bot_instance.create_or_update_player(
        user.id, 
        user.username or "", 
        user.full_name or user.username or f"User {user.id}"
    ):
        await update.message.reply_text("❌ **ERROR!** Failed to load your profile. Please try again.")
        return
    
    # Find player
    player = bot_instance.find_player_by_identifier(str(user.id))
    
    if not player:
        await update.message.reply_text("❌ **ERROR!** Failed to load your profile. Please contact admin.")
        return
    
    # Get achievements
    achievements = bot_instance.get_player_achievements(player['id'])
    
    # Format and send message
    message = bot_instance.format_achievements_message(player, achievements)
    await update.message.reply_text(message, parse_mode='HTML')

async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """View your comprehensive player profile with caching"""
    user = update.effective_user
    
    try:
        # Create or update player record
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
        
        # Get cached profile data
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
        
        # Build profile message with title positioned correctly
        message = f"👤 <b>PLAYER PROFILE</b> 🏏\n"
        
        # Show title immediately after header if exists and not empty
        if player.get('title') and player['title'].strip():
            message += f"👑 <b>{H(player['title'])}</b>\n"
        
        message += "\n━━━━━━━━━━━━━━━━━━━━\n"
        message += f"👤 <b>Name:</b> {H(player['display_name'])}\n"
        
        if player.get('username'):
            message += f"🆔 <b>Username:</b> @{player['username']}\n"
        
        # Format join date
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
        
        # Chase Statistics Section
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
        
        # Guess Game Statistics Section
        guess_stats = bot_instance.get_guess_game_stats(user.id)
        
        # Get player's scores and unlocked levels
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
        
        # Footer with helpful tips
        message += f"💡 <b>QUICK TIPS</b>\n"
        message += "━━━━━━━━━━━━━━━━━━━━\n"
        message += "🎮 <b>Commands:</b> /chase, /guess, /dailyguess\n"
        message += "🔍 <b>View:</b> /achievements, /shards, /leaderboard\n"
        message += "📊 <b>Stats:</b> /chasestats, /guessstats\n"
        message += "━━━━━━━━━━━━━━━━━━━━"
        
        await update.message.reply_text(message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Profile command error for user {user.id}: {e}")
        await update.message.reply_text(
            "❌ <b>ERROR!</b> Failed to load your profile. Please try again later.",
            parse_mode='HTML'
        )

async def shardslb_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
        
        # Build leaderboard message
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
            
            # Add rich user indicator for high balances
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
        
        # Add economy summary
        total_circulation = shard_stats.get('total_circulation', 0)
        total_earned = shard_stats.get('total_earned', 0)
        
        message += f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        message += f"📊 <b>ECONOMY SUMMARY</b>\n"
        message += f"• <b>Total Circulation:</b> {total_circulation:,} 💠\n"
        message += f"• <b>Total Ever Earned:</b> {total_earned:,} 💠\n\n"
        message += f"💡 <b>Earn shards by playing games!</b>"
        
        await update.message.reply_text(message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Error in shardslb command: {e}")
        await update.message.reply_text("❌ Error getting shard leaderboard!", parse_mode='HTML')

async def adminpanel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Comprehensive Super Admin Control Panel"""
    if not bot_instance.is_super_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n👑 Only Super Admin can access the admin panel!", parse_mode='HTML')
        return
    
    # Main admin panel with categorized buttons
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
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    message = f"""👑 <b>SUPER ADMIN CONTROL PANEL</b> 🚀

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 <b>SPL Achievement Bot Management</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>Welcome, Super Admin!</b> 
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

    await update.message.reply_text(message, parse_mode='HTML', reply_markup=reply_markup)

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
        # Return to main panel
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
🤖 <b>SPL Achievement Bot Management</b>
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
        # Bot Statistics Panel
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
        # Economy Control Panel
        keyboard = [
            [InlineKeyboardButton("💰 Give Shards", callback_data="action_giveshards"),
             InlineKeyboardButton("🗑️ Remove Shards", callback_data="action_removeshards")],
            [InlineKeyboardButton("📊 Shard Leaderboard", callback_data="action_shardslb"),
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
        # Game Management Panel
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
    
    # Continue with other panels...
    elif data == "panel_admins":
        # Admin Control Panel
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
• List: /aadmins"""
        
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))
    
    elif data == "panel_users":
        # User Management Panel
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
        # Broadcasting Panel
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
        # Achievement System Panel
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
    
    elif data == "panel_system":
        # System Tools Panel
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
    
    # Action handlers for admin panel buttons
    elif data == "action_shardstats":
        # Shard Economy Statistics
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
    
    elif data == "action_shardslb":
        # Shard Leaderboard
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

<b>Usage:</b> /cleanup chase
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
            # Get active games count
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
    
    # New action handlers for additional panels
    elif data == "action_listusers":
        try:
            players = bot_instance.get_all_players()
            if players:
                message = f"👥 <b>REGISTERED USERS ({len(players)} total)</b>\n\n"
                for i, player in enumerate(players[:20], 1):  # Show first 20
                    message += f"{i}. <b>{player['display_name']}</b>"
                    if player.get('username'):
                        message += f" (@{player['username']})"
                    message += f"\n   ID: {player['telegram_id']}\n"
                    if player.get('title'):
                        message += f"   Title: {player['title']}\n"
                    message += "\n"
                
                if len(players) > 20:
                    message += f"... and {len(players) - 20} more users"
            else:
                message = "👥 No users registered yet."
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
        # Unknown action
        message = "❓ <b>Unknown Action</b>\n\nThis feature is not yet implemented."
        keyboard = [back_button]
        await query.edit_message_text(message, parse_mode='HTML', reply_markup=InlineKeyboardMarkup(keyboard))

# ============ GUESS THE NUMBER GAME COMMANDS ============

async def guess_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start a new Guess the Number game"""
    user = update.effective_user
    chat_id = update.effective_chat.id
    
    try:
        # Check if user is registered
        player = bot_instance.find_player_by_identifier(str(user.id))
        if not player:
            await update.message.reply_text(
                "❌ <b>Not registered!</b>\n\n"
                "Please use /start to register first!",
                parse_mode='HTML'
            )
            return
        
        # Check if user already has active game
        active_game = bot_instance.get_guess_game(user.id)
        if active_game:
            # Create smooth transition message
            transition_msg, has_conflicts = bot_instance.create_game_switch_message(user.id, "a new Guess Game")
            if has_conflicts:
                await update.message.reply_text(transition_msg, parse_mode='HTML')
                return
        
        # Check if difficulty was specified
        args = context.args
        if args and len(args) > 0:
            difficulty = args[0].lower()
            if difficulty in bot_instance.guess_difficulties:
                # Check if level is unlocked
                unlocked_levels = bot_instance.get_unlocked_levels(user.id)
                if difficulty not in unlocked_levels:
                    await update.message.reply_text(
                        f"🔒 <b>Level Locked!</b>\n\n"
                        f"You need to complete easier levels first to unlock <b>{difficulty.title()}</b>.\n"
                        f"Currently unlocked: {', '.join(unlocked_levels)}",
                        parse_mode='HTML'
                    )
                    return
                    
                # Start game directly with specified difficulty
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
        
        # Get unlocked levels
        unlocked_levels = bot_instance.get_unlocked_levels(user.id)
        
        # Show difficulty selection with lock status
        keyboard = []
        for difficulty, config in bot_instance.guess_difficulties.items():
            range_str = f"{config['range'][0]}-{config['range'][1]}"
            
            if difficulty in unlocked_levels:
                # Unlocked level
                button_text = f"{config['emoji']} {difficulty.title()} ({range_str})"
                keyboard.append([InlineKeyboardButton(
                    button_text, callback_data=f"guess_start:{difficulty}"
                )])
            else:
                # Locked level
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

async def guess_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """View guess game statistics - Public command with personal stats"""
    user = update.effective_user
    
    # Get player info first
    player = bot_instance.get_player_by_telegram_id(user.id)
    if not player:
        await update.message.reply_text("❌ <b>Player not found!</b>\n\n� Use /start to register first.", parse_mode='HTML')
        return
    
    try:
        # Get personal stats for the user
        personal_stats = bot_instance.get_guess_game_stats(user.id)
        
        # Get global stats
        global_stats = bot_instance.get_guess_game_stats()
        
        message = (
            f"🎯 <b>GUESS GAME STATISTICS</b>\n"
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
            f"� <b>Want to play?</b> Use /guess to start!"
        )
        
        await update.message.reply_text(message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Error in guess_stats_command: {e}")
        await update.message.reply_text(
            "❌ <b>Error retrieving statistics!</b>\n\n"
            "Please try again later.",
            parse_mode='HTML'
        )


async def guess_leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show leaderboard command options"""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    
    # Create copyable command buttons
    keyboard = [
        [InlineKeyboardButton("🏆 Highest Scores", callback_data="guess_lb_highest")],
        [InlineKeyboardButton("🎯 Total Scores", callback_data="guess_lb_total")],
        [InlineKeyboardButton("🎮 Most Games", callback_data="guess_lb_games")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    message = (
        "📊 <b>GUESS GAME LEADERBOARDS</b>\n\n"
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
    
    # Handle both callback queries and regular messages
    query = update.callback_query
    if query:
        await query.answer()
    
    try:
        # Get highest score leaderboard
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
            # Format leaderboard message
            message = "📊 <b>GUESS GAME LEADERBOARD</b>\n"
            message += "🏆 <b>HIGHEST SCORES</b>\n\n"
            
            for i, player in enumerate(leaderboard, 1):
                name = H(player.get('player_name', 'Unknown'))
                
                # Position emoji
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
    
    # Handle both callback queries and regular messages
    query = update.callback_query
    if query:
        await query.answer()
    
    try:
        # Get total score leaderboard
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
            # Format leaderboard message
            message = "📊 <b>GUESS GAME LEADERBOARD</b>\n"
            message += "🎯 <b>TOTAL SCORES</b>\n\n"
            
            for i, player in enumerate(leaderboard, 1):
                name = H(player.get('player_name', 'Unknown'))
                
                # Position emoji
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
    
    # Handle both callback queries and regular messages
    query = update.callback_query
    if query:
        await query.answer()
    
    try:
        # Get games leaderboard
        leaderboard = bot_instance.get_guess_leaderboard('games', 10)
        
        if not leaderboard:
            message = (
                "📊 <b>MOST GAMES LEADERBOARD</b>\n\n"
                "🚫 <b>No data available yet!</b>\n\n"
                "Play some games to populate the leaderboard! 🎲"
            )
        else:
            # Format leaderboard message
            message = "📊 <b>GUESS GAME LEADERBOARD</b>\n"
            message += "🎮 <b>MOST GAMES</b>\n\n"
            
            for i, player in enumerate(leaderboard, 1):
                name = H(player.get('player_name', 'Unknown'))
                
                # Position emoji
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

async def daily_guess_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Play daily guess challenge - ONE ATTEMPT ONLY per day"""
    user = update.effective_user
    chat_id = update.effective_chat.id
    today = date.today().isoformat()
    
    try:
        # Check if user is registered
        player = bot_instance.find_player_by_identifier(str(user.id))
        if not player:
            await update.message.reply_text(
                "❌ <b>Not registered!</b>\n\n"
                "Please use /start to register first!",
                parse_mode='HTML'
            )
            return
        
        # Check if user already attempted today's challenge (ONCE ONLY - win OR lose)
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
        
        # Generate today's challenge with anti-sharing measures
        # Each player gets a DIFFERENT number but with same difficulty
        import hashlib
        
        # Create a unique seed for each player based on date + user_id
        combined_seed = f"{today}_spl_secret_2024_{user.id}"
        date_seed = int(hashlib.md5(combined_seed.encode()).hexdigest(), 16) % 1000000
        random.seed(date_seed)
        
        # Today's challenge - each player gets different number (1-100)
        challenge_difficulty = 'medium'  # Use medium difficulty for daily challenges
        target = random.randint(1, 100)  # Each player gets unique number
        
        # Reset random seed
        random.seed()
        
        # Create challenge game with STRICT rules
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
        
        # Store in active games
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

async def guess_challenge_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """View daily challenge leaderboard"""
    await update.message.reply_text(
        "🚧 <b>Daily Challenge Leaderboard</b> 🚧\n\n"
        "This feature is coming soon!\n\n"
        "Use /guessleaderboard to see current rankings! 📊",
        parse_mode='HTML'
    )

# Admin command for guess game cleanup
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
        # Cleanup expired games
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
    """Reset all database data (Super Admin only)"""
    if not bot_instance.is_super_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n👑 Only Super Admin can reset all data!", parse_mode='HTML')
        return
    
    # Require confirmation
    if len(context.args) == 0 or context.args[0] != 'CONFIRM':
        await update.message.reply_text(
            "⚠️ <b>DANGER ZONE!</b> ⚠️\n\n"
            "🔥 <b>This will DELETE ALL:</b>\n"
            "• All player achievements\n"
            "• All player titles\n"
            "• All pending achievements\n\n"
            "✅ <b>Players will remain registered</b>\n"
            "💾 <b>Backups will be created for recovery</b>\n\n"
            "⚡ <b>To proceed, use:</b>\n"
            "<code>/resetall CONFIRM</code>",
            parse_mode='HTML'
        )
        return
    
    # Perform the reset
    success = bot_instance.reset_all_data(update.effective_user.id)
    
    if success:
        await update.message.reply_text(
            "💥 <b>DATABASE RESET COMPLETE!</b> 💥\n\n"
            "🔥 <b>All data cleared:</b>\n"
            "• ✅ All achievements removed\n"
            "• ✅ All titles cleared\n"
            "• ✅ All pending achievements deleted\n\n"
            "💾 <b>Recovery:</b> All actions backed up - use /backups to restore\n"
            "👥 <b>Players:</b> Registration data preserved\n\n"
            "🎯 <b>Ready for fresh start!</b> 🏏",
            parse_mode='HTML'
        )
    else:
        await update.message.reply_text("❌ <b>RESET FAILED!</b>\n\nSomething went wrong.", parse_mode='HTML')



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
        
        # Get all players with their achievement counts
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
        
        # Create player list message
        message = f"👥 <b>REGISTERED PLAYERS ({len(players)})</b> 📋\n\n"
        message += "━━━━━━━━━━━━━━━━━━━━\n"
        
        for i, player in enumerate(players, 1):
            player_id, telegram_id, username, display_name, title, created_at, ach_count, total_awards = player
            
            # Player info
            message += f"<b>{i}.</b> {H(display_name)}\n"
            
            if username:
                message += f"   📱 @{H(username)}\n"
            
            message += f"   🆔 {telegram_id}\n"
            
            if title:
                message += f"   👑 <b>{H(title)}</b>\n"
            
            # Achievement stats
            achievements_text = f"{ach_count or 0} types" if ach_count else "0 types"
            awards_text = f"{total_awards or 0} total" if total_awards else "0 total"
            message += f"   🏆 {achievements_text}, {awards_text}\n"
            
            # Registration date
            reg_date = created_at.strftime('%m/%d/%Y')
            message += f"   📅 {reg_date}\n\n"
            
            # Split long messages
            if len(message) > 3500:  # Telegram limit is ~4096
                message += "━━━━━━━━━━━━━━━━━━━━\n"
                message += f"📊 <b>Showing {i} of {len(players)} players</b>"
                await update.message.reply_text(message, parse_mode='HTML')
                
                # Start new message
                message = f"👥 <b>REGISTERED PLAYERS (continued)</b> 📋\n\n"
                message += "━━━━━━━━━━━━━━━━━━━━\n"
        
        # Send final message
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
        
        # Get total users
        cursor.execute("SELECT COUNT(*) FROM players")
        total_users = cursor.fetchone()[0] or 0
        
        # Get users with titles
        cursor.execute("SELECT COUNT(*) FROM players WHERE title IS NOT NULL AND title != ''")
        users_with_titles = cursor.fetchone()[0] or 0
        
        # Get total achievement types
        cursor.execute("SELECT COUNT(*) FROM achievements")
        total_achievement_records = cursor.fetchone()[0] or 0
        
        # Get total awards (sum of all counts)
        cursor.execute("SELECT SUM(count) FROM achievements")
        total_awards = cursor.fetchone()[0] or 0
        
        # Get unique achievement types
        cursor.execute("SELECT COUNT(DISTINCT achievement_name) FROM achievements")
        unique_achievements = cursor.fetchone()[0] or 0
        
        # Get admin count
        cursor.execute("SELECT COUNT(*) FROM admins")
        admin_count = cursor.fetchone()[0] or 0
        
        # Get recent activity (last 7 days)
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
        
        # Get total shard circulation
        cursor.execute("SELECT SUM(shards) FROM players WHERE shards > 0")
        total_shard_circulation = cursor.fetchone()[0] or 0
        
        # Get top achievers
        cursor.execute("""
            SELECT p.display_name, SUM(a.count) as total_awards
            FROM players p
            JOIN achievements a ON p.id = a.player_id
            GROUP BY p.id, p.display_name
            ORDER BY total_awards DESC
            LIMIT 3
        """)
        top_achievers = cursor.fetchall()
        
        # Build status message
        from datetime import datetime
        current_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        
        message = f"🤖 <b>SPL BOT STATUS</b> 📊\n\n"
        message += f"⏰ <b>Status Check:</b> {current_time}\n"
        message += f"━━━━━━━━━━━━━━━━━━━━\n\n"
        
        # Database Status
        message += "💾 <b>DATABASE STATUS:</b>\n"
        message += f"✅ <b>Connection:</b> Active\n"
        message += f"👥 <b>Total Users:</b> {total_users}\n"
        message += f"👑 <b>Users with Titles:</b> {users_with_titles}\n"
        message += f"🛡️ <b>Total Admins:</b> {admin_count}\n"
        message += f"💠 <b>Total Shard Circulation:</b> {total_shard_circulation:,} 💠\n\n"
        
        # Achievement Statistics
        message += "🏆 <b>ACHIEVEMENT STATS:</b>\n"
        message += f"🎯 <b>Total Awards Given:</b> {total_awards}\n"
        message += f"📝 <b>Achievement Records:</b> {total_achievement_records}\n"
        message += f"🔢 <b>Unique Types:</b> {unique_achievements}\n\n"
        
        # Recent Activity
        message += "📈 <b>RECENT ACTIVITY (7 days):</b>\n"
        message += f"🏅 <b>New Awards:</b> {recent_achievements}\n"
        message += f"👤 <b>New Users:</b> {recent_registrations}\n\n"
        
        # Top Achievers
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

async def testlog_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Test admin logs system (Admin only)"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ <b>ACCESS DENIED!</b>\n\n🛡️ Only admins can test logs.", parse_mode='HTML')
        return
    
    try:
        # Send test log
        success = await bot_instance.send_admin_log(
            'admin_action',
            f"Test log message from admin | Command executed successfully",
            update.effective_user.id,
            update.effective_user.username
        )
        
        if success:
            await update.message.reply_text(
                "✅ <b>TEST LOG SENT!</b>\n\n"
                f"📝 Log sent to admin logs group\n"
                f"📊 Check the logs channel to verify delivery",
                parse_mode='HTML'
            )
        else:
            await update.message.reply_text(
                "❌ <b>TEST LOG FAILED!</b>\n\n"
                f"⚠️ Failed to send log to admin group\n"
                f"🔧 Check bot configuration and logs",
                parse_mode='HTML'
            )
    except Exception as e:
        await update.message.reply_text(
            f"❌ <b>ERROR TESTING LOGS!</b>\n\n"
            f"🚨 Exception: {str(e)[:100]}",
            parse_mode='HTML'
        )

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
    
    # Get user statistics
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

async def banuser_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ban user from using the bot (Admin only)"""
    if not bot_instance.is_admin(update.effective_user.id):
        await update.message.reply_text(
            "❌ <b>ACCESS DENIED!</b>\n\n🛡️ This command requires administrator privileges.",
            parse_mode='HTML'
        )
        return

    if len(context.args) < 1:
        await update.message.reply_text(
            "❌ <b>USAGE ERROR!</b>\n\n"
            "<b>Usage:</b> /banuser @username [reason]\n"
            "<b>Example:</b> /banuser @baduser Spam and abuse",
            parse_mode='HTML'
        )
        return

    identifier = context.args[0]
    reason = ' '.join(context.args[1:]) if len(context.args) > 1 else "No reason provided"
    
    player = bot_instance.find_player_by_identifier(identifier)
    
    if not player:
        await update.message.reply_text(
            f"❌ <b>USER NOT FOUND!</b>\n\n"
            f"🔍 No user found with identifier: <b>{identifier}</b>",
            parse_mode='HTML'
        )
        return
    
    try:
        # Implement ban functionality by marking user as banned
        success = bot_instance.ban_user(player['player_id'], reason, update.effective_user.id)
        
        if success:
            await update.message.reply_text(
                f"🚫 <b>USER BANNED SUCCESSFULLY</b>\n\n"
                f"👤 <b>Banned:</b> {player['display_name']}\n"
                f"📝 <b>Reason:</b> {reason}\n"
                f"� <b>Banned by:</b> {update.effective_user.first_name}\n\n"
                f"⚠️ User is now restricted from bot functions.",
                parse_mode='HTML'
            )
        else:
            await update.message.reply_text(
                f"❌ <b>BAN FAILED</b>\n\n"
                f"Failed to ban user {player['display_name']}. Please try again.",
                parse_mode='HTML'
            )
    except Exception as e:
        log_exception("ban_user_command", e, update.effective_user.id)
        await update.message.reply_text(
            "❌ <b>ERROR</b>\n\nFailed to execute ban command. Please try again.",
            parse_mode='HTML'
        )

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
        # Implement unban functionality
        success = bot_instance.unban_user(player['player_id'], update.effective_user.id)
        
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
        
        # Properly restart the bot
        logger.info(f"Bot restart requested by Super Admin {update.effective_user.id}")
        
        # Give a moment for the message to send
        await asyncio.sleep(2)
        
        # Clean shutdown and restart
        import os
        import sys
        
        # Stop all background tasks
        bot_instance.shutdown_flag = True
        
        # Exit and let the hosting service restart us
        logger.info("Bot shutting down for restart...")
        os._exit(0)
        
    except Exception as e:
        log_exception("restart_command", e, update.effective_user.id)
        await update.message.reply_text(
            "❌ <b>RESTART FAILED</b>\n\nError occurred during restart. Please check logs.",
            parse_mode='HTML'
        )
    
async def backup_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Create database backup (Super Admin only)"""
    if not bot_instance.is_super_admin(update.effective_user.id):
        await update.message.reply_text(
            "❌ <b>ACCESS DENIED!</b>\n\n👑 Only Super Admin can create backups!",
            parse_mode='HTML'
        )
        return

    await update.message.reply_text(
        "💾 <b>DATABASE BACKUP</b>\n\n"
        "🚧 <b>Feature in development!</b>\n\n"
        "💡 For now, manually backup your database using:\n"
        "<code>pg_dump your_database > backup_$(date +%Y%m%d_%H%M%S).sql</code>",
        parse_mode='HTML'
    )

async def cleancache_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clean bot cache (Super Admin only)"""
    if not bot_instance.is_super_admin(update.effective_user.id):
        await update.message.reply_text(
            "❌ <b>ACCESS DENIED!</b>\n\n👑 Only Super Admin can clean cache!",
            parse_mode='HTML'
        )
        return

    # Clear various caches
    try:
        # Clear profile cache if it exists
        if hasattr(bot_instance, 'profile_cache'):
            bot_instance.profile_cache.clear()
        
        # Clear other caches
        if hasattr(bot_instance, 'leaderboard_cache'):
            bot_instance.leaderboard_cache.clear()
            
        if hasattr(bot_instance, 'goat_cache'):
            bot_instance.goat_cache.clear()
            
        await update.message.reply_text(
            "🧹 <b>CACHE CLEANED SUCCESSFULLY!</b>\n\n"
            "✅ Profile cache cleared\n"
            "✅ Leaderboard cache cleared\n" 
            "✅ GOAT cache cleared\n\n"
            "🚀 Bot performance may improve!",
            parse_mode='HTML'
        )
    except Exception as e:
        logger.error(f"Error cleaning cache: {e}")
        await update.message.reply_text(
            "❌ <b>CACHE CLEANUP FAILED!</b>\n\n"
            f"Error: {str(e)}",
            parse_mode='HTML'
        )

# ============ GUESS GAME CALLBACK HANDLERS ============

async def guess_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle guess game callback queries"""
    query = update.callback_query
    await query.answer()
    
    user = query.from_user
    data = query.data
    
    try:
        if data.startswith('guess_start:'):
            # Start a new game with selected difficulty
            difficulty = data.split(':')[1]
            
            # Check if user is registered
            player = bot_instance.find_player_by_identifier(str(user.id))
            if not player:
                await query.edit_message_text(
                    "❌ <b>Not registered!</b>\n\n"
                    "Please use /start to register first!",
                    parse_mode='HTML'
                )
                return
            
            # Check if level is unlocked
            unlocked_levels = bot_instance.get_unlocked_levels(user.id)
            if difficulty not in unlocked_levels:
                await query.edit_message_text(
                    f"🔒 <b>Level Locked!</b>\n\n"
                    f"You need to complete easier levels first to unlock <b>{difficulty.title()}</b>.\n"
                    f"Currently unlocked: {', '.join(unlocked_levels)}",
                    parse_mode='HTML'
                )
                return
            
            # Check if user already has active game
            active_game = bot_instance.get_guess_game(user.id)
            if active_game:
                # Create smooth transition message
                transition_msg, has_conflicts = bot_instance.create_game_switch_message(user.id, f"{difficulty.title()} Guess Game")
                if has_conflicts:
                    await query.edit_message_text(transition_msg, parse_mode='HTML')
                    return
            
            # Create new game
            game = bot_instance.create_guess_game(
                user.id, difficulty, player['display_name'], query.message.chat_id
            )
            
            if game:
                message = format_guess_game_start(game)
                
                # Add hint button
                keyboard = [[InlineKeyboardButton("🍀 Lucky Hint", callback_data="guess_hint")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await query.edit_message_text(message, parse_mode='HTML', reply_markup=reply_markup)
            else:
                await query.edit_message_text(
                    "❌ <b>Error!</b> Failed to start game. Please try again.",
                    parse_mode='HTML'
                )
        
        elif data.startswith('guess_locked:'):
            # Handle locked level click
            difficulty = data.split(':')[1]
            unlocked_levels = bot_instance.get_unlocked_levels(user.id)
            
            await query.edit_message_text(
                f"🔒 <b>Level Locked!</b>\n\n"
                f"<b>{difficulty.title()}</b> is not yet available.\n"
                f"Complete easier levels to unlock it!\n\n"
                f"🔓 <b>Currently unlocked:</b> {', '.join(unlocked_levels)}\n\n"
                f"💡 <b>Tip:</b> Win games to progressively unlock harder levels!",
                parse_mode='HTML'
            )
                
        elif data == 'guess_hint':
            # Provide hint for active game
            game = bot_instance.get_guess_game(user.id)
            if not game or not game.get('game_active'):
                await query.edit_message_text(
                    "❌ <b>No active game!</b>\n\nUse /guess to start a new game.",
                    parse_mode='HTML'
                )
                return
            
            if game['hint_used']:
                await query.answer("You've already used your hint for this game!", show_alert=True)
                return
            
            # Generate and show hint
            hint = bot_instance.generate_hint(game)
            game['hint_used'] = True
            
            # Update game display with hint
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
            
            # Remove hint button since it's used
            await query.edit_message_text(message, parse_mode='HTML')
            
        elif data == 'guess_lb_highest':
            # Show highest scores leaderboard
            await show_guess_leaderboard_highest(update, context)
            
        elif data == 'guess_lb_total':
            # Show total scores leaderboard  
            await show_guess_leaderboard_total(update, context)
            
        elif data == 'guess_lb_games':
            # Show most games leaderboard
            await show_guess_leaderboard_games(update, context)
            
    except Exception as e:
        logger.error(f"Guess callback error: {e}")
        await query.edit_message_text(
            "❌ <b>Error!</b> Please try again later.",
            parse_mode='HTML'
        )

async def handle_guess_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle number guesses from users"""
    user = update.effective_user
    text = update.message.text
    
    # First check if this is a nightmare mode guess
    nightmare_handled = await handle_nightmare_guess(update, context)
    if nightmare_handled:
        return  # Nightmare mode handled the input
    
    # Check if user has active regular guess game
    game = bot_instance.get_guess_game(user.id)
    if not game or not game.get('game_active'):
        return  # Not playing, ignore message
    
    try:
        # Validate input is a number
        try:
            guess = int(text.strip())
        except ValueError:
            return  # Not a number, ignore
        
        # Check if game has timed out
        elapsed = time.time() - game['start_time']
        if elapsed > game['time_limit']:
            bot_instance.end_guess_game(user.id, 'timeout')
            
            # For daily challenges, don't reveal the answer
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
        
        # Validate guess is in range
        config = bot_instance.guess_difficulties[game['difficulty']]
        if guess < config['range'][0] or guess > config['range'][1]:
            await update.message.reply_text(
                f"❌ <b>Invalid guess!</b>\n\n"
                f"Please guess between {config['range'][0]}-{config['range'][1]}",
                parse_mode='HTML'
            )
            return
        
        # Process the guess
        game['attempts_used'] += 1
        game['guesses'].append(guess)
        
        if guess == game['target_number']:
            # Correct guess - game won!
            success = bot_instance.end_guess_game(user.id, 'won')
            time_taken = int(elapsed)
            final_score = bot_instance.calculate_guess_score(game, 'won', time_taken)
            
            # For daily challenges, don't reveal the answer to prevent sharing
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
            
            # Add shard reward info if available
            if game.get('shard_reward'):
                message += f"💠 <b>Shards Earned:</b> +{game['shard_reward']}\n\n"
                
                if game['attempts_used'] == 1:
                    message += "⚡ <b>Perfect guess!</b> Outstanding! 🌟\n"
                elif game['attempts_used'] <= 3:
                    message += "🔥 <b>Excellent guessing!</b> Great job! 👏\n"
                else:
                    message += "👍 <b>Well done!</b> Good game! 🎯\n"
                
                # Check if new level was unlocked
                if hasattr(game, 'new_level_unlocked') and game['new_level_unlocked']:
                    message += f"\n🎊 <b>NEW LEVEL UNLOCKED!</b>\n🔓 You can now play <b>{game['new_level_unlocked'].title()}</b> difficulty!\n"
                
                message += "\n🎲 Use /guess to play again!"
            
            await update.message.reply_text(message, parse_mode='HTML')
            
        elif game['attempts_used'] >= game['max_attempts']:
            # Out of attempts - game lost
            bot_instance.end_guess_game(user.id, 'lost')
            
            # For daily challenges, don't reveal the answer
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
            
            # Add shard reward info if available (small consolation prize)
            if game.get('shard_reward'):
                message += f"\n💠 <b>Shards Earned:</b> +{game['shard_reward']}"
            
            await update.message.reply_text(message, parse_mode='HTML')
            
        else:
            # Give hint and continue
            attempts_left = game['max_attempts'] - game['attempts_used']
            time_left = max(0, game['time_limit'] - int(elapsed))
            
            if guess < game['target_number']:
                hint_msg = "🔼 <b>Higher!</b>"
                game['range_min'] = max(game['range_min'], guess + 1)
            else:
                hint_msg = "🔽 <b>Lower!</b>"
                game['range_max'] = min(game['range_max'], guess - 1)
            
            # Create progress bar
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
            # Approve single achievement
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
            # Deny single achievement
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
            
            # Set server timeout to prevent hanging
            server.timeout = 30
            
            # Start the server
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
            # Check bot health
            await application.bot.get_me()
            logger.debug("Bot health check passed")
            
            # Ping our own service if we're on Render
            if base_url:
                async with aiohttp.ClientSession() as session:
                    async with session.get(base_url) as response:
                        logger.debug(f"Self-ping status: {response.status}")
            
        except Exception as e:
            logger.error(f"Error in keep_alive: {e}")
        
        # Sleep for 12 minutes (well before Render's 15-min timeout)
        await asyncio.sleep(300)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)

    # Send message to the user only if we have a valid message to reply to
    error_message = "❌ <b>An error occurred!</b>\n\nPlease try again or contact support if the issue persists."
    
    try:
        if update and update.effective_message:
            await update.effective_message.reply_text(error_message, parse_mode='HTML')
    except Exception as e:
        logger.error(f"Error in error handler: {e}")
    
    # If it's a conflict error (multiple instances), log it clearly
    if isinstance(context.error, telegram_error.Conflict):
        logger.error("Bot instance conflict detected. Make sure no other instances are running.")
        return

async def view_achievements_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display achievements for a user"""
    user = update.effective_user
    player = bot_instance.find_player_by_identifier(str(user.id))
    
    if not player:
        await update.message.reply_text(
            "❌ <b>Not registered!</b>\n\n"
            "Please use /start to register first!",
            parse_mode='HTML'
        )
        return
    
    achievements = bot_instance.get_player_achievements(player['id'])
    message = bot_instance.format_achievements_message(player, achievements)
    await update.message.reply_text(message, parse_mode='HTML')

async def button_click(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button clicks - simplified without pending achievements"""
    query = update.callback_query
    await query.answer()  # Answer callback query to remove loading state
    
    # Basic handling for any future button interactions
    try:
        new_text = "❌ No valid actions available"
        
        # Update the message to remove buttons and show result
        await query.edit_message_text(
            text=new_text,
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
            # Track the chat for broadcast purposes
            bot_instance.track_chat(
                chat_id=chat.id,
                chat_type=chat.type,
                title=getattr(chat, 'title', None),
                username=getattr(chat, 'username', None)
            )
    except Exception as e:
        logger.error(f"Error tracking message: {e}")
    # This handler doesn't consume the update, it just logs the chat


async def periodic_cleanup():
    """Periodic cleanup task to prevent memory leaks"""
    while True:
        try:
            await asyncio.sleep(600)  # Run every 10 minutes
            
            # Clean up chase games
            chase_cleaned = cleanup_expired_games()
            if chase_cleaned > 0:
                logger.info(f"Periodic cleanup: removed {chase_cleaned} expired chase games")
            
            # Clean up guess games
            guess_cleaned = bot_instance.cleanup_expired_guess_games()
            if guess_cleaned > 0:
                logger.info(f"Periodic cleanup: removed {guess_cleaned} expired guess games")
                
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")


def main() -> None:
    """Start the bot."""
    logger.info("Starting SPL Achievement Bot...")
    
    # Initialize database
    if not bot_instance.init_database():
        logger.error("Failed to initialize database!")
        return
    
    # Add a small delay to allow any existing instances to timeout
    logger.info("Waiting for any existing bot instances to timeout...")
    time.sleep(2)
    
    # Start health check server in a separate thread
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()
    
    # Start the watchdog
    base_url = os.environ.get("RENDER_EXTERNAL_URL")
    if base_url:
        watchdog = BotWatchdog(base_url)
        watchdog.start()

    # Create and start the application with proper timeout configuration
    application = (Application.builder()
                  .token(bot_instance.bot_token)
                  .read_timeout(30)
                  .connect_timeout(30) 
                  .write_timeout(30)
                  .build())
    
    # Add startup callback to initialize periodic cleanup
    async def post_init(app):
        """Initialize periodic cleanup task after bot starts"""
        import asyncio
        asyncio.create_task(periodic_cleanup())
        logger.info("Started periodic cleanup task for chase games")
    
    application.post_init = post_init
    
    # Add message pre-processor to track all chats (must be first)
    from telegram.ext import MessageHandler, filters
    application.add_handler(MessageHandler(filters.ALL, track_message), group=-1)
    
    # Basic commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("update", update_command))
    application.add_handler(CommandHandler("quit", quit_command))
    application.add_handler(CommandHandler("emojis", emojis_command))
    application.add_handler(CommandHandler("achievements", view_achievements_command))
    application.add_handler(CommandHandler("profile", profile_command))
    application.add_handler(CommandHandler("goat", goat_command))
    application.add_handler(CommandHandler("myroast", my_roast_command))
    application.add_handler(CommandHandler("chase", chase_command))
    application.add_handler(CommandHandler("chasestats", chase_stats_command))
    application.add_handler(CommandHandler("leaderboard", leaderboard_command))
    application.add_handler(CommandHandler("dailylb", dailylb_command))
    
    # Guess game commands
    application.add_handler(CommandHandler("guess", guess_command))
    application.add_handler(CommandHandler("guessstats", guess_stats_command))
    application.add_handler(CommandHandler("guessleaderboard", guess_leaderboard_command))
    application.add_handler(CommandHandler("dailyguess", daily_guess_command))
    application.add_handler(CommandHandler("guesschallenge", guess_challenge_command))
    application.add_handler(CommandHandler("nightmare", nightmare_command))
    
    # Currency commands  
    application.add_handler(CommandHandler("balance", balance_command))
    application.add_handler(CommandHandler("daily", daily_command))
    application.add_handler(CommandHandler("shards", shards_command))
    application.add_handler(CommandHandler("shardlb", shard_leaderboard_command))
    
    # Admin commands
    application.add_handler(CommandHandler("adminpanel", adminpanel_command))  # Super admin control panel
    application.add_handler(CommandHandler("addachievement", add_achievement_command))
    application.add_handler(CommandHandler("removeachievement", remove_achievement_command))
    application.add_handler(CommandHandler("settitle", set_title_command))
    application.add_handler(CommandHandler("aadmins", list_admins_command))
    application.add_handler(CommandHandler("bulkward", bulk_award_command))
    application.add_handler(CommandHandler("cleanupchase", cleanup_chase_command))
    application.add_handler(CommandHandler("cleanupguess", cleanup_guess_command))
    application.add_handler(CommandHandler("broadcast", broadcast_command))
    application.add_handler(CommandHandler("draftbroadcast", draftbroadcast_command))  # Draft broadcast messages
    application.add_handler(CommandHandler("testbroadcast", testbroadcast_command))  # Test broadcast to admins
    application.add_handler(CommandHandler("transactions", transactions_command))  # View transaction log
    application.add_handler(CommandHandler("giveshards", give_shards_command))
    application.add_handler(CommandHandler("removeshards", remove_shards_command))
    application.add_handler(CommandHandler("distributedailyrewards", distribute_daily_rewards_command))
    application.add_handler(CommandHandler("resetdailylb", reset_daily_leaderboard_command))
    application.add_handler(CommandHandler("dailylbstats", daily_leaderboard_stats_command))
    application.add_handler(CommandHandler("confirmachievement", confirm_achievement_command))
    application.add_handler(CommandHandler("listpending", list_pending_confirmations_command))
    
    # Super admin commands
    application.add_handler(CommandHandler("addadmin", add_admin_command))
    application.add_handler(CommandHandler("rmadmin", remove_admin_command))
    application.add_handler(CommandHandler("resetall", reset_all_command))
    application.add_handler(CommandHandler("resetplayer", reset_player_command))
    application.add_handler(CommandHandler("listplayers", list_players_command))
    application.add_handler(CommandHandler("botstatus", bot_status_command))
    application.add_handler(CommandHandler("testlog", testlog_command))  # Test admin logs
    
    # Admin panel command aliases (for easier access)
    application.add_handler(CommandHandler("giveachievement", add_achievement_command))  # Alias for addachievement
    application.add_handler(CommandHandler("rmachievement", remove_achievement_command))  # Alias for removeachievement
    application.add_handler(CommandHandler("givetitle", set_title_command))  # Alias for settitle
    application.add_handler(CommandHandler("rmtitle", remove_title_command))  # Remove title command
    application.add_handler(CommandHandler("finduser", finduser_command))  # Find user information
    application.add_handler(CommandHandler("banuser", banuser_command))  # Ban user (placeholder)
    application.add_handler(CommandHandler("unbanuser", unbanuser_command))  # Unban user (placeholder)
    application.add_handler(CommandHandler("restart", restart_command))  # Restart bot
    application.add_handler(CommandHandler("backup", backup_command))  # Database backup
    application.add_handler(CommandHandler("cleancache", cleancache_command))  # Clean cache
    
    # Handle chase game callbacks (specific pattern)
    application.add_handler(CallbackQueryHandler(chase_callback, pattern=r"^chase:"))
    
    # Handle guess game callbacks
    application.add_handler(CallbackQueryHandler(guess_callback, pattern=r"^guess_"))
    
    # Handle nightmare mode callbacks
    application.add_handler(CallbackQueryHandler(nightmare_callback, pattern=r"^nightmare_"))
    
    # Handle help menu callbacks
    application.add_handler(CallbackQueryHandler(help_callback, pattern=r"^help_"))
    
    # Handle update menu callbacks
    application.add_handler(CallbackQueryHandler(update_callback, pattern=r"^update_"))
    
    # Handle admin panel callbacks
    application.add_handler(CallbackQueryHandler(admin_panel_callback, pattern=r"^panel_"))
    application.add_handler(CallbackQueryHandler(admin_panel_callback, pattern=r"^action_"))
    
    # Handle daily leaderboard callbacks
    application.add_handler(CallbackQueryHandler(dailylb_callback, pattern=r"^dailylb_"))
    
    # Handle broadcast confirmation callbacks
    application.add_handler(CallbackQueryHandler(broadcast_callback, pattern=r"^broadcast_"))
    
    # Handle inline button clicks
    application.add_handler(CallbackQueryHandler(button_click))
    
    # Add message handler for guess game input (must be after track_message)
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_guess_input), group=1)
    
    # Set up error handler only if not already registered
    if not any(handler.callback == error_handler for handler in application.error_handlers):
        application.add_error_handler(error_handler)
    
    logger.info("All handlers registered successfully")
    
    # Start health check server in a separate thread
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()
    
    # Start daily reward scheduler in background
    import asyncio
    loop = asyncio.new_event_loop()
    scheduler_thread = threading.Thread(
        target=lambda: asyncio.run(schedule_daily_rewards()),
        daemon=True
    )
    scheduler_thread.start()
    logger.info("Daily reward scheduler started (MANUAL MODE - Use /distributedailyrewards)")
    
    # Run the bot with optimized polling settings
    logger.info("Starting SPL Achievement Bot...")
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True  # Clean start each time
    )

if __name__ == '__main__':
    while True:  # Keep the bot running forever
        try:
            main()  # Run the main bot logic
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            break  # Exit on manual interruption
        except Exception as e:
            logger.error(f"Bot stopped due to error: {e}")
            logger.info("Restarting bot in 10 seconds...")
            time.sleep(10)  # Wait before restarting
            continue
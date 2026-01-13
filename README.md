# 🏆 Arena Of Champions Bot

> **The Ultimate Cricket Gaming & Auction Bot for Telegram**

A comprehensive Telegram bot featuring interactive cricket games, IPL-style team auctions, achievements system, leaderboards, and forced channel membership - all with a modern, emoji-rich interface.

[![Telegram Bot](https://img.shields.io/badge/Telegram-Bot-blue?logo=telegram)](https://t.me/your_bot_username)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 🌟 Key Features

### 🎮 **Interactive Games**
- **Guess The Number**: 5 difficulty levels (Beginner to Expert)
  - Dynamic range from 1-20 to 1-500
  - Timed challenges with attempt limits
  - Score multipliers based on difficulty
- **Hand Cricket Chase**: Cricket-style run chase with wickets
  - Multiple difficulty levels
  - Strategic gameplay
  - Real-time scoring
- **Nightmare Mode**: Ultimate 10,000-number challenge
  - Expert-level difficulty
  - High risk, high reward
- **Daily Challenges**: Daily competitions with rewards

### 🎪 **IPL-Style Auction System**
- **Complete Auction Flow**: From proposal to final teams
  - Admin approval system for new auctions
  - Separate captain and player registration
  - Host control panel with live bidding
- **Team Management**: Captain & player registration with validation
- **Live Bidding**: Real-time chat-based bidding with countdown timer
- **Budget Control**: Purse management and team building
- **Admin Controls**: Complete auction management tools
- **Auction Results**: Sold/Unsold tracking with captain assignments

### 💎 **Shard Economy System**
- **Dynamic Currency**: Earn shards through gameplay
- **Daily Rewards**: Streak-based login bonuses (1 shard/day)
- **Achievement Bonuses**: 5 shards per achievement
- **Game Rewards**: 2 shards per game win
- **Transaction History**: Complete tracking of all shard movements
- **Leaderboards**: Compete for the top shard balance

### 🏆 **Achievement & Profile System**
- **Dynamic Achievements**: 20+ achievements including:
  - Winner, Orange Cap, Purple Cap, MVP
  - First Game, Century Master, Nightmare Survivor
  - Chase Champion, Guess Master, Daily Warrior
  - Weekly Champion, Monthly Legend, Season King
  - Perfect Scorer, Lucky Guesser, Streak Master
- **Custom Titles**: Earn and display custom titles
- **Comprehensive Profiles**: Track all stats across games
- **Achievement Confirmation**: Admin-approved achievement system

### 🔒 **Channel Membership Enforcement**
- **Mandatory Join**: Users must join official channels to use bot
- **Real-time Verification**: Instant membership checking
- **Join Buttons**: Easy one-click join interface
- **Re-verification**: Check membership on every command
- **No Spam**: Only checks for actual commands, not mentions

### 👨‍💼 **Advanced Administration**
- **3-Tier Admin System**: Super Admin, ENV Admin, DB Admin
- **User Management**: Ban, unban, and user lookup
- **Broadcast System**: Mass messaging with draft and test modes
- **Database Management**: Complete data control and backups
- **Bulk Operations**: Bulk achievement awards
- **Cache Management**: Smart cache with invalidation
- **Performance Monitoring**: Thread safety and cache status

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL database (or SQLite for local development)
- Telegram Bot Token from [@BotFather](https://t.me/BotFather)
- Official channel/group for forced membership

### Installation

```bash
# Clone the repository
git clone https://github.com/Krutik08062/SPL.git
cd SPL

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

### Environment Configuration

Create a `.env` file:

```env
# Bot Configuration
BOT_TOKEN=your_telegram_bot_token_here

# Database Configuration  
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Admin Configuration
SUPER_ADMIN_ID=123456789
ADMIN_IDS=987654321,111111111,222222222

# Logs Configuration (Optional)
LOGS_BOT_TOKEN=your_logs_bot_token
LOGS_CHAT_ID=your_logs_chat_id
LOGS_ENABLED=True

# Deployment Configuration (Optional)
PORT=8080
RENDER_EXTERNAL_URL=https://your-app.onrender.com
```

### Required Channels Configuration

Edit `bot.py` to set your required channels:

```python
# Required channels/groups for bot access
self.required_channels = [
    {'username': 'YourChannelUsername', 'name': 'Your Channel Name', 'url': 'https://t.me/YourChannel'},
    {'username': 'YourGroupUsername', 'name': 'Your Group Name', 'url': 'https://t.me/YourGroup'}
]
```

### Run the Bot

```bash
# Local development
python bot.py

# Production (with proper environment variables)
python bot.py
```

## 📋 Complete Command List

### **🎯 General Commands**
```
/start          - Welcome message and bot introduction
/help           - Get help information
/commands       - List all available commands
/ping           - Check bot responsiveness
/update         - Check for bot updates
/emojis         - View available emoji list
```

### **💎 Economy & Profile**
```
/shards         - Check your shards balance
/daily          - Claim daily shards reward
/profile        - View your complete profile
/achievements   - View your achievements collection
/shardlb        - View shards leaderboard
/dailylb        - View daily rewards leaderboard
```

### **🎮 Game Commands**

#### **Guess The Number**
```
/guess          - Play guess the number game (5 difficulty levels)
/guessstats     - View your guess game statistics
/guesslb        - View guess game leaderboard
/dailyguess     - Play daily guess challenge
```

#### **Hand Cricket Chase**
```
/chase          - Play hand cricket chase game
/chasestats     - View your chase game statistics
/leaderboard    - View main game leaderboard
```

#### **Special Challenges**
```
/nightmare      - Play nightmare mode challenge
/quit           - Quit your current active game
/goat           - View GOAT (Greatest Of All Time) player
/myroast        - Get a personalized roast
```

### **🎪 Auction Commands**

#### **Basic Auction Operations**
```
/auctionhelp    - Learn about auction system
/register       - Register for auction
/hostpanel      - Open auction host control panel
/regcap         - Register as team captain
/regplay        - Register as player for auction
```

#### **Team & Budget Management**
```
/myteam         - View your auction team
/purse          - Check your remaining purse amount
/transfercap    - Transfer captain role to another user
```

#### **Live Bidding**
```
/sell           - Sell current player to highest bidder
/rebid          - Place another bid on current player
```

#### **Auction Status & Information**
```
/status         - Check current auction status
/participants   - View auction participants list
/unsold         - View list of unsold players
```

#### **Host Controls**
```
/setgc          - Set group chat for auction
/addpt          - Manually add player to team
/pauseauc       - Pause/resume ongoing auction
```

### **👨‍💼 Admin Commands**

#### **User & Shard Management**
```
/giveshards     - Give shards to user (Admin)
/removeshards   - Remove shards from user (Admin)
/finduser       - Find user information (Admin)
/banuser        - Ban user from bot (Admin)
/unbanuser      - Unban previously banned user (Admin)
```

#### **Achievement Management**
```
/addach         - Add achievement to user (Admin)
/remach         - Remove achievement from user (Admin)
/settitle       - Set custom title for user (Admin)
/removetitle    - Remove custom title from user (Admin)
/confach        - Confirm pending achievement (Admin)
/pending_conf   - View pending achievement confirmations (Admin)
```

#### **Leaderboard Management**
```
/ddrlb          - Distribute daily rewards leaderboard (Admin)
/resetdlb       - Reset daily leaderboard (Admin)
/dlbstats       - View daily leaderboard statistics (Admin)
```

#### **System Maintenance**
```
/cleanupchase   - Cleanup old chase game sessions (Admin)
/cleanupguess   - Cleanup old guess game sessions (Admin)
/cleancache     - Clear all bot caches (Admin)
/cstatus        - View cache system status (Admin)
/tstatus        - View thread safety status (Admin)
```

#### **Broadcasting**
```
/broadcast      - Broadcast message to all users (Admin)
/bulkaward      - Bulk award achievements (Admin)
```

#### **Admin Management**
```
/adminstatus    - View administrator status and permissions (Admin)
/listadmins     - List all bot administrators
/addadmin       - Add new admin (Super Admin)
/removeadmin    - Remove admin privileges (Super Admin)
```

#### **Super Admin Only**
```
/resetplayer    - Reset player statistics (Admin)
/transactions   - View shard transaction history (Admin)
/resetall       - Reset all bot data (Super Admin)
/listplay       - List all registered players (Super Admin)
/botstatus      - View complete bot system status (Super Admin)
/restart        - Restart bot instance (Super Admin)
/backup         - Create database backup (Super Admin)
```

## 🔒 Channel Membership System

### How It Works

The bot enforces mandatory channel membership for all users:

1. **First-time Users**: Must join required channels before registration
2. **Existing Users**: Must join to continue using any command
3. **Real-time Verification**: Checks membership on every command execution
4. **Smart Detection**: Only checks for commands (starting with `/`), not mentions or tags

### User Flow

```
User sends /command
    ↓
Check if user is in all required channels
    ↓
Not member? → Show join buttons with "Check Again" option
    ↓
Member? → Check ban status
    ↓
Not banned? → Check registration
    ↓
Registered? → Execute command
```

### Setup Required Channels

Edit the `required_channels` list in `bot.py`:

```python
self.required_channels = [
    {
        'username': 'SagaArenaOfficial',
        'name': 'Saga Arena | Official',
        'url': 'https://t.me/SagaArenaOfficial'
    },
    {
        'username': 'SagaArenaChat',
        'name': 'Saga Arena • Community',
        'url': 'https://t.me/SagaArenaChat'
    }
]
```

### Admin Override

- Super Admins can bypass channel membership checks
- Useful for bot testing and emergency access

## 🏗️ Technical Architecture

### **Core Technologies**
- **Language**: Python 3.8+
- **Framework**: python-telegram-bot (async)
- **Database**: PostgreSQL with connection pooling
- **Async Operations**: asyncio for concurrent operations
- **HTTP Server**: aiohttp for web server and keep-alive

### **File Structure**
```
arena-of-champions/
├── bot.py                  # Main bot application (18,000+ lines)
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
├── render.yaml            # Render deployment config
├── .env                   # Environment variables (not in git)
├── .gitignore            # Git ignore rules
├── README.md             # This documentation
├── RENDER_DEPLOYMENT.md  # Deployment guide
└── runtime.txt           # Python version specification
```

### **Database Schema**

The bot automatically creates and manages these tables:

```sql
-- User Management
players (telegram_id, username, display_name, score, shards, level, created_at, last_active, custom_title, title_set_at, banned, ban_reason, ban_timestamp)

-- Achievements System
achievements (user_id, achievement_name, achievement_type, awarded_at, awarded_by, notes)
achievement_confirmations (user_id, achievement_name, requested_at, request_notes, admin_id, confirmed_at, confirmation_notes, status)

-- Admin System
admins (user_id, username, added_at, added_by, admin_level)
banned_users (user_id, username, banned_at, banned_by, reason, is_active)

-- Game Statistics
chase_games (user_id, level, target, runs, wickets, balls, result, timestamp, time_taken, perfect_chase, difficulty)
guess_games (user_id, difficulty, target_number, attempts_used, time_taken, result, score_earned, timestamp, min_range, max_range)
nightmare_games (user_id, target_number, attempts_used, time_taken, result, timestamp, min_range, max_range, hints_used)

-- Economy & Rewards
shard_transactions (user_id, amount, transaction_type, reason, timestamp, admin_id)
daily_leaderboard_entries (date, user_id, username, shards_earned, rank, created_at)
daily_goat (date, user_id, username, total_shards, created_at)

-- Cooldown Management
cooldowns (user_id, command_type, last_used)

-- Message Tracking
chats (chat_id, chat_type, title, username, first_seen, last_seen)
```

### **Key Features & Optimizations**

#### **Thread Safety**
- `ThreadSafeDict` for concurrent cache access
- `ThreadSafeCounter` for usage tracking
- `asyncio.Lock()` for auction bidding race conditions
- `threading.RLock()` for nested operations

#### **Smart Caching**
- **Roast Cache**: 5-minute duration with usage tracking
- **Leaderboard Cache**: 1-minute duration with invalidation
- **Profile Cache**: 2-minute duration with data hashing
- **GOAT Cache**: 1-hour duration for daily GOAT
- Manual invalidation on database operations

#### **Database Optimization**
- Connection pooling (5-25 connections)
- Parameterized queries for SQL injection prevention
- Automatic reconnection on failure
- Transaction management with rollback

#### **Error Handling**
- `@db_query()` decorator for database operations
- `@check_banned()` decorator for user validation
- `safe_send()` wrapper for Telegram API calls
- Comprehensive exception logging

#### **Migration System**
- Version-tracked database migrations
- Checksum validation
- Automatic table creation
- Rollback support

## 🎯 Usage Examples

### **Complete Auction Flow**

```
1. User: /register
   → Bot asks for auction details (name, teams, purse, base price)
   → Creates proposal awaiting admin approval

2. Admin: Receives approval notification with buttons
   → Clicks "Approve" or "Reject"
   → Auction moves to active state

3. Host: /hostpanel
   → Opens control panel with all options
   → /setgc - Sets the auction group chat
   → /regcap opens captain registration
   → /regplay opens player registration

4. Captains: /regcap
   → Enter team name
   → Awaits host approval

5. Players: /regplay
   → Automatically registered
   → Awaits host approval

6. Host: Approves captains and players via panel
   → Clicks "Start Bidding" when ready
   → Bot shows current player with base price

7. Captains: Type numbers in chat (1, 2, 5, 10, etc.)
   → Bot updates highest bid in real-time
   → 30-second countdown timer

8. Host: /sell - Sells player to highest bidder
   → Player assigned to captain's team
   → Purse updated
   → Next player shown automatically

9. Final: All players sold or marked unsold
   → /myteam shows complete team roster
   → /status shows auction summary
```

### **Gaming Progression Example**

```
Day 1:
/start          → Get 1000 welcome shards
/daily          → Claim 1 shard (start streak)
/guess          → Choose difficulty: medium
                → Win game, earn 2 shards
/profile        → View updated stats

Day 2:
/daily          → Claim 1 shard (2-day streak)
/chase          → Play hand cricket
                → Complete level 5, earn achievement
                → Get 5 bonus shards for achievement
/achievements   → View "Chase Champion" badge

Day 7:
/daily          → Claim 1 shard (7-day streak maintained)
/shardlb        → Check ranking (top 10!)
/nightmare      → Attempt ultimate challenge
                → Set new record
/goat           → View today's GOAT player
```

## � Deployment Guide

### **Local Development**

```bash
# 1. Clone and setup
git clone https://github.com/Krutik08062/SPL.git
cd SPL

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup environment
cp .env.example .env
# Edit .env with your credentials

# 4. Run bot
python bot.py
```

### **Cloud Deployment (Render)**

See [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md) for detailed deployment guide.

Quick steps:
1. Fork this repository
2. Create new Web Service on Render
3. Connect your GitHub repository
4. Set environment variables in Render dashboard
5. Deploy automatically

**Required Environment Variables:**
```
BOT_TOKEN=your_telegram_bot_token
DATABASE_URL=your_postgresql_url
SUPER_ADMIN_ID=your_telegram_id
ADMIN_IDS=comma,separated,admin,ids
```

### **Docker Deployment**

```bash
# Build image
docker build -t arena-of-champions .

# Run container
docker run -d \
  --name arena-bot \
  --env-file .env \
  -p 8080:8080 \
  arena-of-champions

# View logs
docker logs -f arena-bot

# Stop container
docker stop arena-bot
```

### **Railway Deployment**

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Add environment variables
railway variables set BOT_TOKEN=your_token
railway variables set DATABASE_URL=your_db_url

# Deploy
railway up
```

## 🔧 Configuration & Customization

### **Modifying Required Channels**

Edit `bot.py` around line 1870:

```python
self.required_channels = [
    {
        'username': 'YourChannelUsername',  # Without @
        'name': 'Your Channel Display Name',
        'url': 'https://t.me/YourChannel'
    },
    {
        'username': 'YourGroupUsername',
        'name': 'Your Group Name', 
        'url': 'https://t.me/YourGroup'
    }
]
```

### **Adjusting Game Difficulty**

Edit `bot.py` around line 1900:

```python
self.guess_difficulties = {
    'beginner': {'range': (1, 20), 'attempts': 6, 'multiplier': 1.0},
    'easy': {'range': (1, 50), 'attempts': 8, 'multiplier': 1.2},
    'medium': {'range': (1, 100), 'attempts': 7, 'multiplier': 1.5},
    'hard': {'range': (1, 200), 'attempts': 8, 'multiplier': 2.0},
    'expert': {'range': (1, 500), 'attempts': 10, 'multiplier': 3.0}
}
```

### **Modifying Shard Rewards**

Edit `bot.py` Constants class:

```python
class Constants:
    SHARDS_PER_NEW_ACHIEVEMENT = 5
    SHARDS_PER_GAME_WIN = 2
    SHARDS_PER_DAILY_LOGIN = 1
```

### **Customizing Cache Duration**

```python
self.roast_cache = ThreadSafeDict({
    'cache_duration': 300,  # 5 minutes
})

self.leaderboard_cache = ThreadSafeDict({
    'cache_duration': 60,  # 1 minute
})

self.profile_cache = {
    'cache_duration': 120,  # 2 minutes
}
```

## 🔧 Troubleshooting

### **Common Issues**

#### **Bot Not Starting**
```bash
# Check Python version
python --version  # Should be 3.8+

# Verify dependencies
pip install -r requirements.txt

# Check environment variables
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('BOT_TOKEN'))"
```

#### **Database Connection Errors**
```bash
# Verify DATABASE_URL format
# PostgreSQL: postgresql://user:password@host:port/database
# Example: postgresql://postgres:password@localhost:5432/arena_bot

# Test database connection
psql postgresql://user:password@host:port/database
```

#### **Channel Membership Not Working**
- Ensure bot is admin in both channels/groups
- Verify channel usernames are correct (without @)
- Check bot has permission to access channel member list
- Test with: `/start` → Should show join buttons if not member

#### **Commands Not Responding**
```bash
# Check bot logs for errors
python bot.py 2>&1 | tee bot.log

# Verify bot token
# Talk to @BotFather and use /token

# Check if bot is running
ps aux | grep python
```

#### **Admin Commands Not Working**
- Verify your Telegram ID is in `SUPER_ADMIN_ID` or `ADMIN_IDS`
- Get your ID: Send `/start` to @userinfobot
- Restart bot after changing environment variables

#### **Auction System Issues**
- Ensure group chat is set: `/setgc`
- Bot must be admin in the auction group
- Check if players/captains are approved via `/hostpanel`
- Verify purse amounts are sufficient for bidding

### **Getting Help**

1. **Check Logs**: Look for error messages in console output
2. **Database Issues**: Verify connection string and credentials
3. **Bot API Issues**: Check @BotSupport on Telegram
4. **GitHub Issues**: Open an issue with full error logs

### **Debug Mode**

Enable detailed logging in `bot.py`:

```python
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # Change from INFO to DEBUG
)
```

## 📊 Performance & Monitoring

### **Performance Metrics**
- **Concurrent Users**: Handles 1000+ simultaneous users
- **Response Time**: <500ms for most commands
- **Database Queries**: Optimized with connection pooling
- **Memory Usage**: ~200-500MB depending on active users
- **CPU Usage**: Low (<10%) during normal operation

### **Monitoring Commands**
```bash
/botstatus      # Bot health check
/cstatus        # Cache statistics
/tstatus        # Thread safety status
/adminstatus    # Admin system status
```

### **Database Maintenance**
```bash
# Cleanup old games
/cleanupchase
/cleanupguess

# Clear caches
/cleancache

# View transactions
/transactions

# Backup database
/backup
```

### **System Requirements**
- **RAM**: Minimum 512MB, Recommended 1GB
- **CPU**: 1 vCPU minimum
- **Storage**: 1GB for database
- **Network**: Stable internet connection

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### **Development Setup**

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/SPL.git
cd SPL

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and test
python bot.py

# Commit changes
git add .
git commit -m 'Add amazing feature'

# Push to your fork
git push origin feature/amazing-feature

# Open Pull Request on GitHub
```

### **Contribution Guidelines**
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Test thoroughly before submitting
- Update README if adding new features
- Keep commits atomic and well-described

### **Priority Areas**
- 🐛 Bug fixes and error handling
- 📈 Performance optimizations
- 🎨 UI/UX improvements
- 📝 Documentation updates
- 🌐 Internationalization support
- ✨ New game modes or features

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Arena Of Champions

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

## 🎉 Community & Support

### **Get Help**
- 📖 [Documentation](https://github.com/Krutik08062/SPL/wiki)
- 🐛 [Report Issues](https://github.com/Krutik08062/SPL/issues)
- 💬 [Discussions](https://github.com/Krutik08062/SPL/discussions)
- 📧 Email: support@example.com

### **Stay Updated**
- ⭐ Star the repository
- 👁️ Watch for updates
- 🔔 Enable notifications

### **Official Channels**
- 📢 Official Channel: https://t.me/SagaArenaOfficial
- 💬 Community Group: https://t.me/SagaArenaChat

---

## 🏆 **Welcome to the Arena Of Champions!**

*Where every game matters, every bid counts, and every player is a champion! 🎮⚡*

### **Quick Links**
- 🤖 [Bot Demo](https://t.me/your_bot_username)
- 📖 [Full Documentation](https://github.com/Krutik08062/SPL/wiki)
- 🎬 [Video Tutorial](#)
- 💻 [Source Code](https://github.com/Krutik08062/SPL)

### **Statistics**
- 📊 18,000+ Lines of Code
- 🎮 3 Game Modes
- 🏆 20+ Achievements
- 💎 Complete Economy System
- 🎪 Full IPL-Style Auctions
- 👨‍💼 Advanced Admin Tools

**Built with ❤️ for Gaming & Cricket Communities**

---

<div align="center">

**[⬆ Back to Top](#-arena-of-champions-bot)**

Made with 💖 by the Arena Of Champions Team

</div>

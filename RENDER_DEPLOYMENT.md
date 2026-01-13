# 🎄 SPL Achievement Bot - Render.com Deployment Guide

## ✅ What We've Set Up

Your bot is now configured for 24/7 hosting on Render.com with automatic keep-alive!

### Files Added/Modified:

1. ✅ **web.py** - Web server with automatic ping system (pings every 3 minutes)
2. ✅ **bot.py** - Updated with web server integration
3. ✅ **requirements.txt** - Already has all dependencies

---

## 🚀 Deployment Steps

### Step 1: Prepare Your Repository

1. Make sure all changes are committed to GitHub:
   ```bash
   git add .
   git commit -m "Add Render deployment support"
   git push origin main
   ```

### Step 2: Deploy on Render.com

1. Go to https://dashboard.render.com/
2. Click **"New +"** → Select **"Web Service"**
3. Connect your GitHub account (if not already connected)
4. Select your repository: **Krutik08062/SPL**
5. Configure the following settings:

   **Basic Settings:**
   - Name: `spl-achievement-bot` (or any name you prefer)
   - Region: Choose closest to you
   - Branch: `main`
   - Root Directory: `./` (leave empty or use `./`)
   - Runtime: `Python 3`

   **Build & Deploy:**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python bot.py`

   **Environment Variables:**
   Click "Add Environment Variable" and add these:
   
   | Key | Value |
   |-----|-------|
   | `BOT_TOKEN` | Your Telegram Bot Token |
   | `DATABASE_URL` | Your Supabase PostgreSQL URL |
   | `SUPER_ADMIN_ID` | Your Telegram User ID |
   | `PORT` | `8080` |
   | `PYTHON_VERSION` | `3.11.0` |

   **Instance Type:**
   - Select **Free** tier

6. Click **"Create Web Service"**

### Step 3: Update Web URL

1. After deployment, Render will give you a URL like:
   ```
   https://spl-achievement-bot.onrender.com
   ```

2. Copy this URL and update `web.py` line 1:
   ```python
   WEB_URL = "https://your-actual-render-url.onrender.com/"
   ```

3. Commit and push the change:
   ```bash
   git add web.py
   git commit -m "Update Render URL"
   git push origin main
   ```

4. Render will automatically redeploy with the new URL

---

## 🎯 How It Works

### Keep-Alive System:
- **Web Server**: Runs on port 8080, responds to HTTP requests
- **Auto-Ping**: Every 3 minutes, the bot pings itself at the Render URL
- **No Sleep**: This prevents Render's free tier from shutting down due to inactivity
- **24/7 Uptime**: Your bot stays online continuously!

### What Happens:
```
┌─────────────────┐
│   Render.com    │
│                 │
│  ┌──────────┐   │     Every 3 min
│  │ Web Server│◄──┼─────────────────┐
│  │  Port 8080│   │                 │
│  └─────┬────┘   │                 │
│        │        │            ┌────▼────┐
│  ┌─────▼────┐   │            │  Ping   │
│  │Telegram  │   │            │ System  │
│  │   Bot    │   │            └─────────┘
│  └──────────┘   │
└─────────────────┘
```

---

## 🔧 Troubleshooting

### If Bot Doesn't Start:

1. **Check Logs** on Render Dashboard:
   - Click on your service
   - Go to "Logs" tab
   - Look for error messages

2. **Common Issues**:
   
   **Missing Environment Variables:**
   ```
   Error: BOT_TOKEN environment variable is required
   ```
   → Add BOT_TOKEN in Render environment variables

   **Database Connection Failed:**
   ```
   Error: could not connect to server
   ```
   → Check DATABASE_URL is correct
   → Ensure Supabase is accessible

   **Port Already in Use:**
   → Render handles this automatically, but ensure PORT=8080

3. **Manual Restart**:
   - Go to Render Dashboard
   - Click "Manual Deploy" → "Clear build cache & deploy"

### If Bot Keeps Sleeping:

1. Verify `web.py` has correct URL:
   ```python
   WEB_URL = "https://your-app-name.onrender.com/"
   ```

2. Check logs for ping messages:
   ```
   Pinged https://your-app-name.onrender.com/ with response: 200
   ```

3. If pings fail, the URL might be wrong

---

## 📊 Monitoring

### Check Bot Status:

1. **Render Dashboard**:
   - Green dot = Running
   - Red dot = Stopped/Error
   - Yellow dot = Deploying

2. **Telegram**:
   - Send `/ping` to your bot
   - Should respond instantly if online

3. **Web Browser**:
   - Visit: `https://your-app-name.onrender.com/`
   - Should see: "SPL Achievement Bot is running! 🏏"

---

## 💡 Pro Tips

1. **First Deployment Takes Time**: 5-10 minutes for initial setup
2. **Auto-Deploys**: Every git push triggers a new deployment
3. **Free Tier Limits**: 
   - 750 hours/month (enough for 24/7 if you have one service)
   - Shared CPU/RAM
   - May slow down under heavy load
4. **Logs**: Keep checking logs for first 24 hours to ensure stability

---

## 🎉 Success Checklist

- ✅ Repository pushed to GitHub
- ✅ Render web service created
- ✅ Environment variables configured
- ✅ Bot deployed successfully
- ✅ Web URL updated in `web.py`
- ✅ Bot responding to commands
- ✅ Logs show successful pings every 3 minutes
- ✅ Web page accessible and shows bot status

---

## 🆘 Need Help?

1. Check Render logs for errors
2. Verify all environment variables are set
3. Test database connection separately
4. Make sure GitHub repository is up to date
5. Try manual deploy with "Clear build cache"

---

## 📝 Environment Variables Reference

```env
# Required
BOT_TOKEN=your_bot_token_here
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Optional but Recommended
SUPER_ADMIN_ID=your_telegram_id
PORT=8080
PYTHON_VERSION=3.11.0
```

---

**Your bot is now ready for 24/7 deployment! 🚀**

Good luck with your live SPL auction! 🏏🎯

import os
import asyncio
import aiohttp
import logging
import traceback
from aiohttp import web

log = logging.getLogger(__name__)

# Render service URL
WEB_URL = os.getenv("WEB_URL") or os.getenv("RENDER_EXTERNAL_URL") or "https://arena-of-champions.onrender.com/"
if not WEB_URL.endswith('/'):
    WEB_URL += '/'

WEB_SLEEP = 3 * 60  # Ping every 3 minutes (Render sleeps after 15 minutes)

routes = web.RouteTableDef()

@routes.get('/', allow_head=True)
@routes.get('/health', allow_head=True)
@routes.get('/ping', allow_head=True)
async def hello(request):
    return web.Response(text="SPL Achievement Bot is running! 🏏\nStatus: OK", content_type="text/plain")

def web_server():
    app = web.Application()
    app.add_routes(routes)
    return app

async def keep_alive():
    """Background task that pings the Render service URL to keep it awake"""
    if not WEB_URL:
        print("⚠️ [Keep-Alive] No WEB_URL provided. Keep-alive disabled.", flush=True)
        return

    print(f"⏰ [Keep-Alive] Initialized. Will ping {WEB_URL} every {WEB_SLEEP}s.", flush=True)
    # Give the web server and bot a few seconds to start before first ping
    await asyncio.sleep(15)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    while True:
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers=headers
            ) as session:
                async with session.get(WEB_URL) as resp:
                    status = resp.status
                    msg = f"⏰ [Keep-Alive] Pinged {WEB_URL} - Status: {status} OK"
                    log.info(msg)
                    print(msg, flush=True)
        except asyncio.TimeoutError:
            msg = f"⚠️ [Keep-Alive] Timeout connecting to {WEB_URL}"
            log.warning(msg)
            print(msg, flush=True)
        except Exception as e:
            msg = f"❌ [Keep-Alive] Ping failed: {e}"
            log.error(msg)
            print(msg, flush=True)

        await asyncio.sleep(WEB_SLEEP)


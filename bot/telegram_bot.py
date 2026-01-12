from __future__ import annotations

import asyncio
import logging
import os
import io
import requests
import yfinance as yf
import ta
import anthropic
import os
from weather import get_weather, get_forecast

claude_client = None

def get_claude_client():
    global claude_client
    if claude_client is None:
        claude_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return claude_client

from utils import summarize_url, fetch_page_with_playwright  # ✅ thêm hàm mới
from uuid import uuid4
from telegram import BotCommandScopeAllGroupChats, Update, constants
from telegram import InlineKeyboardMarkup, InlineKeyboardButton, InlineQueryResultArticle
from telegram import InputTextMessageContent, BotCommand
from telegram.error import RetryAfter, TimedOut, BadRequest
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, \
    filters, InlineQueryHandler, CallbackQueryHandler, Application, ContextTypes, CallbackContext

from pydub import AudioSegment
from PIL import Image

from image_diff import draw_bugs_on_image, format_bug_report, create_ssim_diff, create_edge_comparison

from utils import is_group_chat, get_thread_id, message_text, wrap_with_indicator, split_into_chunks, \
    edit_message_with_retry, get_stream_cutoff_values, is_allowed, get_remaining_budget, is_admin, is_within_budget, \
    get_reply_to_message_id, add_chat_request_to_usage_tracker, error_handler, is_direct_result, handle_direct_result, \
    cleanup_intermediate_files
from openai_helper import OpenAIHelper, localized_text
from usage_tracker import UsageTracker

import requests
import yfinance as yf
import ta

from utils import summarize_url, fetch_page_with_playwright  # ✅ thêm hàm mới



async def extract_city_from_text(text: str) -> str:
    """
    Dùng Claude để phân tích text và trích xuất city.
    Nếu không có city rõ ràng → trả về chuỗi rỗng "".
    """
    prompt = (
        f"Người dùng hỏi: \"{text}\"\n\n"
        "Trong câu trên, nếu có nhắc đến tên một thành phố hoặc tỉnh thành ở Việt Nam, "
        "hãy trả lại đúng tên thành phố đó (chỉ 1 từ, không giải thích gì thêm).\n"
        "Nếu không rõ hoặc không có địa danh, chỉ trả về \"\" (chuỗi rỗng)."
    )
    try:
        response = await get_claude_client().messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        city = response.content[0].text.strip()
        return city
    except Exception:
        return ""
async def should_respond_weather(update, context) -> Union[bool, str]:
    """
    ✅ Trả về:
    - False: nếu không đủ điều kiện
    - City string: nếu đủ điều kiện
    """
    text = (update.message.text or "").lower()

    if update.effective_chat.type in [constants.ChatType.GROUP, constants.ChatType.SUPERGROUP]:
        bot_username = context.bot.username.lower()

        is_mentioned = f"@{bot_username}" in text
        is_reply_to_bot = (
            update.message.reply_to_message
            and update.message.reply_to_message.from_user.id == context.bot.id
        )

        if not (is_mentioned or is_reply_to_bot):
            return False  # 🚫 Không mention cũng không reply → không xử lý

        if "thời tiết" not in text and "dự báo" not in text:
            return False  # 🚫 Không nói về thời tiết → không xử lý

    # --- Nếu qua hết => Gọi GPT extract city
    city = await extract_city_from_text(update.message.text)

    # Nếu không rõ city → fallback mặc định Hồ Chí Minh
    return city if city else "hồ chí minh"




def find_coin_id_by_symbol(symbol):
    try:
        url = "https://api.coingecko.com/api/v3/coins/list"
        coins = requests.get(url).json()
        for coin in coins:
            if coin["symbol"].lower() == symbol.lower():
                return coin["id"]
    except:
        return None
    return None
def get_price_from_coingecko(coin_id="bitcoin"):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    try:
        res = requests.get(url).json()
        if coin_id in res and "usd" in res[coin_id]:
            price = res[coin_id]["usd"]
            return f"💰 {coin_id.replace('-', ' ').title()}: ${price}"
        else:
            return f"❌ Không lấy được giá của {coin_id}."
    except Exception as e:
        return f"❌ Lỗi khi lấy giá: {str(e)}"

class ChatGPTTelegramBot:
    """
    Class representing a ChatGPT Telegram Bot.
    """
    def stop(self):
        """
        Stops the bot and performs cleanup
        """
        if hasattr(self, 'application'):
            self.application.stop()

    def __init__(self, config: dict, openai: OpenAIHelper):
        """
        Initializes the bot with the given configuration and GPT bot object.
        :param config: A dictionary containing the bot configuration
        :param openai: OpenAIHelper object
        """
        self.config = config
        self.openai = openai
        bot_language = self.config['bot_language']
        self.commands = [
            BotCommand(command='check', description='Bắt đầu check bug giữa 2 hình'),
            BotCommand(command='feedback', description='Feedback design như Steve Jobs'),
            BotCommand(command='reset', description=localized_text('reset_description', bot_language)),
        ]
        # If imaging is enabled, add the "image" command to the list
        #if self.config.get('enable_image_generation', False):
            #self.commands.append(BotCommand(command='image', description=localized_text('image_description', bot_language)))

        #if self.config.get('enable_tts_generation', False):
            #self.commands.append(BotCommand(command='tts', description=localized_text('tts_description', bot_language)))

        self.group_commands = [BotCommand(
            command='chat', description=localized_text('chat_description', bot_language)
        )] + self.commands
        self.disallowed_message = localized_text('disallowed', bot_language)
        self.budget_limit_message = localized_text('budget_limit', bot_language)
        self.usage = {}
        self.last_message = {}
        self.inline_queries_cache = {}
        self.pending_compare = {}  # Store comparison state: {chat_id: {'state': 'waiting_dev'|'waiting_design', 'dev_image': ...}}
        self.last_compare = {}  # Store last compared images for re-check: {chat_id: {'dev': bytes, 'design': bytes}}
        self.pending_feedback = set()  # Store chat_ids waiting for feedback image

    async def summarize_and_reply(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Tự động tóm tắt nội dung:
        - Nếu người dùng nhắc tới 'tóm tắt' và có URL trong tin nhắn bị reply.
        - Hoặc trong tin nhắn hiện tại có URL và chứa 'tóm tắt'.
        """
        prompt = message_text(update.message) or ""
        text = prompt.lower()
    
        # ✅ Nếu trong nhóm, phải @mention hoặc reply thì mới cho chạy
        if update.effective_chat.type in [constants.ChatType.GROUP, constants.ChatType.SUPERGROUP]:
            bot_username = context.bot.username.lower()
            is_mentioned = f"@{bot_username}" in text
            is_reply_to_bot = (
                update.message.reply_to_message
                and update.message.reply_to_message.from_user.id == context.bot.id
            )
            if not is_mentioned and not is_reply_to_bot:
                return False  # 🚫 Không chạy nếu không mention hoặc reply
    
        url = None
    
        # Nếu đang reply tin nhắn khác, tìm URL trong tin nhắn bị reply
        if update.message.reply_to_message:
            reply_text = message_text(update.message.reply_to_message) or ""
            url = next((word for word in reply_text.split() if word.startswith("http")), None)
    
        # Nếu không tìm thấy URL ở tin nhắn bị reply, tìm trong tin nhắn hiện tại
        if not url:
            url = next((word for word in prompt.split() if word.startswith("http")), None)
    
        # Điều kiện phải có "tóm tắt" và có URL
        if "tóm tắt" in text and url:
            try:
                await update.message.reply_chat_action(action=constants.ChatAction.TYPING)
                summary = await summarize_url(url, update, context)
                if summary and isinstance(summary, str) and summary.strip():
                    await update.message.reply_text(summary[:4096])
                return True
            except Exception as e:
                await update.message.reply_text(f"❌ Lỗi khi tóm tắt: {e}")
                return True
    
        return False



    async def help(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
                """
                Shows the help menu.
                """
                help_text = """🤖 **CHÀO MỪNG ĐẾN VỚI SOI BUG BOT!**

Tao là bot soi bug UI của anh @kieumanhhuy đẹp trai tạo ra 😎

━━━━━━━━━━━━━━━━━━━━━━
🎯 **TAO LÀM GÌ?**
Soi từng pixel DEV vs DESIGN, tìm bug như tìm mụn trên mặt vậy đó! 👀

━━━━━━━━━━━━━━━━━━━━━━
📋 **CÁCH DÙNG:**

1️⃣ Gửi `/check` để bắt đầu
2️⃣ Quăng hình **DEV** (hình cần check)
3️⃣ Quăng hình **DESIGN** (hình chuẩn)
4️⃣ Chờ tao soi và báo bug 🔍

━━━━━━━━━━━━━━━━━━━━━━
🔧 **LỆNH:**
/check - Bắt đầu soi bug
/reset - Huỷ bỏ, làm lại

━━━━━━━━━━━━━━━━━━━━━━
🎨 **TAO SOI GÌ?**
• SPACING - Khoảng cách, padding
• ALIGNMENT - Căn chỉnh, thẳng hàng
• COLOR - Màu sắc

━━━━━━━━━━━━━━━━━━━━━━
💡 Có bug gì thì hỏi ông chủ @kieumanhhuy nha!
💼 Có job design thì liên hệ ông @kieumanhhuy đi, ổng đang đói lắm 😭🍚

**LET'S GO SOI BUG! 🚀**
"""
                await update.message.reply_text(help_text, parse_mode='Markdown', disable_web_page_preview=True)

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Returns token usage statistics for current day and month.
        """
        if not await is_allowed(self.config, update, context):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                            'is not allowed to request their usage statistics')
            await self.send_disallowed_message(update, context)
            return

        logging.info(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                     'requested their usage statistics')

        user_id = update.message.from_user.id
        if user_id not in self.usage:
            self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

        tokens_today, tokens_month = self.usage[user_id].get_current_token_usage()
        images_today, images_month = self.usage[user_id].get_current_image_count()
        (transcribe_minutes_today, transcribe_seconds_today, transcribe_minutes_month,
         transcribe_seconds_month) = self.usage[user_id].get_current_transcription_duration()
        vision_today, vision_month = self.usage[user_id].get_current_vision_tokens()
        characters_today, characters_month = self.usage[user_id].get_current_tts_usage()
        current_cost = self.usage[user_id].get_current_cost()

        chat_id = update.effective_chat.id
        chat_messages, chat_token_length = self.openai.get_conversation_stats(chat_id)
        remaining_budget = get_remaining_budget(self.config, self.usage, update)
        bot_language = self.config['bot_language']

        text_current_conversation = (
            f"*{localized_text('stats_conversation', bot_language)[0]}*:\n"
            f"{chat_messages} {localized_text('stats_conversation', bot_language)[1]}\n"
            f"{chat_token_length} {localized_text('stats_conversation', bot_language)[2]}\n"
            "----------------------------\n"
        )

        # Check if image generation is enabled and, if so, generate the image statistics for today
        text_today_images = ""
        if self.config.get('enable_image_generation', False):
            text_today_images = f"{images_today} {localized_text('stats_images', bot_language)}\n"

        text_today_vision = ""
        if self.config.get('enable_vision', False):
            text_today_vision = f"{vision_today} {localized_text('stats_vision', bot_language)}\n"

        text_today_tts = ""
        if self.config.get('enable_tts_generation', False):
            text_today_tts = f"{characters_today} {localized_text('stats_tts', bot_language)}\n"

        text_today = (
            f"*{localized_text('usage_today', bot_language)}:*\n"
            f"{tokens_today} {localized_text('stats_tokens', bot_language)}\n"
            f"{text_today_images}"  # Include the image statistics for today if applicable
            f"{text_today_vision}"
            f"{text_today_tts}"
            f"{transcribe_minutes_today} {localized_text('stats_transcribe', bot_language)[0]} "
            f"{transcribe_seconds_today} {localized_text('stats_transcribe', bot_language)[1]}\n"
            f"{localized_text('stats_total', bot_language)}{current_cost['cost_today']:.2f}\n"
            "----------------------------\n"
        )

        text_month_images = ""
        if self.config.get('enable_image_generation', False):
            text_month_images = f"{images_month} {localized_text('stats_images', bot_language)}\n"

        text_month_vision = ""
        if self.config.get('enable_vision', False):
            text_month_vision = f"{vision_month} {localized_text('stats_vision', bot_language)}\n"

        text_month_tts = ""
        if self.config.get('enable_tts_generation', False):
            text_month_tts = f"{characters_month} {localized_text('stats_tts', bot_language)}\n"

        # Check if image generation is enabled and, if so, generate the image statistics for the month
        text_month = (
            f"*{localized_text('usage_month', bot_language)}:*\n"
            f"{tokens_month} {localized_text('stats_tokens', bot_language)}\n"
            f"{text_month_images}"  # Include the image statistics for the month if applicable
            f"{text_month_vision}"
            f"{text_month_tts}"
            f"{transcribe_minutes_month} {localized_text('stats_transcribe', bot_language)[0]} "
            f"{transcribe_seconds_month} {localized_text('stats_transcribe', bot_language)[1]}\n"
            f"{localized_text('stats_total', bot_language)}{current_cost['cost_month']:.2f}"
        )

        # text_budget filled with conditional content
        text_budget = "\n\n"
        budget_period = self.config['budget_period']
        if remaining_budget < float('inf'):
            text_budget += (
                f"{localized_text('stats_budget', bot_language)}"
                f"{localized_text(budget_period, bot_language)}: "
                f"${remaining_budget:.2f}.\n"
            )
        # No longer works as of July 21st 2023, as OpenAI has removed the billing API
        # add OpenAI account information for admin request
        # if is_admin(self.config, user_id):
        #     text_budget += (
        #         f"{localized_text('stats_openai', bot_language)}"
        #         f"{self.openai.get_billing_current_month():.2f}"
        #     )

        usage_text = text_current_conversation + text_today + text_month + text_budget
        await update.message.reply_text(usage_text, parse_mode=constants.ParseMode.MARKDOWN)

    async def resend(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Resend the last request
        """
        if not await is_allowed(self.config, update, context):
            logging.warning(f'User {update.message.from_user.name}  (id: {update.message.from_user.id})'
                            ' is not allowed to resend the message')
            await self.send_disallowed_message(update, context)
            return

        chat_id = update.effective_chat.id
        if chat_id not in self.last_message:
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id})'
                            ' does not have anything to resend')
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=localized_text('resend_failed', self.config['bot_language'])
            )
            return

        # Update message text, clear self.last_message and send the request to prompt
        logging.info(f'Resending the last prompt from user: {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})')
        with update.message._unfrozen() as message:
            message.text = self.last_message.pop(chat_id)

        await self.prompt(update=update, context=context)

    async def check(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Start the bug check flow - compare DEV vs DESIGN images.
        """
        chat_id = update.effective_chat.id

        # Clear feedback mode if active
        self.pending_feedback.discard(chat_id)

        # Initialize comparison state
        self.pending_compare[chat_id] = {
            'state': 'waiting_dev',
            'dev_image': None,
            'design_image': None
        }

        await update.effective_message.reply_text(
            "🔍 **SOI BUG MODE ON!**\n\n"
            "📤 Quăng hình **DEV** (hình cần soi) vô đây đi!\n\n"
            "💡 Gửi /reset nếu đổi ý\n"
            "🤖 Bot by @kieumanhhuy",
            parse_mode='Markdown'
        )
        logging.info(f'Started check flow for chat {chat_id}')

    async def recheck(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Re-check the last compared images when user asks to check again.
        """
        chat_id = update.effective_chat.id

        if chat_id not in self.last_compare:
            await update.effective_message.reply_text(
                "🤔 Chưa có hình nào để check lại!\n\n"
                "📋 Gửi /check → DEV → DESIGN để bắt đầu nha!"
            )
            return

        await update.effective_message.reply_text("🔄 OK check lại nha! Chờ tí...")

        dev_image = self.last_compare[chat_id]['dev']
        design_image = self.last_compare[chat_id]['design']
        dev_image.seek(0)
        design_image.seek(0)

        logging.info(f'Re-checking images for chat {chat_id}')

        # Run comparison again using pixelmatch
        from image_diff import draw_bugs_on_image, format_bug_report, create_pixelmatch_diff

        pixelmatch_diff, diff_count, grouped_regions, shift_analysis = create_pixelmatch_diff(
            dev_image, design_image, threshold=0.1
        )

        # If no differences, report immediately
        if diff_count == 0 or not grouped_regions:
            import random
            comments = [
                "✅ Check lại vẫn 0 bug! Pixelmatch xác nhận 🔥",
                "✅ Vẫn perfect! Không có khác biệt nào 💯",
            ]
            await update.effective_message.reply_text(random.choice(comments))
            return

        analysis_info = f"📊 Pixelmatch: {diff_count} pixels khác biệt\n"
        analysis_info += f"Phát hiện {len(grouped_regions)} vùng khác biệt.\n"

        dev_image.seek(0)
        design_image.seek(0)

        try:
            bugs = await self.openai.analyze_images_for_bugs(
                dev_image, design_image, analysis_info,
                pixelmatch_diff_bytes=pixelmatch_diff,
                shift_analysis=shift_analysis,
                grouped_regions=grouped_regions
            )

            if bugs:
                dev_image.seek(0)
                annotated_image = draw_bugs_on_image(dev_image, bugs)
                if annotated_image:
                    bug_report = format_bug_report(bugs)
                    await update.effective_message.reply_photo(photo=annotated_image, caption=bug_report)
                else:
                    await update.effective_message.reply_text(format_bug_report(bugs))
            else:
                import random
                comments = [
                    "✅ Check lại vẫn 0 bug! Dev ngon thiệt 🔥",
                    "✅ Vẫn perfect! Không có gì mới đâu 💯",
                ]
                await update.effective_message.reply_text(random.choice(comments))
        except Exception as e:
            logging.error(f"Error in recheck: {e}")
            await update.effective_message.reply_text(f"❌ Lỗi khi check lại: {e}")

    async def feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Start feedback mode - critique design like Steve Jobs.
        """
        chat_id = update.effective_chat.id

        # Clear check mode if active
        if chat_id in self.pending_compare:
            del self.pending_compare[chat_id]

        self.pending_feedback.add(chat_id)

        await update.effective_message.reply_text(
            "🎯 **STEVE JOBS MODE ON!**\n\n"
            "📤 Quăng design vô đây để tao chê như Steve Jobs!\n\n"
            "⚠️ Chuẩn bị tinh thần nghe feedback thật nhé 😤\n"
            "💡 Gửi /reset nếu đổi ý",
            parse_mode='Markdown'
        )
        logging.info(f'Started feedback mode for chat {chat_id}')

    async def give_steve_jobs_feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE, image_bytes):
        """
        Give Steve Jobs style feedback on a design.
        """
        chat_id = update.effective_chat.id
        self.pending_feedback.discard(chat_id)

        await update.effective_message.reply_text("🧐 Để tao nhìn cái design này như Steve Jobs...")

        steve_jobs_prompt = """Bạn là Steve Jobs - người có con mắt thiết kế khắt khe nhất thế giới.

TÍNH CÁCH KHI FEEDBACK:
- Cực kỳ khắt khe, đòi hỏi sự HOÀN HẢO
- Ghét sự phức tạp không cần thiết - "Simplicity is the ultimate sophistication"
- Focus vào USER EXPERIENCE - mọi thứ phải intuitive
- Không chấp nhận "good enough" - chỉ chấp nhận "insanely great"
- Thẳng thắn, không ngại hurt feelings để có sản phẩm tốt
- Hay dùng các câu iconic: "This is shit", "It's not good enough", "Start over", "Think different"

CÁCH FEEDBACK:
1. Nhìn tổng thể design và đánh giá first impression
2. Chỉ ra những điểm YẾU về:
   - Visual hierarchy - mắt người dùng nhìn vào đâu trước?
   - Simplicity - có element nào thừa không?
   - Typography - font có clean không? Spacing đúng chưa?
   - Color - màu sắc có harmony không? Có quá nhiều màu không?
   - Whitespace - có đủ "room to breathe" không?
   - User flow - người dùng có biết làm gì tiếp theo không?
3. Cho điểm từ 1-10 theo tiêu chuẩn Apple
4. Kết thúc bằng một câu motivational kiểu Steve Jobs

FORMAT:
🍎 **STEVE JOBS FEEDBACK**

**First Impression:** [Phản ứng đầu tiên - thường là harsh]

**Những điểm CẦN SỬA:**
[Liệt kê các vấn đề, mỗi điểm bắt đầu bằng ❌]

**Những điểm TẠM ĐƯỢC:**
[Nếu có điểm tốt, bắt đầu bằng ✓]

**Điểm số:** X/10 🍎

**Lời khuyên của Steve:** [Một câu quote phong cách Steve Jobs]

Hãy feedback design này như thể bạn đang review sản phẩm cho Apple. Không cần tử tế - cần THẬT.
"""

        import base64
        image_bytes.seek(0)
        image_data = base64.b64encode(image_bytes.read()).decode('utf-8')

        content = [
            {'type': 'text', 'text': steve_jobs_prompt},
            {
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': 'image/png',
                    'data': image_data
                }
            }
        ]

        try:
            response = await self.openai.claude_client.messages.create(
                model=self.config.get('vision_model', 'claude-sonnet-4-20250514'),
                max_tokens=1500,
                messages=[{'role': 'user', 'content': content}],
                temperature=0.8,
            )

            feedback_text = response.content[0].text.strip()

            # Split if too long for Telegram
            if len(feedback_text) > 4000:
                feedback_text = feedback_text[:4000] + "..."

            await update.effective_message.reply_text(feedback_text, parse_mode='Markdown')

        except Exception as e:
            logging.error(f"Error giving Steve Jobs feedback: {e}")
            await update.effective_message.reply_text(
                f"❌ Lỗi: {e}\n\nSteve Jobs would say: 'This is unacceptable!'"
            )

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Resets the conversation.
        """
        if not await is_allowed(self.config, update, context):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                            'is not allowed to reset the conversation')
            await self.send_disallowed_message(update, context)
            return

        logging.info(f'Resetting the conversation for user {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})...')

        chat_id = update.effective_chat.id
        reset_content = message_text(update.message)
        self.openai.reset_chat_history(chat_id=chat_id, content=reset_content)

        # Clear pending comparison and feedback
        if chat_id in self.pending_compare:
            del self.pending_compare[chat_id]
        self.pending_feedback.discard(chat_id)

        await update.effective_message.reply_text(
            message_thread_id=get_thread_id(update),
            text="✅ Đã reset!\n/check → Check bug\n/feedback → Steve Jobs feedback"
        )

    async def image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Generates an image for the given prompt using DALL·E APIs
        """
        if not self.config['enable_image_generation'] \
                or not await self.check_allowed_and_within_budget(update, context):
            return

        image_query = message_text(update.message)
        if image_query == '':
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=localized_text('image_no_prompt', self.config['bot_language'])
            )
            return

        logging.info(f'New image generation request received from user {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})')

        async def _generate():
            try:
                image_url, image_size = await self.openai.generate_image(prompt=image_query)
                if self.config['image_receive_mode'] == 'photo':
                    await update.effective_message.reply_photo(
                        reply_to_message_id=get_reply_to_message_id(self.config, update),
                        photo=image_url
                    )
                elif self.config['image_receive_mode'] == 'document':
                    await update.effective_message.reply_document(
                        reply_to_message_id=get_reply_to_message_id(self.config, update),
                        document=image_url
                    )
                else:
                    raise Exception(f"env variable IMAGE_RECEIVE_MODE has invalid value {self.config['image_receive_mode']}")
                # add image request to users usage tracker
                user_id = update.message.from_user.id
                self.usage[user_id].add_image_request(image_size, self.config['image_prices'])
                # add guest chat request to guest usage tracker
                if str(user_id) not in self.config['allowed_user_ids'].split(',') and 'guests' in self.usage:
                    self.usage["guests"].add_image_request(image_size, self.config['image_prices'])

            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=f"{localized_text('image_fail', self.config['bot_language'])}: {str(e)}",
                    parse_mode=constants.ParseMode.MARKDOWN
                )

        await wrap_with_indicator(update, context, _generate, constants.ChatAction.UPLOAD_PHOTO)

    async def tts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Generates an speech for the given input using TTS APIs
        """
        if not self.config['enable_tts_generation'] \
                or not await self.check_allowed_and_within_budget(update, context):
            return

        tts_query = message_text(update.message)
        if tts_query == '':
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=localized_text('tts_no_prompt', self.config['bot_language'])
            )
            return

        logging.info(f'New speech generation request received from user {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})')

        async def _generate():
            try:
                speech_file, text_length = await self.openai.generate_speech(text=tts_query)

                await update.effective_message.reply_voice(
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    voice=speech_file
                )
                speech_file.close()
                # add image request to users usage tracker
                user_id = update.message.from_user.id
                self.usage[user_id].add_tts_request(text_length, self.config['tts_model'], self.config['tts_prices'])
                # add guest chat request to guest usage tracker
                if str(user_id) not in self.config['allowed_user_ids'].split(',') and 'guests' in self.usage:
                    self.usage["guests"].add_tts_request(text_length, self.config['tts_model'], self.config['tts_prices'])

            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=f"{localized_text('tts_fail', self.config['bot_language'])}: {str(e)}",
                    parse_mode=constants.ParseMode.MARKDOWN
                )

        await wrap_with_indicator(update, context, _generate, constants.ChatAction.UPLOAD_VOICE)

    async def transcribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Transcribe audio messages.
        """
        if not self.config['enable_transcription'] or not await self.check_allowed_and_within_budget(update, context):
            return

        if is_group_chat(update) and self.config['ignore_group_transcriptions']:
            logging.info('Transcription coming from group chat, ignoring...')
            return

        chat_id = update.effective_chat.id
        filename = update.message.effective_attachment.file_unique_id

        async def _execute():
            filename_mp3 = f'{filename}.mp3'
            bot_language = self.config['bot_language']
            try:
                media_file = await context.bot.get_file(update.message.effective_attachment.file_id)
                await media_file.download_to_drive(filename)
            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=(
                        f"{localized_text('media_download_fail', bot_language)[0]}: "
                        f"{str(e)}. {localized_text('media_download_fail', bot_language)[1]}"
                    ),
                    parse_mode=constants.ParseMode.MARKDOWN
                )
                return

            try:
                audio_track = AudioSegment.from_file(filename)
                audio_track.export(filename_mp3, format="mp3")
                logging.info(f'New transcribe request received from user {update.message.from_user.name} '
                             f'(id: {update.message.from_user.id})')

            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=localized_text('media_type_fail', bot_language)
                )
                if os.path.exists(filename):
                    os.remove(filename)
                return

            user_id = update.message.from_user.id
            if user_id not in self.usage:
                self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

            try:
                transcript = await self.openai.transcribe(filename_mp3)

                transcription_price = self.config['transcription_price']
                self.usage[user_id].add_transcription_seconds(audio_track.duration_seconds, transcription_price)

                allowed_user_ids = self.config['allowed_user_ids'].split(',')
                if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                    self.usage["guests"].add_transcription_seconds(audio_track.duration_seconds, transcription_price)

                # check if transcript starts with any of the prefixes
                response_to_transcription = any(transcript.lower().startswith(prefix.lower()) if prefix else False
                                                for prefix in self.config['voice_reply_prompts'])

                if self.config['voice_reply_transcript'] and not response_to_transcription:

                    # Split into chunks of 4096 characters (Telegram's message limit)
                    transcript_output = f"_{localized_text('transcript', bot_language)}:_\n\"{transcript}\""
                    chunks = split_into_chunks(transcript_output)

                    for index, transcript_chunk in enumerate(chunks):
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(self.config, update) if index == 0 else None,
                            text=transcript_chunk,
                            parse_mode=constants.ParseMode.MARKDOWN
                        )
                else:
                    # Get the response of the transcript
                    response, total_tokens = await self.openai.get_chat_response(chat_id=chat_id, query=transcript)

                    self.usage[user_id].add_chat_tokens(total_tokens, self.config['token_price'])
                    if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                        self.usage["guests"].add_chat_tokens(total_tokens, self.config['token_price'])

                    # Split into chunks of 4096 characters (Telegram's message limit)
                    transcript_output = (
                        f"_{localized_text('transcript', bot_language)}:_\n\"{transcript}\"\n\n"
                        f"_{localized_text('answer', bot_language)}:_\n{response}"
                    )
                    chunks = split_into_chunks(transcript_output)

                    for index, transcript_chunk in enumerate(chunks):
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(self.config, update) if index == 0 else None,
                            text=transcript_chunk,
                            parse_mode=constants.ParseMode.MARKDOWN
                        )

            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=f"{localized_text('transcribe_fail', bot_language)}: {str(e)}",
                    parse_mode=constants.ParseMode.MARKDOWN
                )
            finally:
                if os.path.exists(filename_mp3):
                    os.remove(filename_mp3)
                if os.path.exists(filename):
                    os.remove(filename)

        await wrap_with_indicator(update, context, _execute, constants.ChatAction.TYPING)

    async def vision(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Interpret image using vision model. Supports 2-image comparison.
        Send 2 images (one at a time) to compare design vs dev.
        """
        if not self.config['enable_vision'] or not await self.check_allowed_and_within_budget(update, context):
            return

        chat_id = update.effective_chat.id
        prompt = update.message.caption

        if is_group_chat(update):
            bot_username = context.bot.username.lower()
            caption = update.message.caption.lower() if update.message.caption else ""

            is_mentioned = f"@{bot_username}" in caption
            is_reply_to_bot = (
                update.message.reply_to_message
                and update.message.reply_to_message.from_user.id == context.bot.id
            )

            if not is_mentioned and not is_reply_to_bot:
                logging.info("Vision: group chat without mention or reply — skipping.")
                return

        image = update.message.effective_attachment[-1]

        # Download image
        try:
            media_file = await context.bot.get_file(image.file_id)
            temp_file = io.BytesIO(await media_file.download_as_bytearray())
            original_image = Image.open(temp_file)
            temp_file_png = io.BytesIO()
            original_image.save(temp_file_png, format='PNG')
        except Exception as e:
            logging.exception(e)
            await update.effective_message.reply_text("❌ Lỗi tải hình!")
            return

        # Check for feedback mode first
        if chat_id in self.pending_feedback:
            await self.give_steve_jobs_feedback(update, context, temp_file_png)
            return

        # Check comparison flow state
        if chat_id not in self.pending_compare:
            # No active flow - ask user to start with /check or /feedback
            await update.effective_message.reply_text(
                "🤔 Ê ê, gửi lệnh trước đi rồi hẵng quăng hình!\n\n"
                "📋 /check → So sánh DEV vs DESIGN\n"
                "🎯 /feedback → Steve Jobs feedback\n"
                "🤖 Bot by @kieumanhhuy"
            )
            return

        state = self.pending_compare[chat_id]['state']

        if state == 'waiting_dev':
            # Received DEV image
            self.pending_compare[chat_id]['dev_image'] = temp_file_png
            self.pending_compare[chat_id]['state'] = 'waiting_design'

            await update.effective_message.reply_text(
                "✅ **OK nhận hình DEV rồi!**\n\n"
                "📤 Giờ quăng hình **DESIGN** (hình chuẩn) vô để tao soi nha!\n\n"
                "💡 /reset nếu đổi ý",
                parse_mode='Markdown'
            )
            logging.info(f'Received DEV image for chat {chat_id}')
            return

        elif state == 'waiting_design':
            # Received DESIGN image - time to compare!
            dev_image = self.pending_compare[chat_id]['dev_image']

            # Clear state
            del self.pending_compare[chat_id]

            # Save images for re-check
            dev_image.seek(0)
            temp_file_png.seek(0)
            self.last_compare[chat_id] = {
                'dev': io.BytesIO(dev_image.read()),
                'design': io.BytesIO(temp_file_png.read())
            }

            await update.effective_message.reply_text("🔍 Chờ tí nha, đang soi từng pixel như soi da mụn vậy đó! 👀✨")

            logging.info(f'Comparing DEV vs DESIGN for chat {chat_id}')

            # Step 1: Use Pixelmatch for accurate pixel comparison
            dev_image.seek(0)
            temp_file_png.seek(0)

            from image_diff import create_pixelmatch_diff
            pixelmatch_diff, diff_count, grouped_regions, shift_analysis = create_pixelmatch_diff(
                dev_image, temp_file_png, threshold=0.1
            )

            logging.info(f"Pixelmatch: {diff_count} different pixels, {len(grouped_regions)} regions")

            # If no significant differences, report 0 bugs immediately
            if diff_count == 0 or not grouped_regions:
                import random
                comments = [
                    "✅ 0 bug! Pixelmatch xác nhận 2 hình giống y chang! 🔥",
                    "✅ Perfect! Không tìm thấy khác biệt nào! 💰",
                    "✅ Pixel-perfect! Dev đỉnh quá! 😍",
                ]
                await update.effective_message.reply_text(random.choice(comments))
                return

            # Step 2: If cascade detected, log it
            if shift_analysis.get('is_cascade'):
                logging.info(f"Cascade effect detected: {shift_analysis}")

            # Build analysis info
            analysis_info = f"📊 Pixelmatch: {diff_count} pixels khác biệt\n"
            analysis_info += f"Phát hiện {len(grouped_regions)} vùng khác biệt.\n"

            if shift_analysis.get('is_cascade'):
                analysis_info += f"\n⚠️ CASCADE EFFECT: Có thể 1 lỗi gốc gây lệch nhiều vùng\n"

            # Step 3: Send to Claude for ROOT CAUSE analysis
            dev_image.seek(0)
            temp_file_png.seek(0)

            try:
                bugs = await self.openai.analyze_images_for_bugs(
                    dev_image,
                    temp_file_png,
                    analysis_info,
                    pixelmatch_diff_bytes=pixelmatch_diff,
                    shift_analysis=shift_analysis,
                    grouped_regions=grouped_regions
                )
                logging.info(f'Claude found {len(bugs)} root cause bugs')

                if bugs:
                    # Step 3: Draw boxes on DEV image based on Claude's coordinates
                    dev_image.seek(0)
                    annotated_image = draw_bugs_on_image(dev_image, bugs)

                    if annotated_image:
                        # Step 4: Send annotated image with bug report
                        bug_report = format_bug_report(bugs)
                        await update.effective_message.reply_photo(
                            photo=annotated_image,
                            caption=bug_report
                        )
                    else:
                        # Fallback to text only
                        await update.effective_message.reply_text(format_bug_report(bugs))
                else:
                    import random
                    comments = [
                        "✅ 0 bug! Dev hôm nay uống thuốc gì ngon vậy? 🔥",
                        "✅ Perfect! Cho dev tăng lương đi sếp ơi! 💰",
                        "✅ Ủa khớp pixel-perfect luôn? Dev đỉnh quá! 😍",
                        "✅ Clean! Hôm nay dev không ngủ gật hen 👏",
                    ]
                    await update.effective_message.reply_text(random.choice(comments))

            except Exception as e:
                logging.error(f"Error in smart comparison: {e}")
                # Fallback to regular Claude analysis
                dev_image.seek(0)
                temp_file_png.seek(0)
                final_prompt = "Hình 1 là DEV. Hình 2 là DESIGN chuẩn. So sánh và báo lỗi."
                await self._process_vision(update, context, chat_id, [dev_image, temp_file_png], final_prompt)

            return

        # Fallback - shouldn't reach here
        async def _execute():
            bot_language = self.config['bot_language']
            try:
                media_file = await context.bot.get_file(image.file_id)
                temp_file = io.BytesIO(await media_file.download_as_bytearray())
            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=(
                        f"{localized_text('media_download_fail', bot_language)[0]}: "
                        f"{str(e)}. {localized_text('media_download_fail', bot_language)[1]}"
                    ),
                    parse_mode=constants.ParseMode.MARKDOWN
                )
                return

            # convert jpg from telegram to png as understood by openai

            temp_file_png = io.BytesIO()

            try:
                original_image = Image.open(temp_file)

                original_image.save(temp_file_png, format='PNG')
                logging.info(f'New vision request received from user {update.message.from_user.name} '
                             f'(id: {update.message.from_user.id})')

            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=localized_text('media_type_fail', bot_language)
                )



            user_id = update.message.from_user.id
            if user_id not in self.usage:
                self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

            if self.config['stream']:

                stream_response = self.openai.interpret_image_stream(chat_id=chat_id, fileobj=temp_file_png, prompt=prompt)
                i = 0
                prev = ''
                sent_message = None
                backoff = 0
                stream_chunk = 0

                async for content, tokens in stream_response:
                    if is_direct_result(content):
                        return await handle_direct_result(self.config, update, content)

                    if len(content.strip()) == 0:
                        continue

                    stream_chunks = split_into_chunks(content)
                    if len(stream_chunks) > 1:
                        content = stream_chunks[-1]
                        if stream_chunk != len(stream_chunks) - 1:
                            stream_chunk += 1
                            try:
                                await edit_message_with_retry(context, chat_id, str(sent_message.message_id),
                                                              stream_chunks[-2])
                            except:
                                pass
                            try:
                                sent_message = await update.effective_message.reply_text(
                                    message_thread_id=get_thread_id(update),
                                    text=content if len(content) > 0 else "..."
                                )
                            except:
                                pass
                            continue

                    cutoff = get_stream_cutoff_values(update, content)
                    cutoff += backoff

                    if i == 0:
                        try:
                            if sent_message is not None:
                                await context.bot.delete_message(chat_id=sent_message.chat_id,
                                                                 message_id=sent_message.message_id)
                            sent_message = await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(self.config, update),
                                text=content,
                            )
                        except:
                            continue

                    elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                        prev = content

                        try:
                            use_markdown = tokens != 'not_finished'
                            await edit_message_with_retry(context, chat_id, str(sent_message.message_id),
                                                          text=content, markdown=use_markdown)

                        except RetryAfter as e:
                            backoff += 5
                            await asyncio.sleep(e.retry_after)
                            continue

                        except TimedOut:
                            backoff += 5
                            await asyncio.sleep(0.5)
                            continue

                        except Exception:
                            backoff += 5
                            continue

                        await asyncio.sleep(0.01)

                    i += 1
                    if tokens != 'not_finished':
                        total_tokens = int(tokens)


            else:

                try:
                    interpretation, total_tokens = await self.openai.interpret_image(chat_id, temp_file_png, prompt=prompt)


                    try:
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(self.config, update),
                            text=interpretation,
                            parse_mode=constants.ParseMode.MARKDOWN
                        )
                    except BadRequest:
                        try:
                            await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(self.config, update),
                                text=interpretation
                            )
                        except Exception as e:
                            logging.exception(e)
                            await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(self.config, update),
                                text=f"{localized_text('vision_fail', bot_language)}: {str(e)}",
                                parse_mode=constants.ParseMode.MARKDOWN
                            )
                except Exception as e:
                    logging.exception(e)
                    await update.effective_message.reply_text(
                        message_thread_id=get_thread_id(update),
                        reply_to_message_id=get_reply_to_message_id(self.config, update),
                        text=f"{localized_text('vision_fail', bot_language)}: {str(e)}",
                        parse_mode=constants.ParseMode.MARKDOWN
                    )
            vision_token_price = self.config['vision_token_price']
            self.usage[user_id].add_vision_tokens(total_tokens, vision_token_price)

            allowed_user_ids = self.config['allowed_user_ids'].split(',')
            if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                self.usage["guests"].add_vision_tokens(total_tokens, vision_token_price)

        await wrap_with_indicator(update, context, _execute, constants.ChatAction.TYPING)

    async def _process_vision(self, update: Update, context: ContextTypes.DEFAULT_TYPE, chat_id, images, prompt):
        """
        Process multiple images for comparison.
        """
        bot_language = self.config['bot_language']
        user_id = update.message.from_user.id

        if user_id not in self.usage:
            self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

        async def _execute():
            if self.config['stream']:
                stream_response = self.openai.interpret_image_stream(chat_id=chat_id, fileobj=images, prompt=prompt)
                i = 0
                prev = ''
                sent_message = None
                backoff = 0
                stream_chunk = 0
                total_tokens = 0

                async for content, tokens in stream_response:
                    if is_direct_result(content):
                        return await handle_direct_result(self.config, update, content)

                    if len(content.strip()) == 0:
                        continue

                    stream_chunks = split_into_chunks(content)
                    if len(stream_chunks) > 1:
                        content = stream_chunks[-1]
                        if stream_chunk != len(stream_chunks) - 1:
                            stream_chunk += 1
                            try:
                                await edit_message_with_retry(context, chat_id, str(sent_message.message_id),
                                                              stream_chunks[-2])
                            except:
                                pass
                            try:
                                sent_message = await update.effective_message.reply_text(
                                    message_thread_id=get_thread_id(update),
                                    text=content if len(content) > 0 else "..."
                                )
                            except:
                                pass
                            continue

                    cutoff = get_stream_cutoff_values(update, content)
                    cutoff += backoff

                    if i == 0:
                        try:
                            if sent_message is not None:
                                await context.bot.delete_message(chat_id=sent_message.chat_id,
                                                                 message_id=sent_message.message_id)
                            sent_message = await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(self.config, update),
                                text=content,
                            )
                        except:
                            continue

                    elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                        prev = content
                        try:
                            use_markdown = tokens != 'not_finished'
                            await edit_message_with_retry(context, chat_id, str(sent_message.message_id),
                                                          text=content, markdown=use_markdown)
                        except RetryAfter as e:
                            backoff += 5
                            await asyncio.sleep(e.retry_after)
                            continue
                        except TimedOut:
                            backoff += 5
                            await asyncio.sleep(0.5)
                            continue
                        except Exception:
                            backoff += 5
                            continue

                        await asyncio.sleep(0.01)

                    i += 1
                    if tokens != 'not_finished':
                        total_tokens = int(tokens)

                vision_token_price = self.config['vision_token_price']
                self.usage[user_id].add_vision_tokens(total_tokens, vision_token_price)

        await wrap_with_indicator(update, context, _execute, constants.ChatAction.TYPING)

    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        React to incoming messages and respond accordingly.
        """
        if update.edited_message or not update.message or update.message.via_bot:
            return

        if not await self.check_allowed_and_within_budget(update, context):
            return

        logging.info(
            f'New message received from user {update.message.from_user.name} (id: {update.message.from_user.id})')
        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        prompt = message_text(update.message)
        text = update.message.text.lower()


        if await self.summarize_and_reply(update, context):
            return  # đã xử lý tóm tắt thì không chạy tiếp

        # Check for re-check request
        recheck_keywords = ["check lại", "kiểm tra lại", "soi lại", "check lai", "kiem tra lai", "soi lai", "recheck"]
        if any(keyword in text for keyword in recheck_keywords):
            await self.recheck(update, context)
            return

        if is_group_chat(update):
            bot_username = context.bot.username.lower()
            message_text_lower = prompt.lower()

            is_mentioned = f"@{bot_username}" in message_text_lower
            is_reply_to_bot = (
                update.message.reply_to_message
                and update.message.reply_to_message.from_user.id == context.bot.id
            )

            if not is_mentioned and not is_reply_to_bot:
                return ""  # <- thêm dòng này để không lỗi indent

            coin_keywords = ["giá"]
            is_coin_related = any(keyword in text for keyword in coin_keywords)

            if is_coin_related and (is_mentioned or is_reply):
                coin_aliases = {
                    "eth": "ethereum",
                    "ethereum": "ethereum",
                    "btc": "bitcoin",
                    "bitcoin": "bitcoin",
                    "sol": "solana",
                    "solana": "solana",
                    "bnb": "binancecoin",
                    "binance": "binancecoin",
                    "doge": "dogecoin",
                    "dogecoin": "dogecoin",
                    "vrtx": "vertex-protocol",
                    "vertex": "vertex-protocol",
                    "vertex-protocol": "vertex-protocol"
                }

                coin_id = None
                for word, mapped_id in coin_aliases.items():
                    if word in text:
                        coin_id = mapped_id
                        break

                if not coin_id:
                    for word in text.split():
                        possible = find_coin_id_by_symbol(word)
                        if possible:
                            coin_id = possible
                            break

                if not coin_id:
                    await update.message.reply_text("❌ Không xác định được coin bạn hỏi.")
                    return

                if "giá" in text:
                    reply = get_price_from_coingecko(coin_id)
                else:
                    reply = "❓ Bot chưa hiểu bạn muốn xem gì về coin."

                await update.message.reply_text(reply)
                return

        try:
            total_tokens = 0

            if self.config['stream']:
                await update.effective_message.reply_chat_action(
                    action=constants.ChatAction.TYPING,
                    message_thread_id=get_thread_id(update)
                )

                stream_response = self.openai.get_chat_response_stream(chat_id=chat_id, query=prompt)
                i = 0
                prev = ''
                sent_message = None
                backoff = 0
                stream_chunk = 0

                async for content, tokens in stream_response:
                    if is_direct_result(content):
                        return await handle_direct_result(self.config, update, content)

                    if len(content.strip()) == 0:
                        continue

                    stream_chunks = split_into_chunks(content)
                    if len(stream_chunks) > 1:
                        content = stream_chunks[-1]
                        if stream_chunk != len(stream_chunks) - 1:
                            stream_chunk += 1
                            try:
                                await edit_message_with_retry(context, chat_id, str(sent_message.message_id),
                                                              stream_chunks[-2])
                            except:
                                pass
                            try:
                                sent_message = await update.effective_message.reply_text(
                                    message_thread_id=get_thread_id(update),
                                    text=content if len(content) > 0 else "..."
                                )
                            except:
                                pass
                            continue

                    cutoff = get_stream_cutoff_values(update, content)
                    cutoff += backoff

                    if i == 0:
                        try:
                            if sent_message is not None:
                                await context.bot.delete_message(chat_id=sent_message.chat_id,
                                                                 message_id=sent_message.message_id)
                            sent_message = await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(self.config, update),
                                text=content,
                            )
                        except:
                            continue

                    elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                        prev = content

                        try:
                            use_markdown = tokens != 'not_finished'
                            await edit_message_with_retry(context, chat_id, str(sent_message.message_id),
                                                          text=content, markdown=use_markdown)

                        except RetryAfter as e:
                            backoff += 5
                            await asyncio.sleep(e.retry_after)
                            continue

                        except TimedOut:
                            backoff += 5
                            await asyncio.sleep(0.5)
                            continue

                        except Exception:
                            backoff += 5
                            continue

                        await asyncio.sleep(0.01)

                    i += 1
                    if tokens != 'not_finished':
                        total_tokens = int(tokens)

            else:
                async def _reply():
                    nonlocal total_tokens
                    response, total_tokens = await self.openai.get_chat_response(chat_id=chat_id, query=prompt)

                    if is_direct_result(response):
                        return await handle_direct_result(self.config, update, response)

                    # Split into chunks of 4096 characters (Telegram's message limit)
                    chunks = split_into_chunks(response)

                    for index, chunk in enumerate(chunks):
                        try:
                            await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(self.config,
                                                                            update) if index == 0 else None,
                                text=chunk,
                                parse_mode=constants.ParseMode.MARKDOWN
                            )
                        except Exception:
                            try:
                                await update.effective_message.reply_text(
                                    message_thread_id=get_thread_id(update),
                                    reply_to_message_id=get_reply_to_message_id(self.config,
                                                                                update) if index == 0 else None,
                                    text=chunk
                                )
                            except Exception as exception:
                                raise exception

                await wrap_with_indicator(update, context, _reply, constants.ChatAction.TYPING)

            add_chat_request_to_usage_tracker(self.usage, self.config, user_id, total_tokens)

        except Exception as e:
            logging.exception(e)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=f"{localized_text('chat_fail', self.config['bot_language'])} {str(e)}",
                parse_mode=constants.ParseMode.MARKDOWN
            )

    async def inline_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the inline query. This is run when you type: @botusername <query>
        """
        query = update.inline_query.query
        if len(query) < 3:
            return
        if not await self.check_allowed_and_within_budget(update, context, is_inline=True):
            return

        callback_data_suffix = "gpt:"
        result_id = str(uuid4())
        self.inline_queries_cache[result_id] = query
        callback_data = f'{callback_data_suffix}{result_id}'

        await self.send_inline_query_result(update, result_id, message_content=query, callback_data=callback_data)

    async def send_inline_query_result(self, update: Update, result_id, message_content, callback_data=""):
        """
        Send inline query result
        """
        try:
            reply_markup = None
            bot_language = self.config['bot_language']
            if callback_data:
                reply_markup = InlineKeyboardMarkup([[
                    InlineKeyboardButton(text=f'🤖 {localized_text("answer_with_chatgpt", bot_language)}',
                                         callback_data=callback_data)
                ]])

            inline_query_result = InlineQueryResultArticle(
                id=result_id,
                title=localized_text("ask_chatgpt", bot_language),
                input_message_content=InputTextMessageContent(message_content),
                description=message_content,
                thumbnail_url='https://user-images.githubusercontent.com/11541888/223106202-7576ff11-2c8e-408d-94ea-b02a7a32149a.png',
                reply_markup=reply_markup
            )

            await update.inline_query.answer([inline_query_result], cache_time=0)
        except Exception as e:
            logging.error(f'An error occurred while generating the result card for inline query {e}')

    async def handle_callback_inline_query(self, update: Update, context: CallbackContext):
        """
        Handle the callback query from the inline query result
        """
        callback_data = update.callback_query.data
        user_id = update.callback_query.from_user.id
        inline_message_id = update.callback_query.inline_message_id
        name = update.callback_query.from_user.name
        callback_data_suffix = "gpt:"
        query = ""
        bot_language = self.config['bot_language']
        answer_tr = localized_text("answer", bot_language)
        loading_tr = localized_text("loading", bot_language)

        try:
            if callback_data.startswith(callback_data_suffix):
                unique_id = callback_data.split(':')[1]
                total_tokens = 0

                # Retrieve the prompt from the cache
                query = self.inline_queries_cache.get(unique_id)
                if query:
                    self.inline_queries_cache.pop(unique_id)
                else:
                    error_message = (
                        f'{localized_text("error", bot_language)}. '
                        f'{localized_text("try_again", bot_language)}'
                    )
                    await edit_message_with_retry(context, chat_id=None, message_id=inline_message_id,
                                                  text=f'{query}\n\n_{answer_tr}:_\n{error_message}',
                                                  is_inline=True)
                    return

                unavailable_message = localized_text("function_unavailable_in_inline_mode", bot_language)
                if self.config['stream']:
                    stream_response = self.openai.get_chat_response_stream(chat_id=user_id, query=query)
                    i = 0
                    prev = ''
                    backoff = 0
                    async for content, tokens in stream_response:
                        if is_direct_result(content):
                            cleanup_intermediate_files(content)
                            await edit_message_with_retry(context, chat_id=None,
                                                          message_id=inline_message_id,
                                                          text=f'{query}\n\n_{answer_tr}:_\n{unavailable_message}',
                                                          is_inline=True)
                            return

                        if len(content.strip()) == 0:
                            continue

                        cutoff = get_stream_cutoff_values(update, content)
                        cutoff += backoff

                        if i == 0:
                            try:
                                await edit_message_with_retry(context, chat_id=None,
                                                              message_id=inline_message_id,
                                                              text=f'{query}\n\n{answer_tr}:\n{content}',
                                                              is_inline=True)
                            except:
                                continue

                        elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                            prev = content
                            try:
                                use_markdown = tokens != 'not_finished'
                                divider = '_' if use_markdown else ''
                                text = f'{query}\n\n{divider}{answer_tr}:{divider}\n{content}'

                                # We only want to send the first 4096 characters. No chunking allowed in inline mode.
                                text = text[:4096]

                                await edit_message_with_retry(context, chat_id=None, message_id=inline_message_id,
                                                              text=text, markdown=use_markdown, is_inline=True)

                            except RetryAfter as e:
                                backoff += 5
                                await asyncio.sleep(e.retry_after)
                                continue
                            except TimedOut:
                                backoff += 5
                                await asyncio.sleep(0.5)
                                continue
                            except Exception:
                                backoff += 5
                                continue

                            await asyncio.sleep(0.01)

                        i += 1
                        if tokens != 'not_finished':
                            total_tokens = int(tokens)

                else:
                    async def _send_inline_query_response():
                        nonlocal total_tokens
                        # Edit the current message to indicate that the answer is being processed
                        await context.bot.edit_message_text(inline_message_id=inline_message_id,
                                                            text=f'{query}\n\n_{answer_tr}:_\n{loading_tr}',
                                                            parse_mode=constants.ParseMode.MARKDOWN)

                        logging.info(f'Generating response for inline query by {name}')
                        response, total_tokens = await self.openai.get_chat_response(chat_id=user_id, query=query)

                        if is_direct_result(response):
                            cleanup_intermediate_files(response)
                            await edit_message_with_retry(context, chat_id=None,
                                                          message_id=inline_message_id,
                                                          text=f'{query}\n\n_{answer_tr}:_\n{unavailable_message}',
                                                          is_inline=True)
                            return

                        text_content = f'{query}\n\n_{answer_tr}:_\n{response}'

                        # We only want to send the first 4096 characters. No chunking allowed in inline mode.
                        text_content = text_content[:4096]

                        # Edit the original message with the generated content
                        await edit_message_with_retry(context, chat_id=None, message_id=inline_message_id,
                                                      text=text_content, is_inline=True)

                    await wrap_with_indicator(update, context, _send_inline_query_response,
                                              constants.ChatAction.TYPING, is_inline=True)

                add_chat_request_to_usage_tracker(self.usage, self.config, user_id, total_tokens)

        except Exception as e:
            logging.error(f'Failed to respond to an inline query via button callback: {e}')
            logging.exception(e)
            localized_answer = localized_text('chat_fail', self.config['bot_language'])
            await edit_message_with_retry(context, chat_id=None, message_id=inline_message_id,
                                          text=f"{query}\n\n_{answer_tr}:_\n{localized_answer} {str(e)}",
                                          is_inline=True)

    async def check_allowed_and_within_budget(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                                              is_inline=False) -> bool:
        """
        Checks if the user is allowed to use the bot and if they are within their budget
        :param update: Telegram update object
        :param context: Telegram context object
        :param is_inline: Boolean flag for inline queries
        :return: Boolean indicating if the user is allowed to use the bot
        """
        name = update.inline_query.from_user.name if is_inline else update.message.from_user.name
        user_id = update.inline_query.from_user.id if is_inline else update.message.from_user.id

        if not await is_allowed(self.config, update, context, is_inline=is_inline):
            logging.warning(f'User {name} (id: {user_id}) is not allowed to use the bot')
            await self.send_disallowed_message(update, context, is_inline)
            return False
        if not is_within_budget(self.config, self.usage, update, is_inline=is_inline):
            logging.warning(f'User {name} (id: {user_id}) reached their usage limit')
            await self.send_budget_reached_message(update, context, is_inline)
            return False

        return True

    async def send_disallowed_message(self, update: Update, _: ContextTypes.DEFAULT_TYPE, is_inline=False):
        """
        Sends the disallowed message to the user.
        """
        if not is_inline:
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=self.disallowed_message,
                disable_web_page_preview=True
            )
        else:
            result_id = str(uuid4())
            await self.send_inline_query_result(update, result_id, message_content=self.disallowed_message)

    async def send_budget_reached_message(self, update: Update, _: ContextTypes.DEFAULT_TYPE, is_inline=False):
        """
        Sends the budget reached message to the user.
        """
        if not is_inline:
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=self.budget_limit_message
            )
        else:
            result_id = str(uuid4())
            await self.send_inline_query_result(update, result_id, message_content=self.budget_limit_message)

    async def post_init(self, application: Application) -> None:
        """
        Post initialization hook for the bot.
        """
        await application.bot.set_my_commands(self.group_commands, scope=BotCommandScopeAllGroupChats())
        await application.bot.set_my_commands(self.commands)

    def run(self):
        """
        Runs the bot indefinitely until the user presses Ctrl+C
        """
        application = ApplicationBuilder() \
            .token(self.config['token']) \
            .proxy_url(self.config['proxy']) \
            .get_updates_proxy_url(self.config['proxy']) \
            .post_init(self.post_init) \
            .concurrent_updates(True) \
            .build()

        application.add_handler(CommandHandler('reset', self.reset))
        application.add_handler(CommandHandler('check', self.check))
        application.add_handler(CommandHandler('feedback', self.feedback))
        #application.add_handler(CommandHandler('help', self.help))
        #application.add_handler(CommandHandler('image', self.image))
        #application.add_handler(CommandHandler('tts', self.tts))
        application.add_handler(CommandHandler('start', self.help))
        #application.add_handler(CommandHandler('stats', self.stats))
        #application.add_handler(CommandHandler('resend', self.resend))
        application.add_handler(CommandHandler(
            'chat', self.prompt, filters=filters.ChatType.GROUP | filters.ChatType.SUPERGROUP)
        )
        application.add_handler(MessageHandler(
            filters.PHOTO | filters.Document.IMAGE,
            self.vision))
        application.add_handler(MessageHandler(
            filters.AUDIO | filters.VOICE | filters.Document.AUDIO |
            filters.VIDEO | filters.VIDEO_NOTE | filters.Document.VIDEO,
            self.transcribe))
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.prompt))
        application.add_handler(InlineQueryHandler(self.inline_query, chat_types=[
            constants.ChatType.GROUP, constants.ChatType.SUPERGROUP, constants.ChatType.PRIVATE
        ]))
        application.add_handler(CallbackQueryHandler(self.handle_callback_inline_query))

        application.add_error_handler(error_handler)

        application.run_polling()

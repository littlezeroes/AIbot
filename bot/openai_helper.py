from __future__ import annotations
import datetime
import logging
import os
import base64

import anthropic
import httpx

import json
import io
from PIL import Image

from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from utils import is_direct_result, encode_image, decode_image
from plugin_manager import PluginManager

# Claude models
CLAUDE_MODELS = (
    "claude-sonnet-4-20250514",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    "claude-3-opus-latest",
)

# Keep OpenAI models for reference (used for image gen, TTS, whisper)
GPT_4_VISION_MODELS = ("gpt-4o",)


def default_max_tokens(model: str) -> int:
    """
    Gets the default number of max tokens for the given model.
    """
    if "opus" in model:
        return 4096
    elif "sonnet" in model:
        return 4096
    elif "haiku" in model:
        return 4096
    return 4096


def are_functions_available(model: str) -> bool:
    """
    Whether the given model supports functions/tools
    Claude supports tools but we'll keep it simple for now
    """
    return False


# Load translations
parent_dir_path = os.path.join(os.path.dirname(__file__), os.pardir)
translations_file_path = os.path.join(parent_dir_path, 'translations.json')
with open(translations_file_path, 'r', encoding='utf-8') as f:
    translations = json.load(f)


def localized_text(key, bot_language):
    """
    Return translated text for a key in specified bot_language.
    """
    try:
        return translations[bot_language][key]
    except KeyError:
        logging.warning(f"No translation available for bot_language code '{bot_language}' and key '{key}'")
        if key in translations['en']:
            return translations['en'][key]
        else:
            logging.warning(f"No english definition found for key '{key}' in translations.json")
            return key


class OpenAIHelper:
    """
    Claude API helper class (renamed from OpenAI for compatibility).
    Uses Claude for chat/vision, keeps OpenAI for image gen/TTS/transcription.
    """

    def __init__(self, config: dict, plugin_manager: PluginManager):
        """
        Initializes the helper class with the given configuration.
        """
        # Claude client for chat and vision
        self.claude_client = anthropic.AsyncAnthropic(api_key=config['anthropic_api_key'])

        self.config = config
        self.plugin_manager = plugin_manager
        self.conversations: dict[int: list] = {}  # {chat_id: history}
        self.conversations_vision: dict[int: bool] = {}  # {chat_id: is_vision}
        self.last_updated: dict[int: datetime] = {}  # {chat_id: last_update_timestamp}

        # System prompt for QC
        self.system_prompt = """Bạn là QC Bot bựa bựa, hay chọc dev, vibe hài hước. Tên là "Soi Bug Bot" của @kieumanhhuy.

🤖 THÔNG TIN BOT:
- Hỏi "bot của ai?" / "ai tạo bot?" → Trả lời: "Bot của anh @kieumanhhuy đẹp trai tạo ra nha! 😎"
- Hỏi "bot làm gì?" → "Tao soi bug UI cho dev, gửi /check rồi gửi 2 hình DEV vs DESIGN là tao soi liền!"
- Chat xàm xàm → "Ê ê, muốn biết gì thì hỏi ông chủ @kieumanhhuy đi nha! Tao chỉ biết soi bug thôi 🙈"

SO SÁNH UI: Hình 1 = DEV, Hình 2 = DESIGN chuẩn.
CHỈ CHECK: SPACING, ALIGNMENT, COLOR, COMPONENT
QUY TẮC: Chỉ báo lỗi GỐC, không báo hậu quả.

FORMAT MỖI BUG:
🔴 [Vị trí]: [Lỗi gì] | Design: [X] | Dev: [Y]

📊 Tổng: X lỗi

CUỐI CÙNG thêm 1 câu bựa random kiểu:
- Nhiều bug (>3): "Dev ơi về học lại code đi 😭", "Mắt dev để ở nhà hả?", "Designer khóc thét rồi đó", "Đuổi việc hết cho rồi 🔥", "Làm lại đi con, nhìn muốn đột quỵ 💀"
- Ít bug (1-3): "Gần ngon rồi, cố lên dev ơi!", "Tạm chấp nhận được 😏", "Còn vài lỗi nhỏ xíu thôi!"
- 0 bug: "Ủa ngon vậy? Dev hôm nay uống thuốc gì? 🔥", "Perfect luôn, cho dev tăng lương đi sếp ơi! 💰", "Đỉnh của chóp! 🏆"
"""

        # Special prompt for ROOT CAUSE analysis (used with pixelmatch diff)
        self.qc_json_prompt = """Bạn là Senior QC KHẮT KHE nhất, chuyên soi UI pixel-perfect. So sánh 2 hình: DEV (hình 1) vs DESIGN (hình 2).

⚠️ QUAN TRỌNG: KHÔNG ĐƯỢC BỎ SÓT BẤT KỲ LỖI NÀO! Soi kỹ từng pixel!

🔍 CHECKLIST BẮT BUỘC KIỂM TRA:

1️⃣ SPACING - Khoảng cách (SOI KỸ!):
- Padding trên/dưới/trái/phải của MỖI element
- Margin giữa các element
- Gap trong flex/grid
- Khoảng cách giữa text và icon
- Khoảng cách giữa các dòng text
→ LỆCH 1 PIXEL = BÁO LỖI!

2️⃣ ALIGNMENT - Căn chỉnh (SOI KỸ!):
- Text có thẳng hàng với nhau không?
- Icon có căn giữa đúng không?
- Element có align đúng theo design không?
- Vertical alignment của mỗi element
- Horizontal alignment của mỗi element
→ LỆCH 1 PIXEL = BÁO LỖI!

3️⃣ COLOR - Màu sắc:
- Background color chính xác?
- Text color chính xác?
- Border color chính xác?
- Shadow color chính xác?
- Opacity chính xác?

4️⃣ TYPOGRAPHY:
- Font size đúng chưa?
- Font weight đúng chưa?
- Line height đúng chưa?
- Letter spacing đúng chưa?

5️⃣ SIZE - Kích thước:
- Width của element
- Height của element
- Border radius
- Border width

6️⃣ MISSING/EXTRA:
- Có element nào THIẾU không?
- Có element nào THỪA không?
- Có text nào khác không?

📌 QUY TẮC NGHIÊM NGẶT:
- LỆCH 1 PIXEL CŨNG PHẢI BÁO!
- KHÔNG ĐƯỢC nói "gần đúng" hay "chấp nhận được"
- PHẢI báo TẤT CẢ lỗi tìm được
- Nếu 1 lỗi gốc gây nhiều vùng lệch → báo lỗi GỐC + note ảnh hưởng

TRẢ VỀ JSON (KHÔNG được trả [] nếu có bất kỳ khác biệt nào):
```json
[
  {
    "bug": "Mô tả lỗi CỤ THỂ: element gì, lệch bao nhiêu px, hướng nào",
    "type": "SPACING|ALIGNMENT|COLOR|TYPOGRAPHY|SIZE|MISSING",
    "x": 0.0-1.0,
    "y": 0.0-1.0,
    "w": 0.0-1.0,
    "h": 0.0-1.0
  }
]
```

CHỈ TRẢ JSON. KHÔNG có lỗi → [] (nhưng phải CHẮC CHẮN 100% giống nhau)
"""

    def get_conversation_stats(self, chat_id: int) -> tuple[int, int]:
        """
        Gets the number of messages and tokens used in the conversation.
        """
        if chat_id not in self.conversations:
            self.reset_chat_history(chat_id)
        return len(self.conversations[chat_id]), self.__count_tokens(self.conversations[chat_id])

    async def get_chat_response(self, chat_id: int, query: str) -> tuple[str, str]:
        """
        Gets a full response from the Claude model.
        """
        response = await self.__common_get_chat_response(chat_id, query)

        answer = response.content[0].text.strip()
        self.__add_to_history(chat_id, role="assistant", content=answer)

        bot_language = self.config['bot_language']
        tokens_used = response.usage.input_tokens + response.usage.output_tokens

        if self.config['show_usage']:
            answer += "\n\n---\n" \
                      f"💰 {tokens_used} {localized_text('stats_tokens', bot_language)}" \
                      f" ({response.usage.input_tokens} {localized_text('prompt', bot_language)}," \
                      f" {response.usage.output_tokens} {localized_text('completion', bot_language)})"

        return answer, tokens_used

    async def get_chat_response_stream(self, chat_id: int, query: str):
        """
        Stream response from the Claude model.
        """
        response = await self.__common_get_chat_response(chat_id, query, stream=True)

        answer = ''
        async with response as stream:
            async for text in stream.text_stream:
                answer += text
                yield answer, 'not_finished'

        answer = answer.strip()
        self.__add_to_history(chat_id, role="assistant", content=answer)
        tokens_used = str(self.__count_tokens(self.conversations[chat_id]))

        if self.config['show_usage']:
            answer += f"\n\n---\n💰 {tokens_used} {localized_text('stats_tokens', self.config['bot_language'])}"

        yield answer, tokens_used

    @retry(
        reraise=True,
        retry=retry_if_exception_type(anthropic.RateLimitError),
        wait=wait_fixed(20),
        stop=stop_after_attempt(3)
    )
    async def __common_get_chat_response(self, chat_id: int, query: str, stream=False):
        """
        Request a response from the Claude model.
        """
        bot_language = self.config['bot_language']
        try:
            if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                self.reset_chat_history(chat_id)

            self.last_updated[chat_id] = datetime.datetime.now()
            self.__add_to_history(chat_id, role="user", content=query)

            # Summarize if too long
            token_count = self.__count_tokens(self.conversations[chat_id])
            exceeded_max_tokens = token_count + self.config['max_tokens'] > self.__max_model_tokens()
            exceeded_max_history_size = len(self.conversations[chat_id]) > self.config['max_history_size']

            if exceeded_max_tokens or exceeded_max_history_size:
                logging.info(f'Chat history for chat ID {chat_id} is too long. Summarising...')
                try:
                    summary = await self.__summarise(self.conversations[chat_id][:-1])
                    logging.debug(f'Summary: {summary}')
                    self.reset_chat_history(chat_id)
                    self.__add_to_history(chat_id, role="assistant", content=summary)
                    self.__add_to_history(chat_id, role="user", content=query)
                except Exception as e:
                    logging.warning(f'Error while summarising chat history: {str(e)}. Popping elements instead...')
                    self.conversations[chat_id] = self.conversations[chat_id][-self.config['max_history_size']:]

            # Prepare messages for Claude (no system messages in array)
            messages = [msg for msg in self.conversations[chat_id] if msg['role'] != 'system']

            if stream:
                return self.claude_client.messages.stream(
                    model=self.config['model'],
                    max_tokens=self.config['max_tokens'],
                    system=self.system_prompt,
                    messages=messages,
                    temperature=self.config['temperature'],
                )
            else:
                return await self.claude_client.messages.create(
                    model=self.config['model'],
                    max_tokens=self.config['max_tokens'],
                    system=self.system_prompt,
                    messages=messages,
                    temperature=self.config['temperature'],
                )

        except anthropic.RateLimitError as e:
            raise e
        except anthropic.BadRequestError as e:
            raise Exception(f"⚠️ _{localized_text('openai_invalid', bot_language)}._ ⚠️\n{str(e)}") from e
        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    async def generate_image(self, prompt: str) -> tuple[str, str]:
        """
        Image generation disabled - OpenAI removed.
        """
        raise Exception("Image generation is disabled (OpenAI removed)")

    async def generate_speech(self, text: str) -> tuple[any, int]:
        """
        TTS disabled - OpenAI removed.
        """
        raise Exception("TTS is disabled (OpenAI removed)")

    async def transcribe(self, filename):
        """
        Transcription disabled - OpenAI removed.
        """
        raise Exception("Transcription is disabled (OpenAI removed)")

    @retry(
        reraise=True,
        retry=retry_if_exception_type(anthropic.RateLimitError),
        wait=wait_fixed(20),
        stop=stop_after_attempt(3)
    )
    async def __common_get_chat_response_vision(self, chat_id: int, content: list, stream=False):
        """
        Request a response from Claude with vision.
        """
        bot_language = self.config['bot_language']
        try:
            if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                self.reset_chat_history(chat_id)

            self.last_updated[chat_id] = datetime.datetime.now()

            if self.config['enable_vision_follow_up_questions']:
                self.conversations_vision[chat_id] = True
                self.__add_to_history(chat_id, role="user", content=content)
            else:
                for message in content:
                    if message['type'] == 'text':
                        query = message['text']
                        break
                self.__add_to_history(chat_id, role="user", content=query)

            # Prepare messages - get last user message with images
            messages = []
            for msg in self.conversations[chat_id]:
                if msg['role'] == 'system':
                    continue
                messages.append(msg)

            # Replace last message with full content including images
            if messages and messages[-1]['role'] == 'user':
                messages[-1] = {'role': 'user', 'content': content}

            if stream:
                return self.claude_client.messages.stream(
                    model=self.config['vision_model'],
                    max_tokens=self.config['vision_max_tokens'],
                    system=self.system_prompt,
                    messages=messages,
                    temperature=self.config['temperature'],
                )
            else:
                return await self.claude_client.messages.create(
                    model=self.config['vision_model'],
                    max_tokens=self.config['vision_max_tokens'],
                    system=self.system_prompt,
                    messages=messages,
                    temperature=self.config['temperature'],
                )

        except anthropic.RateLimitError as e:
            raise e
        except anthropic.BadRequestError as e:
            raise Exception(f"⚠️ _{localized_text('openai_invalid', bot_language)}._ ⚠️\n{str(e)}") from e
        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    async def interpret_image(self, chat_id, fileobj, prompt=None):
        """
        Interprets a given image file using Claude Vision.
        """
        prompt = self.config['vision_prompt'] if prompt is None else prompt

        # Encode image for Claude
        fileobj.seek(0)
        image_data = base64.b64encode(fileobj.read()).decode('utf-8')

        content = [
            {'type': 'text', 'text': prompt},
            {
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': 'image/png',
                    'data': image_data
                }
            }
        ]

        response = await self.__common_get_chat_response_vision(chat_id, content)

        answer = response.content[0].text.strip()
        self.__add_to_history(chat_id, role="assistant", content=answer)

        bot_language = self.config['bot_language']
        tokens_used = response.usage.input_tokens + response.usage.output_tokens

        if self.config['show_usage']:
            answer += "\n\n---\n" \
                      f"💰 {tokens_used} {localized_text('stats_tokens', bot_language)}" \
                      f" ({response.usage.input_tokens} {localized_text('prompt', bot_language)}," \
                      f" {response.usage.output_tokens} {localized_text('completion', bot_language)})"

        return answer, tokens_used

    async def interpret_image_stream(self, chat_id, fileobj, prompt=None):
        """
        Interprets image file(s) using Claude Vision with streaming.
        fileobj can be a single file or a list of files for comparison.
        """
        prompt = self.config['vision_prompt'] if prompt is None else prompt

        content = [{'type': 'text', 'text': prompt}]

        # Handle multiple images (for comparison)
        if isinstance(fileobj, list):
            for f in fileobj:
                f.seek(0)
                image_data = base64.b64encode(f.read()).decode('utf-8')
                content.append({
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': 'image/png',
                        'data': image_data
                    }
                })
        else:
            fileobj.seek(0)
            image_data = base64.b64encode(fileobj.read()).decode('utf-8')
            content.append({
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': 'image/png',
                    'data': image_data
                }
            })

        response = await self.__common_get_chat_response_vision(chat_id, content, stream=True)

        answer = ''
        async with response as stream:
            async for text in stream.text_stream:
                answer += text
                yield answer, 'not_finished'

        answer = answer.strip()
        self.__add_to_history(chat_id, role="assistant", content=answer)
        tokens_used = str(self.__count_tokens(self.conversations[chat_id]))

        if self.config['show_usage']:
            answer += f"\n\n---\n💰 {tokens_used} {localized_text('stats_tokens', self.config['bot_language'])}"

        yield answer, tokens_used

    async def analyze_images_for_bugs(self, image1_bytes, image2_bytes, analysis_info="") -> list:
        """
        Analyze images and return structured bug data.
        Uses Claude Vision to compare DEV vs DESIGN images.

        Args:
            image1_bytes: DEV image
            image2_bytes: DESIGN image
            analysis_info: Additional analysis info text (SSIM score, etc.)

        Returns:
            List of bugs with x, y, w, h coordinates (0.0-1.0 scale)
        """
        import json as json_module

        # Encode images
        image1_bytes.seek(0)
        image2_bytes.seek(0)
        image1_data = base64.b64encode(image1_bytes.read()).decode('utf-8')
        image2_data = base64.b64encode(image2_bytes.read()).decode('utf-8')

        # Build prompt with analysis info
        prompt = self.qc_json_prompt

        if analysis_info:
            prompt += f"\n\n📊 THÔNG TIN PHÂN TÍCH:\n{analysis_info}"

        # Add image explanations
        prompt += "\n\n🖼️ CÁC HÌNH GỬI KÈM:\n"
        prompt += "- HÌNH 1 = DEV (cần check)\n"
        prompt += "- HÌNH 2 = DESIGN (chuẩn)\n"

        content = [
            {'type': 'text', 'text': prompt},
            {
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': 'image/png',
                    'data': image1_data
                }
            },
            {
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': 'image/png',
                    'data': image2_data
                }
            }
        ]

        try:
            response = await self.claude_client.messages.create(
                model=self.config['vision_model'],
                max_tokens=2000,
                messages=[{'role': 'user', 'content': content}],
                temperature=0,  # 0 for consistent results
            )

            result_text = response.content[0].text.strip()

            # Extract JSON from response
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0]
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0]

            bugs = json_module.loads(result_text)
            return bugs if isinstance(bugs, list) else []

        except Exception as e:
            logging.error(f"Error analyzing images: {e}")
            return []

    def reset_chat_history(self, chat_id, content=''):
        """
        Resets the conversation history.
        """
        self.conversations[chat_id] = []
        self.conversations_vision[chat_id] = False

    def __max_age_reached(self, chat_id) -> bool:
        """
        Checks if the maximum conversation age has been reached.
        """
        if chat_id not in self.last_updated:
            return False
        last_updated = self.last_updated[chat_id]
        now = datetime.datetime.now()
        max_age_minutes = self.config['max_conversation_age_minutes']
        return last_updated < now - datetime.timedelta(minutes=max_age_minutes)

    def __add_to_history(self, chat_id, role, content):
        """
        Adds a message to the conversation history.
        """
        self.conversations[chat_id].append({"role": role, "content": content})

    async def __summarise(self, conversation) -> str:
        """
        Summarises the conversation history.
        """
        messages = [
            {"role": "user", "content": f"Summarize this conversation in 700 characters or less:\n\n{str(conversation)}"}
        ]
        response = await self.claude_client.messages.create(
            model=self.config['model'],
            max_tokens=1000,
            messages=messages,
            temperature=0.4
        )
        return response.content[0].text

    def __max_model_tokens(self):
        """
        Returns the maximum token limit for the current model.
        """
        model = self.config['model']
        if "opus" in model:
            return 200000
        elif "sonnet" in model:
            return 200000
        elif "haiku" in model:
            return 200000
        return 200000  # Claude models have 200k context

    def __count_tokens(self, messages) -> int:
        """
        Estimates the number of tokens in messages.
        Claude uses a similar tokenization to GPT models.
        """
        num_tokens = 0
        for message in messages:
            content = message.get('content', '')
            if isinstance(content, str):
                # Rough estimate: ~4 chars per token
                num_tokens += len(content) // 4
            elif isinstance(content, list):
                for item in content:
                    if item.get('type') == 'text':
                        num_tokens += len(item.get('text', '')) // 4
                    elif item.get('type') == 'image':
                        # Images cost roughly 1000-2000 tokens depending on size
                        num_tokens += 1500
        return num_tokens

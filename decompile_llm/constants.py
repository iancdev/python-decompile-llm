GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

DEFAULT_LLM_MODEL_OPENAI = "gpt-4.1"
DEFAULT_LLM_MODEL_GOOGLE = "gemini-2.5-flash-preview-04-17"
DEFAULT_SYSTEM_MESSAGE = (
    "You are a Python decompiler. Given the following Python bytecode "
    "disassembly, please provide the corresponding Python source code. "
    "Output only the raw Python code. Do not include any explanations, "
    "comments about the process, or markdown code block delimiters."
)
LLM_REQUEST_TIMEOUT_SECONDS = 180
THREAD_COMPLETION_TIMEOUT_SECONDS = LLM_REQUEST_TIMEOUT_SECONDS + 60
DEFAULT_MAX_CHARS = 50000
DEFAULT_MAX_TOKENS = 10000
DEFAULT_MAX_WORKERS_FOR_ITERATIONS = 10

PROVIDER_OPENAI = "openai"
PROVIDER_GOOGLE = "google"
PROVIDER_ALIASES = {
    "chatgpt": PROVIDER_OPENAI,
    "gpt": PROVIDER_OPENAI,
    "gemini": PROVIDER_GOOGLE,
    "google": PROVIDER_GOOGLE,
    "openai": PROVIDER_OPENAI,
}

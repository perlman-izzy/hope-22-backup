#!/usr/bin/env python3
# =============================================================================
#               Autonomous Debugger - Deduplicated & Corrected
# =============================================================================

# --- Core Imports ---
import abc
import asyncio
import ast
import datetime
import difflib
import hashlib
import importlib
import inspect
import io
import json
import logging
import os
import random
import re
import signal
import subprocess
import sys
import textwrap
import time
import traceback
import weakref
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

# --- Optional / third-party imports ----------------------------------------- #
try:
    import google.generativeai as genai
    import google.api_core.exceptions as gexc
    from google.generativeai import GenerativeModel, configure
    import google.auth.exceptions
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    class DummyGoogleException(Exception): ...
    gexc = type(
        "gexc", (), {
            "PermissionDenied": DummyGoogleException,
            "InvalidArgument": DummyGoogleException,
            "ResourceExhausted": DummyGoogleException,
            "ServiceUnavailable": DummyGoogleException,
            "DeadlineExceeded": DummyGoogleException,
        }
    )
    class GenerativeModel: ...
    def configure(**kw): ...
    class google:
        class auth:
            class exceptions:
                class DefaultCredentialsError(Exception): pass

try:
    from aiolimiter import AsyncLimiter
    AIOLIMITER_AVAILABLE = True
except ImportError:
    AIOLIMITER_AVAILABLE = False
    class AsyncLimiter:                 # Dummy (single definition)
        def __init__(self, *a, **k): ...
        async def __aenter__(self): ...
        async def __aexit__(self, *a): ...

try: import astunparse; ASTUNPARSE_AVAILABLE = True
except ImportError: ASTUNPARSE_AVAILABLE = False
try: import tiktoken; TIKTOKEN_AVAILABLE = True
except ImportError: TIKTOKEN_AVAILABLE = False
try: import black; BLACK_AVAILABLE = True
except ImportError: BLACK_AVAILABLE = False
try: import psutil; PSUTIL_AVAILABLE = True
except ImportError: PSUTIL_AVAILABLE = False
try: import nest_asyncio; NEST_ASYNCIO_AVAILABLE = True
except ImportError: NEST_ASYNCIO_AVAILABLE = False
try: import requests; REQUESTS_AVAILABLE = True
except ImportError: REQUESTS_AVAILABLE = False

if NEST_ASYNCIO_AVAILABLE:
    try: nest_asyncio.apply()
    except RuntimeError: pass

# --- Logging Setup ---------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("autonomous_debugger")
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("blib2to3").setLevel(logging.WARNING)
if GOOGLE_GENAI_AVAILABLE:
    logging.getLogger("google.api_core").setLevel(logging.WARNING)
    logging.getLogger("google.generativeai").setLevel(logging.INFO)

# --- Global constants ------------------------------------------------------- #
MAX_TOKEN_LIMIT                 = 1_000_000
CODE_START_DELIMITER            = "<<<PYTHON_CODE_START>>>"
CODE_END_DELIMITER              = "<<<PYTHON_CODE_END>>>"
DIFF_START_DELIMITER            = "<<<DIFF_START>>>"
DIFF_END_DELIMITER              = "<<<DIFF_END>>>"
WRAPPER_EXCEPTION_START_MARKER  = "--- WRAPPER CAUGHT INNER EXCEPTION START ---"
WRAPPER_EXCEPTION_END_MARKER    = "--- WRAPPER CAUGHT INNER EXCEPTION END ---"
WRAPPER_SYSTEM_EXIT_MARKER      = "--- WRAPPER CAUGHT SYSTEMEXIT"
USER_CODE_EXCEPTION_START_MARKER= "--- USER CODE EXCEPTION START ---"
USER_CODE_EXCEPTION_END_MARKER  = "--- USER CODE EXCEPTION END ---"
TARGETED_CODE_START_DELIMITER   = "<<<TARGETED_CODE_START>>>"
TARGETED_CODE_END_DELIMITER     = "<<<TARGETED_CODE_END>>>"
FIX_INFO_START_DELIMITER        = "<<<FIX_INFO_START>>>"
FIX_INFO_END_DELIMITER          = "<<<FIX_INFO_END>>>"
TIMEOUT_SIGNAL                  = getattr(signal, "SIGUSR1", signal.SIGTERM)
SIGNAL_TRACEBACK_START_MARKER   = "--- FAULTHANDLER TRACEBACK DUMP START ---"
SIGNAL_TRACEBACK_END_MARKER     = "--- FAULTHANDLER TRACEBACK DUMP END ---"



# =============================================================================
#                                CONFIG
# =============================================================================
class InitializationError(Exception): ...

# =============================================================================
#                                CONFIG
# =============================================================================
class InitializationError(Exception): ...

# Replace this with your actual API keys or placeholder
# --- Load your real API keys for ApiKeyManager rotation --------------------
# ----------------------------------------------------------------------------
# 1) LOAD YOUR PROXY'S KEYS INTO THIS SCRIPT
# ----------------------------------------------------------------------------
KEYS_FILE = "/Users/williamwhite/myapikeys/old/apikeys"
print(f"[INFO] Loading API keys from {KEYS_FILE}")
if not os.path.isfile(KEYS_FILE):
    raise RuntimeError(f"Keys file not found: {KEYS_FILE}")

_API_KEYS: List[str] = []
with open(KEYS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            _API_KEYS.append(line)

if not _API_KEYS:
    raise RuntimeError(f"No API keys found in {KEYS_FILE}")

print(f"[INFO] Successfully loaded {len(_API_KEYS)} API keys from {KEYS_FILE}")

# Expose them under the name the rest of the code expects:
RAW_KEYS = list(_API_KEYS)
# ----------------------------------------------------------------------------

# --------------------------------------------------------------------------- #


@dataclass
class Config:
    api_key: str = ""  # Will be set from RAW_KEYS later if needed
    target_file_path: str = "/Users/williamwhite/testcodebase5.py"
    log_level: str = "INFO"
    log_file: str = "unified_output_log.json"
    output_filename_base: str = "debugged_output"
    
    # CORRECTED: Use a confirmed, state-of-the-art Gemini 2.5 model.
    llm_model_name: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")
    
    llm_max_retries: int = 10
    llm_base_delay_seconds: float = 2.0
    llm_temperature: float = 0.5
    llm_default_max_output_tokens: int = 8192
    llm_api_call_timeout_seconds: float = 230.0
    llm_requests_per_minute: int = 2
    llm_max_retries_transient: int = 3
    code_execution_timeout_seconds: int = 120
    final_execution_timeout_seconds: int = 300
    max_attempts_per_subgoal: int = 50
    syntax_error_repetition_threshold: int = 2
    stagnation_threshold: int = 10
    api_key_initial_cooldown_seconds: float = 65.0
    api_key_max_cooldown_seconds: float = 300.0
    api_key_transient_failure_threshold: int = 3
    enable_resource_monitoring: bool = PSUTIL_AVAILABLE
    enable_code_style_check: bool = BLACK_AVAILABLE
    verify_api_key_on_startup: bool = True
    save_failed_llm_responses: bool = False
    enable_flask_proxy: bool = True
    flask_proxy_url: str = "http://localhost:8000"
    user_objective: str = ""
    _initialized: bool = field(default=False, repr=False)

    def __post_init__(self):
        if not self.api_key and RAW_KEYS:
            self.api_key = RAW_KEYS[0]
            
        if not self.user_objective and self.target_file_path:
            self.user_objective = (
                f"Debug the Python script '{os.path.basename(self.target_file_path)}' "
                "so it runs with exit code 0 and no stderr."
            )
        self._initialized = True


        
class ApiKeyManager:
    """
    Manages a pool of API keys, handling rotation, cooldowns, and state tracking.
    """
    # Outcome constants
    SUCCESS = "success"
    THROTTLED = "THROTTLED"          # 429 error
    PERMANENT_FAILURE = "permanent_failure"
    TRANSIENT_FAILURE = "transient_failure"

    def __init__(self, keys: List[str], config: Config):
        self.logger = logging.getLogger("autonomous_debugger.ApiKeyManager")
        if not keys:
            raise InitializationError("API key list cannot be empty for ApiKeyManager.")

        self.config = config
        self.ordered_keys = list(keys)            # keep original order for rotation
        random.shuffle(self.ordered_keys)         # start rotation randomly
        self.current_key_index = 0

        self.key_status: Dict[str, Dict[str, Any]] = {
            key: {
                "state": "AVAILABLE",
                "cooldown_until": 0.0,
                "consecutive_429": 0,
                "consecutive_transient": 0,
            }
            for key in keys
        }

        self.initial_cooldown_seconds = getattr(config, "api_key_initial_cooldown_seconds", 65.0)
        self.max_cooldown_seconds = getattr(config, "api_key_max_cooldown_seconds", 300.0)
        self.transient_failure_threshold = getattr(config, "api_key_transient_failure_threshold", 3)

        self.logger.debug(
            f"ApiKeyManager initialized with {len(keys)} keys. "
            f"Initial cooldown: {self.initial_cooldown_seconds}s"
        )

    async def get_key(self) -> Optional[str]:
        """
        Retrieves the next available API key, handling cooldowns and skipping disabled keys.
        Waits if all available keys are currently cooling down.
        Returns None if all keys are permanently disabled.
        """
        start_index = self.current_key_index
        checked_all_once = False
        now = time.monotonic()

        while True:
            if checked_all_once and self.current_key_index == start_index:
                self.logger.debug(
                    "Completed a full key rotation cycle without an immediately available key."
                )
                break

            key_to_check = self.ordered_keys[self.current_key_index]
            status = self.key_status[key_to_check]
            next_index = (self.current_key_index + 1) % len(self.ordered_keys)

            if status["state"] == "AVAILABLE":
                self.current_key_index = next_index
                return key_to_check

            if status["state"] == "THROTTLED" and now >= status["cooldown_until"]:
                self.logger.info(f"Key ...{key_to_check[-5:]} cooldown finished. Marking AVAILABLE.")
                status.update(state="AVAILABLE", cooldown_until=0.0, consecutive_429=0)
                self.current_key_index = next_index
                return key_to_check

            self.current_key_index = next_index
            if self.current_key_index == start_index:
                checked_all_once = True

        # If we reach here, we wait for the nearest cooldown
        active = [k for k, s in self.key_status.items() if s["state"] != "DISABLED"]
        if not active:
            self.logger.error("All API keys are permanently disabled.")
            return None

        min_cd = min(
            (self.key_status[k]["cooldown_until"] for k in active if self.key_status[k]["state"] == "THROTTLED"),
            default=now + 5,
        )
        await asyncio.sleep(max(0.1, min_cd - time.monotonic()))
        return await self.get_key()

    def report_outcome(self, key: str, outcome: str):
        """Updates the status of a key based on the outcome of an API call attempt."""
        status = self.key_status.get(key)
        if not status:
            self.logger.error(f"Unknown key reported: ...{key[-5:]}")
            return

        now = time.monotonic()
        if outcome == self.SUCCESS:
            status.update(state="AVAILABLE", cooldown_until=0.0, consecutive_429=0, consecutive_transient=0)
        elif outcome == self.THROTTLED:
            status["state"] = "THROTTLED"
            status["consecutive_429"] += 1
            backoff = 2 ** max(0, status["consecutive_429"] - 1)
            cd = min(self.max_cooldown_seconds, self.initial_cooldown_seconds * backoff)
            status["cooldown_until"] = now + cd
        elif outcome == self.PERMANENT_FAILURE:
            status.update(state="DISABLED", cooldown_until=float("inf"))
        elif outcome == self.TRANSIENT_FAILURE:
            status["consecutive_transient"] += 1
            if status["consecutive_transient"] >= self.transient_failure_threshold:
                status.update(state="DISABLED", cooldown_until=float("inf"))


# MINIMALLY MODIFIED with required imports and Flask support
import asyncio
import json
import requests
import logging
from google.generativeai import configure, GenerativeModel
import google.api_core.exceptions as gexc
from google.generativeai.types import GenerationConfig, BlockedPromptException
from aiolimiter import AsyncLimiter

import asyncio
import json
import logging
import requests
from aiolimiter import AsyncLimiter
from google.generativeai import configure, GenerativeModel
import google.api_core.exceptions as gexc
from google.generativeai.types import BlockedPromptException

class LLMClient:
    """
    Handles communication with the Google Generative AI API, specifically targeting
    the Gemini 2.5 models.
    PRIMARY: Uses a Flask proxy for key management.
    FALLBACK: Uses local ApiKeyManager for direct calls if the proxy fails.
    """
    def __init__(self, model_name: str, api_key_manager: "ApiKeyManager", config: Config):
        self.model_name = model_name
        self.km = api_key_manager
        self.config = config
        self.logger = logging.getLogger("autonomous_debugger.LLMClient")

        # This is the reliable maximum output for Gemini 1.5/2.0/2.5 models.
        # Reduced for Flask proxy compatibility
        self.MAX_OUTPUT_TOKENS = 4096

        self.transient_retries = config.llm_max_retries_transient
        self.base_delay = config.llm_base_delay_seconds / 2
        self.default_temperature = config.llm_temperature
        self.api_call_timeout = config.llm_api_call_timeout_seconds

        self.use_flask_proxy = (
            bool(getattr(config, "flask_proxy_url", None)) and 
            REQUESTS_AVAILABLE and 
            getattr(config, "enable_flask_proxy", True)
        )
        self.flask_proxy_url = getattr(config, "flask_proxy_url", "http://localhost:8000").rstrip("/")

        if self.use_flask_proxy:
            self.logger.info(f"LLMClient will use Flask proxy at {self.flask_proxy_url} for model {self.model_name}.")
            self.session = requests.Session()
        else:
            self.logger.info(f"LLMClient will use direct API calls for model {self.model_name}.")
            if not REQUESTS_AVAILABLE:
                self.logger.warning("`requests` library not found. Flask proxy is disabled.")

        self.request_limiter = AsyncLimiter(config.llm_requests_per_minute, 60)

    def _create_generation_config(self, max_tokens: Optional[int]) -> GenerationConfig:
        """Creates the generation configuration, ensuring max_output_tokens is correctly set."""
        effective_max_tokens = min(max_tokens, self.MAX_OUTPUT_TOKENS) if max_tokens is not None else self.MAX_OUTPUT_TOKENS
        return genai.types.GenerationConfig(
            max_output_tokens=effective_max_tokens,
            temperature=self.default_temperature
        )

    async def _process_direct_response(self, response: genai.types.GenerateContentResponse) -> str:
        """Processes a direct response from the `google-generativeai` library."""
        if not response.candidates:
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                raise BlockedPromptException(f"Prompt blocked with reason: {response.prompt_feedback.block_reason.name}")
            raise ValueError("LLM response is empty and not blocked.")

        candidate = response.candidates[0]
        if candidate.finish_reason == 'MAX_TOKENS':
            self.logger.warning("Model response was truncated due to reaching the maximum output token limit.")
        
        if candidate.content and candidate.content.parts:
            return "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
        
        if candidate.finish_reason == 'SAFETY':
             raise BlockedPromptException(f"Response blocked due to safety filters. Ratings: {candidate.safety_ratings}")
             
        raise ValueError("Unable to extract text from LLM response candidate.")

    async def _process_flask_response(self, response_text: str) -> str:
        """Processes a JSON response from the Flask proxy."""
        try:
            data = json.loads(response_text)
            if data.get("candidates"):
                candidate = data["candidates"][0]
                if candidate.get("finishReason") == "MAX_TOKENS":
                    self.logger.warning("Model response (via proxy) was truncated due to reaching the maximum output token limit.")
                    # Try to extract partial content if available
                    if candidate.get("content", {}).get("parts"):
                        partial_content = "".join(part.get("text", "") for part in candidate["content"]["parts"])
                        if partial_content.strip():
                            self.logger.info("Extracted partial content from truncated proxy response")
                            return partial_content
                    raise ValueError("Response truncated and no extractable content available")
                    
                if candidate.get("content", {}).get("parts"):
                    return "".join(part.get("text", "") for part in candidate["content"]["parts"])
            
            if data.get("promptFeedback", {}).get("blockReason"):
                raise BlockedPromptException(f"Proxy response blocked with reason: {data['promptFeedback']['blockReason']}")
            
            if "error" in data:
                raise RuntimeError(f"Flask proxy returned an error: {data['error']}")
                
            raise ValueError("Could not extract valid content from proxy response.")

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON from Flask proxy response: {response_text[:500]}")
            # If JSON is malformed due to truncation, try to extract any readable content
            if "text" in response_text and len(response_text) > 100:
                self.logger.warning("Attempting to extract content from malformed JSON response")
                # Simple regex to find text content in malformed JSON
                import re
                text_matches = re.findall(r'"text":\s*"([^"]*)"', response_text)
                if text_matches:
                    combined_text = "".join(text_matches)
                    if combined_text.strip():
                        self.logger.info("Extracted partial text from malformed JSON")
                        return combined_text
            raise ValueError("Invalid JSON response from proxy.")

    async def _attempt_flask_proxy_call(self, prompt: str, max_tokens: Optional[int]) -> str:
        """Sends a request to the Flask proxy and handles its response."""
        effective_max_tokens = min(max_tokens, self.MAX_OUTPUT_TOKENS) if max_tokens is not None else self.MAX_OUTPUT_TOKENS
        endpoint = f"{self.flask_proxy_url}/v1beta/models/{self.model_name}:generateContent"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": effective_max_tokens,
                "temperature": self.default_temperature
            }
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        
        self.logger.debug(f"Attempting API call via proxy to {endpoint} with max_tokens={effective_max_tokens}")
        self.logger.debug(f"Payload: {json.dumps(payload, indent=2)[:500]}...")

        resp = await asyncio.to_thread(
            self.session.post,
            endpoint,
            json=payload,
            headers=headers,
            timeout=self.api_call_timeout
        )
        resp.raise_for_status()
        
        self.logger.debug(f"Flask proxy response status: {resp.status_code}")
        self.logger.debug(f"Flask proxy response length: {len(resp.text)}")
        self.logger.debug(f"Flask proxy response preview: {resp.text[:200]}...")
        
        return await self._process_flask_response(resp.text)

    async def call_llm(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Main method to call the LLM. It first tries the Flask proxy and falls back
        to direct API calls with key rotation on any failure.
        """
        if self.use_flask_proxy:
            try:
                return await self._attempt_flask_proxy_call(prompt, max_tokens)
            except Exception as e:
                self.logger.warning(f"Flask proxy call failed: {e}. Falling back to direct API calls.")

        self.logger.info("Using fallback: Direct API call with key rotation.")
        max_attempts = len(self.km.ordered_keys) * (self.transient_retries + 1)
        
        for attempt in range(max_attempts):
            key = await self.km.get_key()
            if not key:
                self.logger.error("Fallback failed: No usable API keys in ApiKeyManager.")
                return "Error: No usable API keys for fallback."

            try:
                async with self.request_limiter:
                    genai.configure(api_key=key)
                    model = genai.GenerativeModel(self.model_name)
                    generation_config = self._create_generation_config(max_tokens)
                    
                    safety_settings = [
                        {"category": c, "threshold": "BLOCK_NONE"}
                        for c in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                                  "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
                    ]
                    
                    self.logger.debug(f"Attempting direct API call with key ...{key[-4:]} and max_tokens={generation_config.max_output_tokens}")
                    
                    response = await asyncio.to_thread(
                        model.generate_content,
                        prompt,
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                    
                    text = await self._process_direct_response(response)
                    self.km.report_outcome(key, self.km.SUCCESS)
                    self.logger.info(f"Fallback API call successful with key ...{key[-4:]}")
                    return text

            except gexc.ResourceExhausted as e:
                self.logger.warning(f"Key ...{key[-4:]} is rate-limited (429). Cooling down. (Attempt {attempt+1}/{max_attempts})")
                self.km.report_outcome(key, self.km.THROTTLED)
                await asyncio.sleep(self.base_delay * (1.5 ** attempt))
            except (BlockedPromptException, gexc.PermissionDenied, gexc.InvalidArgument) as e:
                self.logger.error(f"Key ...{key[-4:]} failed permanently: {e}. Disabling. (Attempt {attempt+1}/{max_attempts})")
                self.km.report_outcome(key, self.km.PERMANENT_FAILURE)
            except Exception as e:
                self.logger.warning(f"Key ...{key[-4:]} encountered transient error: {e}. Retrying. (Attempt {attempt+1}/{max_attempts})", exc_info=True)
                self.km.report_outcome(key, self.km.TRANSIENT_FAILURE)
                await asyncio.sleep(self.base_delay * (1.5 ** attempt))

        self.logger.critical("All fallback attempts failed after exhausting all keys.")
        return "Error: All keys failed after multiple direct attempts."


# COMPLETELY REVERTED to original
def initialize_components(config: Config) -> Dict[str, Any]:
    """
    Creates **every** component the debugger needs and wires them together.
    Adjusted initialization logging to DEBUG for most components.
    """
    logger_init = logging.getLogger("autonomous_debugger.init")
    print("[Setup] Initializing components…")
    components: Dict[str, Any] = {}
    try:
        # --- 1. API Key Management ---
        if not RAW_KEYS: raise InitializationError("RAW_KEYS list is empty or contains only placeholders.")
        
        # Create a properly initialized API key manager with the required parameters
        api_key_manager = ApiKeyManager(keys=RAW_KEYS, config=config)
        components["api_key_manager"] = api_key_manager
        logger_init.debug("ApiKeyManager ✓") # Changed to DEBUG

        # --- 2. Core State / Logging ---
        components["error_memory"]   = ErrorMemory()
        components["debug_session"]  = DebugSession(config=config)
        components["unified_logger"] = UnifiedLogger(log_file=config.log_file)
        logger_init.debug("ErrorMemory, DebugSession, UnifiedLogger ✓") # Changed to DEBUG

        # --- 3. LLM Client ---
        components["llm_client"] = LLMClient(
            model_name=config.llm_model_name,
            api_key_manager=components["api_key_manager"],
            config=config
        )
        # Attempt to link DebugSession (optional)
        if hasattr(components["llm_client"], 'link_debug_session'):
             components["llm_client"].link_debug_session(components["debug_session"])
        logger_init.debug("LLMClient ✓") # Changed to DEBUG

        # --- 4. Code / Execution Utilities ---
        components["codebase_manager"] = CodebaseManager(components["error_memory"])
        components["code_executor"] = CodeExecutor(
            error_memory=components["error_memory"],
            config=config
        )
        components["reality_checker"] = RealityChecker()
        logger_init.debug("CodebaseManager, CodeExecutor, RealityChecker ✓") # Changed to DEBUG

        # --- 5. Coding Agent ---
        components["coding_agent"] = CodingAgent(
            llm_client=components["llm_client"],
            codebase_manager=components["codebase_manager"],
            error_memory=components["error_memory"],
            reality_checker=components["reality_checker"],
            debug_session=components["debug_session"],
            config=config
        )
        logger_init.debug("CodingAgent ✓") # Changed to DEBUG

        # --- 6. Inspection / Validation ---
        inspections: List[BaseInspection] = [FilesystemInspection()]
        if config.enable_code_style_check and BLACK_AVAILABLE:
            inspections.append(CodeStyleInspection())
        components["inspection_agent"] = InspectionAgent(
            agent_id="inspector-001",
            inspections=inspections
        )
        validator = SystemValidator()
        validator.configure(
            inspection_agent=components["inspection_agent"],
            error_memory=components["error_memory"],
            llm_client=components["llm_client"]
        )
        validator.link_debug_session(components["debug_session"])
        components["system_validator"] = validator
        logger_init.debug("InspectionAgent, SystemValidator ✓") # Changed to DEBUG

        # --- 7. Subgoal & Process Management ---
        components["subgoal_agent"] = SubgoalAgent(
            llm_client=components["llm_client"],
            error_memory=components["error_memory"],
            debug_session=components["debug_session"]
        )
        components["debugging_loop"] = DebugLoop( # Placeholder
            code_executor=components["code_executor"],
            error_memory=components["error_memory"],
            codebase_manager=components["codebase_manager"],
            unified_logger=components["unified_logger"],
            debug_session=components["debug_session"]
        )
        components["process_manager"] = ProcessManager(
            coding_agent=components["coding_agent"],
            code_executor=components["code_executor"],
            error_memory=components["error_memory"],
            unified_logger=components["unified_logger"],
            system_validator=components["system_validator"],
            reality_checker=components["reality_checker"],
            codebase_manager=components["codebase_manager"],
            debug_session=components["debug_session"],
            config=config
        )
        logger_init.debug("SubgoalAgent, DebugLoop, ProcessManager ✓") # Changed to DEBUG

        # --- 8. Main Executor + Runtime Metrics ---
        components["main_executor"] = MainExecutor(config=config, components=components)
        components["runtime_counter"] = RuntimeCounter()
        if config.enable_resource_monitoring and PSUTIL_AVAILABLE:
            components["runtime_metrics"] = RuntimeMetrics()
        logger_init.debug("MainExecutor, RuntimeCounter, RuntimeMetrics ✓") # Changed to DEBUG

        # Keep final success message at INFO
        logger_init.info("All components initialised successfully.")
        print("[Setup] All components initialised successfully.")
        return components

    except InitializationError as e_init:
        logger_init.critical(f"Component initialisation failed: {e_init}", exc_info=True)
        print(f"\n[Setup Error] {e_init}")
        raise
    except Exception as e:
        logger_init.critical("Unexpected component-initialisation failure: %s", e, exc_info=True)
        print(f"\n[Setup Error] Unexpected initialisation failure: {e}")
        raise InitializationError(f"Unexpected setup error: {e}") from e
#                   (chunk continues: CodeExtractor, RealityChecker,          #
#                    ErrorMemory, CodeExecutor, …)                            #
# ============================================================================
import re
from typing import List, Pattern, Tuple

class ModuleLoader:
    """
    Load & exec a Python file *after* running it through a configurable
    list of regex-based patches.  Perfect for catching stray typos like
    `from typing: X` → `from typing import X`, or any other mass find/replace.
    """

    def __init__(self, path: str):
        self.path = path
        # list of (compiled_pattern, replacement_template)
        self._patches: List[Tuple[Pattern[str], str]] = []

        # register your default fixes here
        self.add_patch(
            # catch e.g. "from typing: List, Dict, Optional"
            r'^(?P<pre>\s*from\s+\w+)\s*:\s*(?P<rest>.+)$',
            r'\g<pre> import \g<rest>'
        )

    def add_patch(self, pattern: str, replacement: str):
        """
        Register a new regex patch.  `pattern` is applied multiline
        against the source; `replacement` can use backrefs.
        """
        self._patches.append((re.compile(pattern, re.MULTILINE), replacement))

    def load_module(self) -> dict:
        """
        Read, patch, compile & exec the source.  Returns the exec namespace.
        """
        src = self._read_and_patch_source()
        namespace: dict = {}
        exec(compile(src, self.path, 'exec'), namespace)
        return namespace

    def _read_and_patch_source(self) -> str:
        """
        Read the raw file, run *all* registered patches in sequence,
        then return the patched source.
        """
        with open(self.path, 'r', encoding='utf-8') as f:
            src = f.read()

        for pat, repl in self._patches:
            src = pat.sub(repl, src)

        return src
# 
# 
# 

# --- Code Extractor (Class #3 - Corrected Indentation/Formatting) -----------
class CodeExtractor:
    PYTHON_KEYWORDS = (
        "import ",
        "from ",
        "def ",
        "class ",
        "@",
        "#",
        '"""',
        "'''",
    )
    _logger = logging.getLogger("CodeExtractor")

    @staticmethod
    def _prepare_and_validate(code_block: str) -> Optional[str]:
        if not code_block:
            CodeExtractor._logger.debug("Val skip: empty block.")
            return None
        try:
            dedented_code = textwrap.dedent(code_block)
            cleaned_code = dedented_code.strip()
        except Exception as e:
            CodeExtractor._logger.error(f"Dedent/strip err: {e}", exc_info=True)
            return None

        if not cleaned_code or cleaned_code.isspace():
            CodeExtractor._logger.debug("Val fail: empty/ws clean.")
            return None

        try:
            tree = ast.parse(cleaned_code)
            if not tree.body:
                CodeExtractor._logger.debug("Val trivial: Empty AST.")
                return None
            is_trivial = all(
                isinstance(n, (ast.Pass, ast.Expr))
                and (
                    isinstance(n, ast.Pass)
                    or isinstance(getattr(n, "value", None), ast.Constant)
                )
                for n in tree.body
            )
            if is_trivial and not (
                len(tree.body) == 1 and isinstance(tree.body[0], ast.Pass)
            ):
                CodeExtractor._logger.debug("Val trivial: Only Pass/Const.")
                return None
            CodeExtractor._logger.debug("Val pass: syntax valid & non-trivial.")
            return cleaned_code
        except SyntaxError as e:
            try:
                lines = cleaned_code.splitlines()
                lc = (
                    lines[e.lineno - 1]
                    if e.lineno and 0 < e.lineno <= len(lines)
                    else "N/A"
                )
            except IndexError:
                lc = "N/A(Idx Err)"
            CodeExtractor._logger.warning(
                f"Val fail WITHIN block: SyntaxError {e.msg} L{e.lineno} Ctx:'{lc}'. Returning block."
            )
            return cleaned_code  # Return block despite internal SyntaxError
        except Exception as e:
            CodeExtractor._logger.error(
                f"Val fail: Unexpected AST err: {e}", exc_info=True
            )
            return None

    @staticmethod
    def extract_code(response: str) -> Optional[str]:
        CodeExtractor._logger.debug("Attempting code extraction...")
        if not response:
            CodeExtractor._logger.warning(
                "Cannot extract code from empty response."
            )
            return None

        bt = chr(96)
        nl = chr(10)
        end_tag_str = f"{bt*3}"
        python_block_start_tag = f"{end_tag_str}python{nl}"
        fallback_block_start_tag = f"{end_tag_str}{nl}"

        # Strategy 1: Specific ```python block
        try:
            start = response.find(python_block_start_tag)
            if start != -1:
                CodeExtractor._logger.debug(
                    f"Found '{python_block_start_tag.strip()}'."
                )
                end = response.find(end_tag_str, start + len(python_block_start_tag))
                if end != -1:
                    code_raw = response[
                        start + len(python_block_start_tag) : end
                    ]
                    processed = CodeExtractor._prepare_and_validate(code_raw)
                    if processed is not None:
                        CodeExtractor._logger.info(
                            f"Processed via '```python' ({len(processed)} chars)."
                        )
                        return processed
        except Exception as e:
            CodeExtractor._logger.error(
                f"Specific python block err: {e}", exc_info=True
            )

        # Strategy 2: Fallback generic ``` block
        try:
            start = response.find(fallback_block_start_tag)
            specific_start = response.find(python_block_start_tag)
            if start != -1 and (start != specific_start or specific_start == -1):
                CodeExtractor._logger.debug(
                    f"Found distinct '{fallback_block_start_tag.strip()}'."
                )
                end = response.find(end_tag_str, start + len(fallback_block_start_tag))
                if end != -1:
                    code_raw = response[
                        start + len(fallback_block_start_tag) : end
                    ]
                    processed = CodeExtractor._prepare_and_validate(code_raw)
                    if processed is not None:
                        CodeExtractor._logger.info(
                            f"Processed via fallback '```' ({len(processed)} chars)."
                        )
                        return processed
        except Exception as e:
            CodeExtractor._logger.error(f"Fallback block err: {e}", exc_info=True)

        # Strategy 3: Regex (Cleaned formatting)
        processed_blocks = []
        try:
            generic_block_pattern = r"```(?:[a-zA-Z0-9_]*)?\s*?\n(.*?)\n?```"
            generic_block_regex = re.compile(
                generic_block_pattern, re.DOTALL | re.IGNORECASE
            )
            CodeExtractor._logger.debug("Compiled regex pattern.")
            for match in generic_block_regex.finditer(response):
                block = match.group(1)
                processed = CodeExtractor._prepare_and_validate(block)
                if processed is not None:
                    CodeExtractor._logger.debug("Regex block processed.")
                    processed_blocks.append(processed)
                    break  # Use first valid match
        except Exception as e:
            CodeExtractor._logger.error(f"Regex err: {e}", exc_info=True)

        if processed_blocks:
            CodeExtractor._logger.info(
                f"Processed via regex (FIRST valid, {len(processed_blocks[0])} chars)."
            )
            return processed_blocks[0]

        CodeExtractor._logger.warning(
            "Failed to extract/process code via all methods."
        )
        return None


# --- Validation Function (Function #4) --------------------------------------
def preemptive_linting(code_str: str) -> "ValidationResult":
    """Validates the syntax of the code string using ast.parse."""
    if not isinstance(code_str, str):
        return ValidationResult(False, "Type error", {"error_type": "TypeError"})
    if not code_str.strip():
        return ValidationResult(False, "Syntax error: empty", {})
    try:
        ast.parse(code_str)
        return ValidationResult(True, "Syntax valid.", None)
    except SyntaxError as e:
        try:
            lines = code_str.splitlines()
            lc = lines[e.lineno - 1] if e.lineno and 0 < e.lineno <= len(lines) else "N/A"
        except Exception:
            lc = getattr(e, "text", "N/A").strip()
        return ValidationResult(
            False,
            f"Syntax error: {e.msg} L{e.lineno}",
            {
                "line": e.lineno,
                "offset": e.offset,
                "text": lc,
                "full": str(e),
            },
        )
    except Exception as e:
        return ValidationResult(
            False,
            f"Unexpected validation err: {e}",
            {"type": type(e).__name__, "full": str(e)},
        )


# --- Helper Visitor (for RealityChecker) ------------------------------------
class _SuperficialFixVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_superficial_fix = False
        self.func_depth = 0

    def visit_FunctionDef(self, node):
        self.func_depth += 1
        self.generic_visit(node)
        self.func_depth -= 1

    def visit_AsyncFunctionDef(self, node):
        self.func_depth += 1
        self.generic_visit(node)
        self.func_depth -= 1

    def visit_Pass(self, node):
        if self.func_depth > 0:
            self.has_superficial_fix = True

# --- Missing AST Helper Classes (from document 3) ---

# Helper Visitor for finding AST node
class ASTLocatorVisitor(ast.NodeVisitor):
    def __init__(self, target_name: str, target_type: str): # target_type 'function' or 'class'
        self.target_name = target_name
        self.target_type = target_type
        self.found_node: Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]] = None # Added AsyncFunctionDef
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.target_type == 'function' and node.name == self.target_name:
            if self.found_node is None:
                self.logger.debug(f"ASTLocatorVisitor found target function: '{node.name}' at line {node.lineno}")
                self.found_node = node
            return
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if self.target_type == 'function' and node.name == self.target_name:
            if self.found_node is None:
                self.logger.debug(f"ASTLocatorVisitor found target async function: '{node.name}' at line {node.lineno}")
                self.found_node = node
            return
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        if self.target_type == 'class' and node.name == self.target_name:
            if self.found_node is None:
                 self.logger.debug(f"ASTLocatorVisitor found target class: '{node.name}' at line {node.lineno}")
                 self.found_node = node
            return
        self.generic_visit(node)

# Helper Transformer for replacing AST node
class ASTReplaceTransformer(ast.NodeTransformer):
    def __init__(self, target_name: str, target_type: str, replacement_node: ast.AST):
        self.target_name = target_name
        self.target_type = target_type
        self.replacement_node = replacement_node
        self.replaced = False
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if not self.replaced and self.target_type == 'function' and node.name == self.target_name:
            self.logger.info(f"ASTReplaceTransformer replacing function '{node.name}' at line {node.lineno}")
            self.replaced = True
            if isinstance(self.replacement_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                 return self.replacement_node
            else:
                 self.logger.error(f"Replacement node type mismatch for function '{node.name}'. Expected FunctionDef/Async, got {type(self.replacement_node)}")
                 return node
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if not self.replaced and self.target_type == 'function' and node.name == self.target_name:
            self.logger.info(f"ASTReplaceTransformer replacing async function '{node.name}' at line {node.lineno}")
            self.replaced = True
            if isinstance(self.replacement_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return self.replacement_node
            else:
                self.logger.error(f"Replacement node type mismatch for async function '{node.name}'. Expected FunctionDef/Async, got {type(self.replacement_node)}")
                return node
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        if not self.replaced and self.target_type == 'class' and node.name == self.target_name:
            self.logger.info(f"ASTReplaceTransformer replacing class '{node.name}' at line {node.lineno}")
            self.replaced = True
            if isinstance(self.replacement_node, ast.ClassDef):
                 return self.replacement_node
            else:
                 self.logger.error(f"Replacement node type mismatch for class '{node.name}'. Expected ClassDef, got {type(self.replacement_node)}")
                 return node
        return self.generic_visit(node)
    
# --- Reality Checker (Class #5) ---------------------------------------------
class RealityChecker:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("RealityChecker initialized (LLM-independent).")

    async def reality_check(
        self, code: str, subgoal: str, context: str = ""
    ) -> Tuple[bool, str]:
        """Performs basic reality checks: syntax and superficial fixes."""
        syntax_result = preemptive_linting(code)
        if not syntax_result.passed:
            detail = (
                f" ({syntax_result.details})" if syntax_result.details else ""
            )
            msg = f"Syntax fail: {syntax_result.message}{detail}"
            self.logger.warning(f"Reality Check Failed: {msg}")
            return False, msg

        if self._detect_superficial_fixes(code):
            msg = "Superficial fix detected (e.g., adding 'pass' in function)."
            self.logger.warning(f"Reality Check Failed: {msg}")
            return False, msg

        self.logger.debug(
            "Reality Check Passed: Code syntax valid, no obvious superficial fixes."
        )
        return True, "Code passes basic reality checks."

    def _detect_superficial_fixes(self, code: str) -> bool:
        """Detects if the code mainly consists of 'pass' statements within functions."""
        try:
            tree = ast.parse(code)
            visitor = _SuperficialFixVisitor()
            visitor.visit(tree)
            if visitor.has_superficial_fix:
                self.logger.warning("Superficial fix detected by AST visitor.")
                return True
        except SyntaxError:
            self.logger.warning("SyntaxError during superficial fix check.")
            return True
        except Exception as e:
            self.logger.error(
                f"Unexpected error during superficial fix detection: {e}",
                exc_info=True,
            )
            return False
        return False

# --- Error Memory (Class #6) -------------------------------------------------
class ErrorMemory:
    def __init__(self):
        self.errors: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("ErrorMemory initialized.")

    def add_error(self, message: str, context: Any = ""):
        ctx_repr = str(context)
        ctx_short = ctx_repr[:200] + ("..." if len(ctx_repr) > 200 else "")
        entry = {
            "message": message,
            "context": context,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self.errors.append(entry)
        self.logger.warning(f"Error added: {message} (Ctx Hint: {ctx_short})")

    def get_errors(self) -> List[Dict[str, Any]]:
        return self.errors[:]

    def clear_errors(self):
        self.errors = []
        self.logger.info("Error memory cleared.")

    def get_last_error(self) -> Optional[Dict[str, Any]]:
        return self.errors[-1].copy() if self.errors else None

    def get_improvement_prompt(self) -> str:
        if not self.errors:
            return "No recent errors recorded."
        summary_parts = []
        for e in self.errors[-5:]:
            message = e.get("message", "?")
            context = e.get("context", {})
            line_info_str = ""
            stderr_snippet = ""
            if isinstance(context, dict):
                if context.get("line") is not None:
                    line_info_str = f"(L{context.get('line')})"
                if context.get("stderr") is not None:
                    stderr_snippet = f" | Stderr: {str(context.get('stderr'))[:100]}..."
                summary_parts.append(
                    f"- {message} {line_info_str}{stderr_snippet}".strip()
                )
        msgs = "\n".join(summary_parts)
        prompt = f"Recent errors:\n{msgs}\n\nPlease analyze these errors and suggest a fix."
        self.logger.info("Generated improvement prompt based on recent errors.")
        return prompt


# --- Runtime Tracking (Class #7 & Conditional Metrics) -----------------------
class RuntimeCounter:
    def __init__(self):
        self.start_time = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("RuntimeCounter init.")

    def start(self):
        self.start_time = time.monotonic()
        self.logger.info("Runtime counter started.")

    def stop(self) -> str:
        rt = time.monotonic() - self.start_time if self.start_time else 0
        fmt = self._format_runtime(rt)
        self.logger.info(f"Total runtime: {fmt}")
        return fmt

    def get_current_runtime(self) -> str:
        return (
            self._format_runtime(time.monotonic() - self.start_time)
            if self.start_time
            else "00:00:00.000"
        )

    def _format_runtime(self, rt: float) -> str:
        h = int(rt // 3600)
        m = int((rt % 3600) // 60)
        s = rt % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"


if PSUTIL_AVAILABLE:

    class RuntimeMetrics:
        def __init__(self):
            self.mem = []
            self.cpu = []
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.debug("RuntimeMetrics init.")

        def track(self):
            try:
                self.mem.append(psutil.virtual_memory().percent)
                self.cpu.append(psutil.cpu_percent(interval=0.1))
                if self.mem and self.cpu:
                    self.logger.debug(
                        f"M:{self.mem[-1]:.1f}% C:{self.cpu[-1]:.1f}%"
                    )
            except Exception as e:
                self.logger.warning(f"Metric track fail: {e}")

        def get_avg(self) -> dict:
            mem_avg = sum(self.mem) / len(self.mem) if self.mem else 0
            cpu_avg = sum(self.cpu) / len(self.cpu) if self.cpu else 0
            return {"avg_mem": mem_avg, "avg_cpu": cpu_avg}

        def reset(self):
            self.mem.clear()
            self.cpu.clear()
            self.logger.info("Runtime metrics reset.")


# --- Debug Session (Class #12) ----------------------------------------------
class DebugSession:
    """Tracks debug session statistics and progress."""

    def __init__(self, config=None):
        self.config_dict = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.llm_api_calls = 0
        self.total_fix_attempts = 0
        self.successful_integrations = 0
        self.llm_fix_attempts = 0
        self.validation_attempts = 0
        self.validation_failures = 0
        self.start_time = time.monotonic()
        self.logger.info("DebugSession initialized.")

    def log_llm_call(self):
        self.llm_api_calls += 1
        self.logger.debug(f"LLM call logged. Total: {self.llm_api_calls}")

    def log_fix_attempt(self, method: str, success: bool, details: str = ""):
        self.total_fix_attempts += 1
        status = "SUCCESS" if success else "FAILURE"
        if method.startswith("LLM Fix"):
            self.llm_fix_attempts += 1
        if success:
            self.successful_integrations += 1
        self.logger.info(
            f"Fix Attempt Log #{self.total_fix_attempts}: "
            f"Method='{method}', Status={status}. {details}"
        )

    def log_validation_attempt(self, success: bool, reason: str = ""):
        self.validation_attempts += 1
        status = "SUCCESS" if success else "FAILURE"
        if not success:
            self.validation_failures += 1
            self.logger.warning(
                f"Validation Attempt #{self.validation_attempts}: {status}. Reason: {reason}"
            )
        else:
            self.logger.info(
                f"Validation Attempt #{self.validation_attempts}: {status}. Reason: {reason}"
            )

    def log_stats(self):
        runtime = time.monotonic() - self.start_time
        fix_success_rate = (
            self.successful_integrations / self.total_fix_attempts * 100
            if self.total_fix_attempts
            else 0
        )
        val_success_rate = (
            100 - (self.validation_failures / self.validation_attempts * 100)
            if self.validation_attempts
            else 100
        )
        stats = (
            f"\n{'='*20} Debug Session Stats {'='*20}\n"
            f"Total Runtime:         {runtime:.2f} seconds\n"
            f"Total Fix Attempts:    {self.total_fix_attempts}\n"
            f"Successful Integrations:{self.successful_integrations}\n"
            f"LLM Targeted Fix Atts: {self.llm_fix_attempts}\n"
            f"Total LLM API Calls:   {self.llm_api_calls}\n"
            f"Integration Success Rate:{fix_success_rate:.1f}%\n"
            f"Validation Attempts:   {self.validation_attempts}\n"
            f"Validation Success Rate:{val_success_rate:.1f}%\n"
            f"{'='*61}"
        )
        self.logger.info(stats)
        print(stats)  # Print stats as well


# --- CodeExecutor (Class #13) -----------------------------------------------
class CodeExecutor:
    """
    Executes Python code in a subprocess with error/output capture.
    Version 4.4 – Direct exec, signal handling.
    """

    def __init__(self, error_memory: "ErrorMemory", config: "Config"):
        self.error_memory = error_memory
        self.config = config
        self.prefix = "temp_code_"
        self.timeout = config.code_execution_timeout_seconds
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(
            "CodeExecutor initialized (v4.4 - Direct Exec, Syntax Fixes)."
        )

    async def execute_code(self, code: str, timeout: Optional[int] = None) -> dict:
        if not isinstance(code, str) or not code.strip():
            self.logger.error(
                "Execution skipped: Invalid/empty code provided."
            )
            return {
                "stdout": "",
                "stderr": "Invalid code provided",
                "returncode": -1,
                "timeout": False,
                "error_type": "ValueError",
            }

        tf_suffix = hashlib.md5(code.encode("utf-8", errors="ignore")).hexdigest()[
            :8
        ]
        tf = f"{self.prefix}{tf_suffix}.py"
        effective_timeout = (
            timeout
            if isinstance(timeout, (int, float)) and timeout > 0
            else self.timeout
        )
        self.logger.info(f"Executing code in temporary file: {tf}")

        process = None
        pid: Optional[int] = None
        stdout_str = ""
        stderr_str = ""
        returncode = -1
        timed_out = False
        error_type = None
        user_traceback = None
        signal_traceback = None

        # --- wrapper construction ------------------------------------------------
        try:
            async_run_pattern = r"asyncio\.run\s*\(([^)]+)\)"
            main_block_match = re.search(async_run_pattern, code)
            code_to_prepare = code
            if main_block_match:
                original_call = main_block_match.group(0)
                original_args = main_block_match.group(1).strip()
                new_args = (
                    f"{original_args}, debug=True"
                    if original_args and not original_args.endswith(",")
                    else f"{original_args}debug=True"
                )
                debug_call = f"asyncio.run({new_args})"
                code_to_prepare = code.replace(original_call, debug_call, 1)

            dedented_code_for_exec = textwrap.dedent(code_to_prepare).strip()

            wrapped_code = f"""# Wrapper v4.0 (Direct Exec, Signal Handling)
import sys, traceback, os, signal, time, asyncio, faulthandler
def _dump(sig, frame):
    print("\\n{SIGNAL_TRACEBACK_START_MARKER}", file=sys.stderr)
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    print("{SIGNAL_TRACEBACK_END_MARKER}", file=sys.stderr)
def _term(sig, frame):
    print(f"Wrapper received signal {{sig}}, exiting.", file=sys.stderr)
    sys.exit(128+sig)
signal.signal(signal.SIGTERM, _term)
signal.signal(signal.SIGINT, _term)
signal.signal({TIMEOUT_SIGNAL}, _dump)
try:
    exec({repr(dedented_code_for_exec)}, {{'__name__':'__main__'}})
except SystemExit as e:
    print('{WRAPPER_SYSTEM_EXIT_MARKER}({{e.code if isinstance(e.code,int) else 1}}) ---', file=sys.stderr)
    raise
except BaseException as e:
    print('{WRAPPER_EXCEPTION_START_MARKER}', file=sys.stderr)
    traceback.print_exc()
    print('{WRAPPER_EXCEPTION_END_MARKER}', file=sys.stderr)
    sys.exit(1)
"""
        except Exception as wrap_error:
            self.logger.error(
                f"Failed during code preparation: {wrap_error}", exc_info=True
            )
            self.error_memory.add_error(
                "Code Prep Error", {"traceback": traceback.format_exc()}
            )
            return {
                "stdout": "",
                "stderr": f"Code Prep Error: {wrap_error}",
                "returncode": -1,
                "timeout": False,
                "error_type": "CodePrepError",
            }

        # --- execute -------------------------------------------------------------
        try:
            with open(tf, "w", encoding="utf-8") as f:
                f.write(wrapped_code)

            cmd = [sys.executable, "-u", tf]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            pid = process.pid

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=effective_timeout
                )
                returncode = process.returncode
                stdout_str = stdout_bytes.decode(errors="replace").strip()
                stderr_str = stderr_bytes.decode(errors="replace").strip()
            except asyncio.TimeoutError:
                timed_out = True
                returncode = -9
                stderr_str = (
                    f"Execution timed out after {effective_timeout} seconds.\n"
                )
                try:
                    os.killpg(os.getpgid(pid), TIMEOUT_SIGNAL)
                    await asyncio.sleep(0.5)
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except Exception:
                    pass

            if returncode != 0 or timed_out:
                self.error_memory.add_error(
                    f"Execution Error (RC={returncode})",
                    {
                        "stdout": stdout_str[:500],
                        "stderr": stderr_str[:500],
                        "returncode": returncode,
                        "timeout": timed_out,
                    },
                )
            else:
                self.logger.info(f"Execution succeeded: {tf}")

            return {
                "stdout": stdout_str,
                "stderr": stderr_str,
                "returncode": returncode,
                "timeout": timed_out,
                "error_type": error_type,
            }

        except FileNotFoundError:
            msg = f"Python executable not found: {sys.executable}"
            self.logger.critical(msg)
            self.error_memory.add_error(msg)
            return {
                "stdout": "",
                "stderr": msg,
                "returncode": -1,
                "timeout": False,
                "error_type": "FileNotFoundError",
            }
        except Exception as e:
            self.logger.error(
                f"Unexpected error during execution: {e}", exc_info=True
            )
            self.error_memory.add_error("Unexpected Execution Error")
            return {
                "stdout": stdout_str,
                "stderr": (
                    stderr_str
                    + f"\n[CodeExecutor Internal Error]: {e}\n{traceback.format_exc()}"
                ).strip(),
                "returncode": -1,
                "timeout": False,
                "error_type": type(e).__name__,
            }
        finally:
            if process and process.returncode is None:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except Exception:
                    pass
            if os.path.exists(tf):
                try:
                    os.remove(tf)
                except Exception:
                    pass

# --- TestExecutor (Class) ----------------------------------------------------
class TestExecutor:
    def __init__(self, error_memory: ErrorMemory):
        self.error_memory = error_memory
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("TestExecutor initialized (basic).")


# --- cleanup_resources (Function) -------------------------------------------
def cleanup_resources():
    files_cleaned = 0
    for fname in os.listdir("."):
        if (
            fname.startswith(("temp_code_", "temp_test_"))
            and fname.endswith(".py")
        ):
            try:
                os.remove(fname)
                files_cleaned += 1
                logger.debug(f"Removed temp file: {fname}")
            except Exception as e:
                logger.error(f"Could not remove {fname}: {e}")
    if files_cleaned:
        logger.info(f"Cleanup removed {files_cleaned} temp files.")
    else:
        logger.info("Cleanup found no temp files.")


# --- DebugLoop (Class) -------------------------------------------------------
class DebugLoop:
    """
    Placeholder loop for running tests; currently inactive.
    """

    def __init__(
        self,
        code_executor: CodeExecutor,
        error_memory: ErrorMemory,
        codebase_manager: "CodebaseManager",
        unified_logger: "UnifiedLogger",
        debug_session: DebugSession,
    ):
        self.code_executor = code_executor
        self.error_memory = error_memory
        self.codebase_manager = codebase_manager
        self.unified_logger = unified_logger
        self.debug_session = debug_session
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("DebugLoop initialized (inactive).")

    async def run_and_evaluate_tests(self, code: str, tests: list) -> list:
        self.logger.warning("run_and_evaluate_tests called but inactive.")
        return []

    async def execute_test(self, test_code: str, main_code: str) -> dict:
        self.logger.warning("execute_test called but inactive.")
        return {"passed": False, "error": "Not implemented"}


# --- SubgoalAgent (Class) ----------------------------------------------------
from typing import Any, Dict, List, Optional
import re
import asyncio
import json
import logging

# Assuming these are imported elsewhere in hope-22:
# from your_module import LLMClient, ErrorMemory, DebugSession, InspectionAgent

logger = logging.getLogger("autonomous_debugger")

import re
import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("autonomous_debugger.SubgoalAgent")

class SubgoalAgent:
    """Generates and refines subgoals for a given objective with feedback and tracking."""

    def __init__(
        self,
        llm_client: LLMClient,
        error_memory: ErrorMemory,
        debug_session: DebugSession,
    ):
        """
        Initializes the SubgoalAgent with required dependencies.

        Args:
            llm_client (LLMClient): Instance to interact with the LLM API.
            error_memory (ErrorMemory): Instance to log and track errors.
            debug_session (DebugSession): Session to log metrics and attempts.
        """
        self.llm_client = llm_client
        self.error_memory = error_memory
        self.debug_session = debug_session
        # Tracks per-subgoal success/total counts
        self.subgoal_success_rate: Dict[str, Dict[str, int]] = {}
        logger.info("SubgoalAgent initialized.")

    async def generate_subgoals(self, objective: str, max_retries: int = 3) -> List[Dict[str, str]]:
        """
        Generates 2–3 major subgoals via the LLM and classifies each as 'coding' or 'non-coding'.

        Args:
            objective: The main goal to break down.
            max_retries: How many times to retry on failure.

        Returns:
            A list of dicts: [{ 'description': str, 'type': 'coding'|'non-coding' }, …]
        """
        base_prompt = (
            f"Break down the following objective into 2-3 major phases without going into specific functions or code steps.\n\n"
            f"Objective: {objective}\n\n"
            f"Provide the response in this format:\n"
            f"1. First identify and analyze potential errors or issues in the given program\n"
            f"2. Implement necessary fixes and validate the solution\n\n"
            f"Keep the format exactly as shown above with a numbered list."
        )
        retries = 0
        while retries < max_retries:
            try:
                self.debug_session.log_llm_call()
                llm_resp = await self.llm_client.call_llm(base_prompt)
                subs = self._parse_subgoals(llm_resp)
                if not subs:
                    raise ValueError("No valid subgoals generated.")
                result: List[Dict[str, str]] = []
                for desc in subs:
                    cat = await self.classify_subgoal(desc) or "coding"
                    result.append({"description": desc, "type": cat})
                logger.info(f"Subgoals generated and classified: {result}")
                return result
            except Exception as e:
                logger.error(f"Error generating/classifying subgoals: {e}")
                self.error_memory.add_error(f"SubgoalAgent.generate_subgoals error: {e}", objective)
                retries += 1
                await asyncio.sleep(2)
        logger.error("Failed to generate and classify subgoals after retries.")
        return []

    def _parse_subgoals(self, response: str) -> List[str]:
        """
        Extracts lines beginning with '1.', '2.', etc., from the LLM response.
        """
        lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
        subs: List[str] = []
        for ln in lines:
            m = re.match(r'^\d+\.\s+(.*)', ln)
            if m:
                subs.append(m.group(1).strip())
        return subs

    async def classify_subgoal(self, subgoal: str) -> Optional[str]:
        """
        Classifies a single subgoal as 'coding' or 'non-coding'.
        """
        prompt = (
            f"Classify the following subgoal into one of the categories: 'coding' or 'non-coding'.\n\n"
            f"Subgoal:\n{subgoal}\n\n"
            f"Provide only the category as 'coding' or 'non-coding'."
        )
        try:
            self.debug_session.log_llm_call()
            resp = await self.llm_client.call_llm(prompt)
            cat = resp.strip().lower()
            if cat in ("coding", "non-coding"):
                return cat
            logger.warning(f"Unexpected classification '{resp}' for subgoal '{subgoal}'")
        except Exception as e:
            logger.error(f"Error in classify_subgoal: {e}")
            self.error_memory.add_error(f"SubgoalAgent.classify_subgoal error: {e}", subgoal)
        return None

    def track_subgoal_success(self, subgoal: str, success: bool):
        """
        Record success/failure counts and log the rolling success rate.
        """
        stats = self.subgoal_success_rate.setdefault(subgoal, {"success": 0, "total": 0})
        stats["total"] += 1
        if success:
            stats["success"] += 1
        rate = stats["success"] / stats["total"] if stats["total"] else 0
        logger.info(f"Subgoal '{subgoal}' success rate: {rate:.2%}")

    def analyze_subgoal_patterns(self) -> Dict[str, Any]:
        """
        Summarizes success rates, flags low-performing subgoals, and suggests revisions.
        """
        analysis: Dict[str, Any] = {
            "total_subgoals": len(self.subgoal_success_rate),
            "success_rates": {},
            "problem_areas": [],
            "recommendations": []
        }
        for sg, stats in self.subgoal_success_rate.items():
            rate = stats["success"] / stats["total"] if stats["total"] else 0
            analysis["success_rates"][sg] = rate
            if rate < 0.5:
                analysis["problem_areas"].append(sg)
                analysis["recommendations"].append(
                    f"Revise '{sg}' (low success rate: {rate:.2%})"
                )
        logger.info(f"Subgoal pattern analysis: {analysis}")
        return analysis


# --- ProcessManager (Class header only) -------------------------------------
class ProcessManager:
    """
    Manages processing of subgoals: execution, targeted fixing via AST, retries,
    stagnation detection.
    """

    def __init__(
        self,
        coding_agent: "CodingAgent",
        code_executor: CodeExecutor,
        error_memory: ErrorMemory,
        unified_logger: "UnifiedLogger",
        system_validator: "SystemValidator",
        reality_checker: "RealityChecker",
        codebase_manager: "CodebaseManager",
        debug_session: DebugSession,
        config: Config,
    ):
        # method body continues in next chunk …
        pass

        # --- constructor body (continued) ---------------------------------
        if not hasattr(coding_agent, "generate_targeted_fix"):
            if not hasattr(coding_agent, "generate_initial_syntax_fix"):
                raise InitializationError(
                    "ProcessManager requires a CodingAgent with "
                    "'generate_targeted_fix' and 'generate_initial_syntax_fix'."
                )
            else:
                raise InitializationError(
                    "ProcessManager requires a CodingAgent with "
                    "'generate_targeted_fix' method."
                )

        self.unparse_available = hasattr(ast, "unparse") or ASTUNPARSE_AVAILABLE
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.unparse_available:
            self.logger.critical(
                "AST unparsing capability (ast.unparse or astunparse) is MISSING."
            )

        self.coding_agent = coding_agent
        self.code_executor = code_executor
        self.error_memory = error_memory
        self.unified_logger = unified_logger
        self.system_validator = system_validator
        self.reality_checker = reality_checker
        self.codebase_manager = codebase_manager
        self.debug_session = debug_session
        self.config = config

        self.MAX_ATTEMPTS_PER_SUBGOAL = config.max_attempts_per_subgoal
        self.SYNTAX_ERROR_REPETITION_THRESHOLD = (
            config.syntax_error_repetition_threshold
        )
        self.STAGNATION_THRESHOLD = config.stagnation_threshold

        self._consecutive_syntax_error_count = 0
        self._last_syntax_error_details: Optional[
            Tuple[str, Optional[int], Optional[str]]
        ] = None
        self._consecutive_stagnation_count = 0
        self._last_stagnant_code_hash: Optional[str] = None
        self.completed_subgoals: List[str] = []

        self.logger.info(
            f"ProcessManager initialized. MaxAttempts={self.MAX_ATTEMPTS_PER_SUBGOAL}, "
            f"SyntaxRepeat={self.SYNTAX_ERROR_REPETITION_THRESHOLD}, "
            f"Stagnation={self.STAGNATION_THRESHOLD}"
        )

    # ------------------------------------------------------------------ #
    # helper utilities
    # ------------------------------------------------------------------ #
    def _get_code_hash(self, code: str) -> str:
        return hashlib.md5(code.encode("utf-8", errors="ignore")).hexdigest()

    def _check_syntax_error_repetition(self, line_proximity: int = 5) -> bool:
        if len(self.error_memory.errors) < 1:
            self._consecutive_syntax_error_count = 0
            self._last_syntax_error_details = None
            return False

        last_err = self.error_memory.errors[-1]
        msg = last_err.get("message", "")
        ctx = last_err.get("context", {})

        is_syntax = False
        line: Optional[int] = None
        err_txt: str = "?"
        code_context: Optional[str] = None

        syntax_msgs = [
            "Syntax error in generated block",
            "LLM fix block invalid syntax",
            "Syntax error:",
            "IndentationError:",
        ]

        if isinstance(ctx, dict):
            error_type = ctx.get("error_type")
            stderr = str(ctx.get("stderr", ""))

            if any(sm in msg for sm in syntax_msgs) or error_type in (
                "SyntaxError",
                "IndentationError",
            ):
                is_syntax = True
                syntax_details = ctx.get("syntax_details", ctx)
                line = syntax_details.get("line")
                err_txt = syntax_details.get("message", msg)
                code_context = syntax_details.get("text")

            elif ctx.get("returncode", 0) != 0 and stderr:
                m = re.search(
                    r'File ".*?(?:\.py|<string>)", line (\d+).*?(SyntaxError|IndentationError): (.+)',
                    stderr,
                    re.DOTALL,
                )
                if m:
                    is_syntax = True
                    line = int(m.group(1))
                    err_txt = f"{m.group(2)}: {m.group(3).strip()}"

        if not is_syntax and any(sm in msg for sm in syntax_msgs):
            is_syntax = True
            err_txt = msg
            m = re.search(r"[Ll]ine (\d+)", msg)
            if m:
                line = int(m.group(1))

        if not is_syntax:
            self._consecutive_syntax_error_count = 0
            self._last_syntax_error_details = None
            return False

        norm_msg = " ".join(err_txt.lower().split())
        current = (norm_msg, line, code_context)
        repeated = False

        if self._last_syntax_error_details:
            last_msg, last_line, last_ctx = self._last_syntax_error_details
            message_match = norm_msg == last_msg
            line_match = (
                line is not None and last_line is not None and line == last_line
            )
            context_match = (
                code_context is not None
                and last_ctx is not None
                and code_context == last_ctx
            )
            close_line = (
                line is not None
                and last_line is not None
                and abs(line - last_line) <= line_proximity
            )
            repeated = message_match and (line_match or context_match or close_line)

        if repeated:
            self._consecutive_syntax_error_count += 1
        else:
            self._consecutive_syntax_error_count = 1

        self._last_syntax_error_details = current
        return self._consecutive_syntax_error_count >= (
            self.SYNTAX_ERROR_REPETITION_THRESHOLD
        )

    # ------------------------------------------------------------------ #
    # AST integration helper
    # ------------------------------------------------------------------ #
    def _integrate_targeted_fix(
        self,
        current_full_code: str,
        fix_block: str,
        fix_name: str,
        fix_type: str,
    ) -> Optional[str]:
        if not self.unparse_available:
            self.error_memory.add_error("AST Integration Failed: unparse missing")
            return None

        try:
            original_ast = ast.parse(current_full_code)
        except SyntaxError as e_orig:
            self.error_memory.add_error(
                "AST Integration Failed: current code invalid",
                {"line": e_orig.lineno, "msg": e_orig.msg},
            )
            return None

        try:
            repl_mod = ast.parse(fix_block)
            if not repl_mod.body or len(repl_mod.body) != 1:
                self.error_memory.add_error(
                    "AST Integration Failed: fix block structure invalid"
                )
                return None
            repl_node = repl_mod.body[0]
            if fix_type == "function" and not isinstance(
                repl_node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                self.error_memory.add_error(
                    "AST Integration Failed: type mismatch (function)"
                )
                return None
            if fix_type == "class" and not isinstance(repl_node, ast.ClassDef):
                self.error_memory.add_error(
                    "AST Integration Failed: type mismatch (class)"
                )
                return None
        except SyntaxError as e_repl:
            self.error_memory.add_error(
                "AST Integration Failed: replacement block invalid",
                {"line": e_repl.lineno, "msg": e_repl.msg},
            )
            return None

        transformer = ASTReplaceTransformer(fix_name, fix_type, repl_node)
        modified_ast = transformer.visit(original_ast)
        if not transformer.replaced:
            self.error_memory.add_error(
                "AST Integration Failed: target not found",
                {"type": fix_type, "name": fix_name},
            )
            return None

        try:
            if hasattr(ast, "unparse"):
                new_code = ast.unparse(modified_ast)
            else:
                new_code = astunparse.unparse(modified_ast)
            if new_code and not new_code.endswith("\n"):
                new_code += "\n"
            return new_code
        except Exception as e_unp:
            self.error_memory.add_error(
                "AST Integration Failed: unparse error", {"err": str(e_unp)}
            )
            return None

    # ------------------------------------------------------------------ #
    # targeted-fix attempt
    # ------------------------------------------------------------------ #
    async def _attempt_debug_fix(
        self, sg: Dict[str, Any], objective: str, code_before: str
    ) -> bool:
        last_err = self.error_memory.get_last_error()
        if not last_err:
            self.debug_session.log_fix_attempt(
                "LLM Fix (Targeted)", False, "Error memory empty"
            )
            return False

        fix_data = await self.coding_agent.generate_targeted_fix(sg)
        if not fix_data:
            if self._check_syntax_error_repetition():
                self.logger.warning("Repeated syntax errors from LLM fixes.")
            return False

        fix_type, fix_name, fix_block = fix_data
        ok, msg = await self.reality_checker.reality_check(
            fix_block, sg.get("description", ""), context=last_err.get("message")
        )
        self.debug_session.log_validation_attempt(ok, msg)
        if not ok:
            self.error_memory.add_error(
                "LLM fix block failed reality check",
                {"fix_type": fix_type, "fix_name": fix_name},
            )
            self.debug_session.log_fix_attempt(
                "LLM Fix (Targeted)", False, "Reality check failed"
            )
            return False

        new_full_code = self._integrate_targeted_fix(
            code_before, fix_block, fix_name, fix_type
        )
        if new_full_code is None or new_full_code == code_before:
            self.debug_session.log_fix_attempt(
                "LLM Fix (Targeted)", False, "Integration failed or no change"
            )
            return False

        await self.codebase_manager.integrate_fixed_code(
            new_full_code, description=f"LLM Fix ({fix_type} {fix_name})"
        )
        self.debug_session.log_fix_attempt(
            "LLM Fix (Targeted)", True, f"Integrated {fix_type} {fix_name}"
        )
        self._consecutive_syntax_error_count = 0
        self._consecutive_stagnation_count = 0
        return True

    # ------------------------------------------------------------------ #
    # main loop per subgoal
    # ------------------------------------------------------------------ #
    async def process_subgoal(
        self, sg: Dict[str, Any], index: int, objective: str
    ) -> bool:
        sg_desc = sg.get("description", f"Subgoal #{index}")
        self.logger.info(f"Processing Subgoal {index}: {sg_desc}")

        self._consecutive_syntax_error_count = 0
        self._consecutive_stagnation_count = 0
        self._last_stagnant_code_hash = None

        for attempt in range(1, self.MAX_ATTEMPTS_PER_SUBGOAL + 1):
            code_before = self.codebase_manager.get_codebase()
            code_hash_before = self._get_code_hash(code_before)

            exec_res = await self.code_executor.execute_code(code_before)
            rc = exec_res.get("returncode")
            timeout = exec_res.get("timeout", False)

            if rc == 0 and not timeout:
                sg["status"] = "completed"
                self.completed_subgoals.append(sg_desc)
                return True

            if self._check_syntax_error_repetition():
                sg["status"] = "failed"
                return False

            if attempt > 1 and code_hash_before == self._last_stagnant_code_hash:
                self._consecutive_stagnation_count += 1
                if (
                    self._consecutive_stagnation_count
                    >= self.STAGNATION_THRESHOLD
                ):
                    sg["status"] = "failed"
                    return False
            else:
                self._consecutive_stagnation_count = 0

            self._last_stagnant_code_hash = code_hash_before

            if attempt < self.MAX_ATTEMPTS_PER_SUBGOAL:
                await self._attempt_debug_fix(sg, objective, code_before)

        sg["status"] = "failed"
        return False

    # ------------------------------------------------------------------ #
    # progress helper
    # ------------------------------------------------------------------ #
    def track_progress(self) -> dict:
        total = len(self.completed_subgoals)
        return {
            "total": total,
            "completed": total,
            "failed": 0,
            "pending": 0,
            "summary": f"{total} completed",
        }

class CodingAgent:
    """
    Generates code-fixes with the LLM.
        • initial full-script syntax fixes
        • targeted function/class fixes
    VERSION: Instrumented + Raw LLM Log + Corrected _extract_targeted_fix
    """

    def __init__(
        self,
        llm_client: "LLMClient",
        codebase_manager: "CodebaseManager",
        error_memory: "ErrorMemory",
        reality_checker: "RealityChecker",
        debug_session: "DebugSession",
        config: "Config",
    ):
        self.llm_client       = llm_client
        self.codebase_manager = codebase_manager
        self.error_memory     = error_memory
        self.reality_checker  = reality_checker
        self.debug_session    = debug_session
        self.config           = config

        self.logger = logging.getLogger(f"autonomous_debugger.{self.__class__.__name__}")
        self.logger.info("CodingAgent initialised (Instrumented + Raw LLM Log + Extract Fix).")
        print("[CodingAgent] Initialised (Instrumented + Raw LLM Log + Extract Fix).")

        # link DebugSession to LLMClient (if not already done)
        try:
            if hasattr(self.llm_client, "link_debug_session") and \
               callable(getattr(self.llm_client, "link_debug_session")) and \
               getattr(self.llm_client, "debug_session", True) is None:
                pass
        except Exception as e:
            self.logger.warning(f"Could not link DebugSession to LLMClient: {e}")

    # --- INTERNAL PROMPT HELPERS ---
    def _build_initial_syntax_fix_prompt(self, broken_code: str, lint: dict) -> str:
        line   = lint.get("line", "N/A")
        offset = lint.get("offset", "N/A")
        text   = lint.get("text_context", "")
        msg    = lint.get("error_message", "Syntax error")

        return (f"You are an expert Python developer.\n\nThe script below fails to parse.\n"
                f"Error:  {msg}\nLine:   {line}\nOffset: {offset}\nContext: {text}\n\n"
                f"Fix **only** that syntax problem – no stylistic refactors – and return "
                f"the *entire corrected file* in one ```python block.\n\n"
                f"```python\n{broken_code}\n```")

    def _build_targeted_fix_prompt(self, subgoal: dict) -> str:
        last_err = self.error_memory.get_last_error() or {}
        err_msg  = last_err.get("message", "Error context not available")
        err_ctx  = last_err.get("context", {}) # This context comes from ErrorMemory
        err_traceback = ""; err_line = "N/A"; err_type = "N/A"

        if isinstance(err_ctx, dict):
            err_line = str(err_ctx.get("line", "N/A"))
            err_type = err_ctx.get("error_type", "N/A")
            tb_keys = ["inner_traceback", "user_traceback", "signal_traceback", "stderr_snippet", "stderr"]
            for key in tb_keys:
                tb = err_ctx.get(key)
                if tb and isinstance(tb, str):
                    err_traceback = f"\nRelevant Traceback/Stderr ({key}):\n```text\n{tb[:1500].strip()}...\n```"
                    break

        full_code = self.codebase_manager.get_codebase()
        error_history_prompt = self.error_memory.get_improvement_prompt() # Get formatted history

        prompt = f"""You are an expert Python debugger. Your task is to analyze the provided Python script and error context, identify the SINGLE function or class definition responsible for the error, and return ONLY the corrected definition for that specific function or class.

**Objective:** {subgoal.get('description', 'Fix the runtime error')}

**Latest Error Context:**
*   **Error Message:** {err_msg}
*   **Error Type:** {err_type}
*   **Error Line Hint:** {err_line}
{err_traceback}

**Recent Error History (if any):**
{error_history_prompt}

**Full Target Python Script (This is the code you MUST modify):**
{CODE_START_DELIMITER}
{full_code}
{CODE_END_DELIMITER}

**Analysis & Instructions:**
1.  **Identify Target:** Based on the **Traceback/Stderr** and **Error Line**, determine the `def function_name(...)` or `class ClassName:` definition where the error originates or that needs modification.
2.  **Isolate & Correct:** Rewrite the **ENTIRE source code** for **ONLY** that single identified function or class definition. Apply the minimal, most direct change needed to fix the specific error. Preserve original signatures/structure unless necessary.
3.  **Format Output:** Return the response ONLY in the following precise format. NO explanation, preamble, or other text outside these delimiters.

    ```text
    {FIX_INFO_START_DELIMITER}
    {{
      "type": "<'function' or 'class'>",
      "name": "<exact_name_of_function_or_class_from_TARGET_SCRIPT_as_defined>"
    }}
    {FIX_INFO_END_DELIMITER}
    {TARGETED_CODE_START_DELIMITER}
    # The complete, corrected source code for the identified function or class definition.
    # Include the 'def' or 'class' line, any decorators, the signature,
    # the docstring (if present), and the entire body. Ensure proper indentation.
    {TARGETED_CODE_END_DELIMITER}
    ```

4.  **CRITICAL:** The output MUST contain BOTH the `{FIX_INFO_START_DELIMITER}` JSON block AND the `{TARGETED_CODE_START_DELIMITER}` code block.
5.  **Avoid superficial fixes:** Address the root cause.
"""
        return prompt

    def _build_meta_debug_prompt(self, original_prompt: str, prev_resp: str, reason: str, bad_block: Optional[str]) -> str:
        hdr = "The previous response could not be processed:\n" + reason
        tail = (f"\nProblematic block:\n```python\n{bad_block[:500]}...\n```" if bad_block else "")
        MAX_ORIG_PROMPT_LEN_FOR_META = 15000 # Adjusted for potentially large original prompts
        truncated_orig_prompt = original_prompt
        if len(original_prompt) > MAX_ORIG_PROMPT_LEN_FOR_META:
            half_len = MAX_ORIG_PROMPT_LEN_FOR_META // 2
            truncated_orig_prompt = original_prompt[:half_len] + "\n... (Original Prompt Truncated) ...\n" + original_prompt[-half_len:]
            self.logger.info(f"Truncated original prompt for meta-debug request (original len {len(original_prompt)}).")

        return f"""--- Meta-Debugging Request: Correct Your Previous Response ---

{hdr}

**Original Request (Summary/Truncated):**
{truncated_orig_prompt}

**Your Previous Malformed/Invalid Response:**
```text
{prev_resp[:1000]}...
```
{tail}

Required Output Format (Reminder):
{FIX_INFO_START_DELIMITER}
{{
  "type": "<'function' or 'class'>",
  "name": "<exact_name_of_function_or_class_from_TARGET_SCRIPT_as_defined>"
}}
{FIX_INFO_END_DELIMITER}
{TARGETED_CODE_START_DELIMITER}
# The complete, corrected source code for the identified function or class definition.
{TARGETED_CODE_END_DELIMITER}

Generate ONLY the corrected response in the required format. No extra text or explanations."""

    # --- PUBLIC METHODS ---
    async def generate_initial_syntax_fix(self, broken_code: str, lint_details: dict) -> Optional[str]:
        prompt = self._build_initial_syntax_fix_prompt(broken_code, lint_details)
        max_tokens = max(self.config.llm_default_max_output_tokens, len(broken_code.splitlines()) * 30 + 1000)
        self.logger.debug(f"Calculated max_tokens for initial syntax fix: {max_tokens}")

        for attempt in (1, 2):
            print(f"[CodingAgent] Initial-fix attempt {attempt} (tokens≈{max_tokens})")
            self.logger.info(f"Calling LLM for initial syntax fix (Attempt {attempt}).")
            global CodeExtractor, preemptive_linting
            resp = await self.llm_client.call_llm(prompt, max_tokens=max_tokens)

            if not resp or resp.startswith("Error:"):
                self.logger.error(f"LLM call failed for initial syntax fix attempt {attempt}: {resp}")
                if attempt == 1:
                     prompt = "Your previous response was empty or an error. Please provide the corrected script in a ```python block."
                continue

            code = CodeExtractor.extract_code(resp)

            if not code:
                self.logger.warning(f"Initial-fix response attempt {attempt} lacked a parsable code block.")
                self.logger.debug(f"RAW LLM Response (Initial Fix Attempt {attempt} - No Code Block):\n{resp[:1000]}...")
                print(f"[CodingAgent DEBUG] Raw LLM Resp (Initial Fix {attempt} - No Code Block):\n{resp[:200]}...")
                if attempt == 1:
                    prompt = ("Your previous reply did not include a valid ```python code block. "
                              "Please return ONLY the full corrected Python script inside a single ```python block.")
                continue

            lint = preemptive_linting(code)
            self.logger.debug(f"Initial-fix lint result (Attempt {attempt}): Passed={lint.passed}, Msg='{lint.message}'")
            if lint.passed:
                print("[CodingAgent] Initial-fix validated OK.")
                return code

            syntax_err_details = f"Syntax error: {lint.message} (Details: {lint.details or 'N/A'})"
            self.logger.warning(f"Initial-fix attempt {attempt} failed lint check: {syntax_err_details}")
            self.logger.debug(f"RAW LLM Response (Initial Fix Attempt {attempt} - Syntax Error):\n{resp[:1000]}...")
            self.logger.debug(f"Extracted Code with Syntax Error (Initial Fix Attempt {attempt}):\n{code[:1000]}...")
            print(f"[CodingAgent DEBUG] Raw LLM Resp (Initial Fix {attempt} - Syntax Error):\n{resp[:200]}...")
            if attempt == 1:
                prompt = (f"The script you produced contained a syntax error: {syntax_err_details}\n"
                          "Please review the error, correct the script, and return the entire valid Python script inside a single ```python block.")

        print("[CodingAgent] Initial-fix attempts exhausted / failed.")
        return None

    async def generate_targeted_fix(self, subgoal: dict) -> Optional[Tuple[str, str, str]]:
        global preemptive_linting
        sg_desc_log = str(subgoal.get("description", "N/A"))[:100]
        self.logger.info(f"CodingAgent:generate_targeted_fix for Subgoal: {sg_desc_log}...")
        self.debug_session.log_llm_call()

        initial_prompt = self._build_targeted_fix_prompt(subgoal)
        max_output_toks = self.config.llm_default_max_output_tokens
        self.logger.debug(f"Generated initial targeted fix prompt (Length: {len(initial_prompt)} chars).")

        llm_response: Optional[str] = None
        final_fix_details: Optional[Tuple[str, str, str]] = None
        last_failure_reason: str = "Unknown failure before LLM call."
        last_failed_code_block_for_meta_prompt: Optional[str] = None

        for attempt in range(1, 3):
            current_prompt_for_llm = initial_prompt
            attempt_description = "Initial targeted fix"

            if attempt > 1:
                self.debug_session.log_llm_call()
                if last_failed_code_block_for_meta_prompt:
                    attempt_description = "Meta-debug (syntax correction)"
                    current_prompt_for_llm = self._build_meta_debug_prompt(
                        initial_prompt, llm_response or "Prev Resp N/A", last_failure_reason, last_failed_code_block_for_meta_prompt)
                else:
                    attempt_description = "Meta-debug (format/extraction correction)"
                    current_prompt_for_llm = self._build_meta_debug_prompt(
                        initial_prompt, llm_response or "Prev Resp N/A", last_failure_reason, None)  # Fixed: explicitly pass None for bad_block
                self.logger.info(f"Attempt {attempt}: Retrying LLM call ({attempt_description}).")
            else:
                self.logger.info(f"Attempt {attempt}: Calling LLM for {attempt_description}.")

            try:
                print(f"[CodingAgent] {attempt_description} attempt {attempt}")
                llm_response = await self.llm_client.call_llm(current_prompt_for_llm, max_tokens=max_output_toks)
                print(f"[DEBUG] LLM Raw Response (Attempt {attempt}):\n{llm_response[:1000]}...") # ADDED LINE

                if not llm_response or (isinstance(llm_response, str) and llm_response.strip().startswith("Error:")):
                    error_msg_from_llm = llm_response if llm_response else "LLM client returned empty response"
                    last_failure_reason = f"LLM client failed or returned error on attempt {attempt}: {error_msg_from_llm[:500]}"
                    self.logger.error(last_failure_reason)
                    self.error_memory.add_error("LLMCallFailed", {"attempt": attempt, "reason": last_failure_reason, "subgoal": sg_desc_log})
                    if not llm_response:
                        break
                    continue

                self.logger.info(f"LLM response received (Attempt {attempt}, len {len(llm_response)}). Attempting extraction...")
                fix_type, fix_name, extracted_block_raw, extract_err = self._extract_targeted_fix(llm_response)

                if extract_err:
                    last_failure_reason = extract_err
                    last_failed_code_block_for_meta_prompt = None
                    self.logger.error(f"Extraction failed (Attempt {attempt}): {last_failure_reason}")
                    self.logger.debug(f"RAW LLM Response (Attempt {attempt} - Extraction Failed):\n{llm_response[:1500]}...")
                    print(f"[CodingAgent DEBUG] Raw LLM Resp (Extraction Fail {attempt}):\n{llm_response[:250]}...")
                    if attempt == 2: self.logger.error("Meta-debug extraction also failed.")
                    continue

                code_to_lint: Optional[str] = None
                try:
                    if extracted_block_raw is not None:
                        code_to_lint = textwrap.dedent(extracted_block_raw).strip()
                    else:
                        raise ValueError("Internal error: Extracted block was None despite no extraction error.")
                except Exception as e_dedent:
                    last_failure_reason = f"Error dedenting extracted code block: {e_dedent}"
                    last_failed_code_block_for_meta_prompt = extracted_block_raw
                    self.logger.error(f"{last_failure_reason} (Attempt {attempt}). Block: {extracted_block_raw[:200] if extracted_block_raw else 'N/A'}")
                    self.error_memory.add_error("LLMFixBlockDedentError", {"attempt": attempt, "reason": last_failure_reason, "code_snippet": extracted_block_raw[:500] if extracted_block_raw else None})
                    if attempt == 2: self.logger.error("Dedent error on meta-debug attempt.")
                    continue

                if not code_to_lint:
                    last_failure_reason = "Extracted code block is empty after dedent/strip."
                    last_failed_code_block_for_meta_prompt = extracted_block_raw
                    self.logger.error(f"{last_failure_reason} (Attempt {attempt}).")
                    self.error_memory.add_error("LLMFixBlockEmpty", {"attempt": attempt, "reason": last_failure_reason})
                    if attempt == 2: self.logger.error("Empty block on meta-debug attempt.")
                    continue

                lint_result = preemptive_linting(code_to_lint)
                if lint_result.passed:
                    self.logger.info(f"Syntax check PASSED for extracted code block (Attempt {attempt}) for {fix_type} '{fix_name}'.")
                    final_fix_details = (fix_type, fix_name, code_to_lint)
                    self.debug_session.log_fix_attempt(f"CodingAgent LLM Gen ({attempt_description})", True, f"Generated valid block for {fix_type} {fix_name}")
                    print(f"[CodingAgent] Targeted-fix OK ({fix_type} {fix_name})")
                    print(f"[DEBUG] Extracted Fix Block ({fix_type} {fix_name}):\n{code_to_lint}") # ADDED LINE
                    break
                else:
                    last_failure_reason = f"Syntax error in generated block (Attempt {attempt}): {lint_result.message} {lint_result.details or ''}"
                    last_failed_code_block_for_meta_prompt = code_to_lint
                    self.logger.error(last_failure_reason)
                    self.error_memory.add_error("LLMFixBlockSyntaxError", {"attempt": attempt, "reason": last_failure_reason, "code_snippet": code_to_lint[:500], "lint_details": lint_result.details})
                    self.logger.debug(f"RAW LLM Response (Attempt {attempt} - Syntax Error in Block):\n{llm_response[:1500]}...")
                    self.logger.debug(f"Extracted Code with Syntax Error (Attempt {attempt}):\n{code_to_lint[:1000]}...")
                    print(f"[CodingAgent DEBUG] Raw LLM Resp (Syntax Fail {attempt}):\n{llm_response[:250]}...")
                    if attempt == 2: self.logger.error("Syntax error in meta-debug attempt's generated code. Giving up.")
                    continue

            except Exception as e_inner_loop:
                last_failure_reason = f"Unexpected error in CodingAgent.generate_targeted_fix loop (Attempt {attempt}): {e_inner_loop}"
                self.logger.error(last_failure_reason, exc_info=True)
                self.error_memory.add_error("CodingAgentInternalError", {"attempt": attempt, "reason": str(e_inner_loop), "traceback": traceback.format_exc()})
                break

        if final_fix_details is None:
            self.logger.error(f"Failed to generate valid targeted fix after {attempt} attempt(s). Last reason: {last_failure_reason}")
            self.debug_session.log_fix_attempt(f"CodingAgent LLM Gen (Overall)", False, f"Failed after {attempt} attempts. Last: {last_failure_reason[:100]}")
            print("[CodingAgent] Targeted-fix attempts exhausted / failed.")

        self.logger.info(f"CodingAgent:generate_targeted_fix finished. Success: {final_fix_details is not None}")
        return final_fix_details

    # --- PRIVATE _extract_targeted_fix HELPER (Corrected Syntax) ---
    def _extract_targeted_fix(self, response: str) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
        self.logger.debug("Attempting extraction from LLM response (len %d)...", len(response) if response else 0)
        if not response or not response.strip():
            reason = "Extraction failed: LLM response empty/whitespace."
            self.logger.error(reason)
            return None, None, None, reason
        info_match = re.search(f"{re.escape(FIX_INFO_START_DELIMITER)}(.*?){re.escape(FIX_INFO_END_DELIMITER)}", response, re.DOTALL)
        if not info_match:
            reason = (f"Extraction failed: Missing fix info delimiters ({FIX_INFO_START_DELIMITER}...). Snippet: {response[:300]}")
            self.logger.error(reason)
            return None, None, None, reason
        info_json_str = info_match.group(1).strip()
        if not info_json_str:
            reason = "Extraction failed: Fix info block empty."
            self.logger.error(reason)
            return None, None, None, reason
        fix_type: Optional[str] = None
        fix_name: Optional[str] = None
        try:
            fix_info = json.loads(info_json_str)
            fix_type = fix_info.get("type")
            fix_name = fix_info.get("name")
            if fix_type not in ["function", "class"] or not fix_name or not isinstance(fix_name, str) or not fix_name.strip():
                raise ValueError(f"Invalid/missing type/name. Type='{fix_type}', Name='{fix_name}'")
            fix_name = fix_name.strip()
            self.logger.debug(f"Extracted fix info: type='{fix_type}', name='{fix_name}'")
        except (json.JSONDecodeError, ValueError) as e_json_val:
            reason = (f"Extraction failed: Failed to parse/validate fix info JSON: {e_json_val}. Content: '{info_json_str}'")
            self.logger.error(reason)
            return None, None, None, reason
        code_pattern_str = rf"{re.escape(TARGETED_CODE_START_DELIMITER)}\s*\n?(.*?)\n?\s*{re.escape(TARGETED_CODE_END_DELIMITER)}"
        code_match = re.search(code_pattern_str, response, re.DOTALL)
        if not code_match:
            reason = (f"Extraction failed: Missing code block delimiters ({TARGETED_CODE_START_DELIMITER}...). Snippet after info: {response[info_match.end():info_match.end()+300]}")
            self.logger.error(reason)
            return None, None, None, reason
        fix_block_raw = code_match.group(1)
        if not fix_block_raw.strip():
            reason = ("Extraction failed: Extracted code block empty/whitespace.")
            self.logger.error(reason)
            return None, None, None, reason
        self.logger.info(f"Successfully extracted fix for {fix_type} '{fix_name}' (raw len: {len(fix_block_raw)} chars).")
        return fix_type, fix_name, fix_block_raw, ""

class CodebaseManager:
    """
    Holds current code state and lightweight history.
    """

    def __init__(self, error_memory: ErrorMemory):
        self.error_memory = error_memory
        self.current_code = ""
        self.history: List[Tuple[str, str]] = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("CodebaseManager initialized.")

    # --------------------------------------------------------------
    def _get_hash(self, code: str) -> str:
        return hashlib.md5(code.encode("utf-8")).hexdigest()[:10]

    # --------------------------------------------------------------
    def initialize_codebase(self, code: str):
        self.current_code = code
        self.history = [("Initial", self._get_hash(code))]
        self.logger.info(
            f"Codebase initialized (len={len(code)}, hash={self.history[0][1]})."
        )

    # --------------------------------------------------------------
    def get_codebase(self) -> str:
        return self.current_code

    def get_full_codebase(self) -> str:
        return self.current_code  # single-file variant

    # --------------------------------------------------------------
    async def integrate_fixed_code(self, new_code: str, description: str = ""):
        old_hash = self._get_hash(self.current_code)
        self.current_code = new_code
        new_hash = self._get_hash(new_code)
        self.history.append((description or "Update", new_hash))
        if len(self.history) > 50:
            self.history = self.history[-50:]
        self.logger.info(
            f"Codebase updated: {old_hash} → {new_hash} ({description})."
        )

# --- StateManager (placeholder) ---------------------------------------------
class StateManager:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("StateManager initialized (placeholder).")

    def create_checkpoint(self, description: str):
        self.logger.info(f"Checkpoint (mock): {description}")

    def monitor_resources(self):
        pass


# --- Inspection Agent Framework --------------------------------------------
class InspectionError(Exception):
    pass


class BaseInspection(abc.ABC):
    name: str = "base_inspection"
    description: str = "Base class for inspections"

    def __init__(self):
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self.logger.info(f"Inspection initialized: {self.name}")

    @abc.abstractmethod
    def execute(
        self, target: Any, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return {
            "insp_name": self.name,
            "status": "skipped",
            "reason": "Base class execute called",
        }


class FilesystemInspection(BaseInspection):
    name = "Filesystem Exists Check"
    description = "Checks if path exists/is directory."

    def execute(
        self, target: Any, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not isinstance(target, str):
            return {
                "insp_name": self.name,
                "status": "skipped",
                "reason": "Target not str",
            }
        try:
            path_exists = os.path.exists(target)
            is_dir = os.path.isdir(target) if path_exists else False
            return {
                "insp_name": self.name,
                "target_path": target,
                "exists": path_exists,
                "is_directory": is_dir,
                "status": "completed",
            }
        except Exception as e:
            return {
                "insp_name": self.name,
                "status": "failed",
                "error": str(e),
            }


class CodeStyleInspection(BaseInspection):
    name = "Code Style Check (Black)"
    description = "Checks formatting with Black."

    def execute(
        self, target: Any, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not BLACK_AVAILABLE:
            return {
                "insp_name": self.name,
                "status": "skipped",
                "reason": "'black' missing",
            }
        if not isinstance(target, str) or not target.strip():
            return {
                "insp_name": self.name,
                "status": "skipped",
                "reason": "Code empty or wrong type",
            }
        try:
            mode = black.FileMode()
            formatted = black.format_str(target, mode=mode)
            ok = target == formatted
            return {
                "insp_name": self.name,
                "is_formatted": ok,
                "findings": [] if ok else ["Needs black formatting."],
                "status": "completed",
            }
        except black.NothingChanged:
            return {
                "insp_name": self.name,
                "is_formatted": True,
                "findings": [],
                "status": "completed",
            }
        except Exception as e:
            return {
                "insp_name": self.name,
                "status": "failed",
                "error": f"Black error: {e}",
            }


class InspectionAgent:
    def __init__(self, agent_id: str, inspections: List[BaseInspection]):
        self.agent_id = agent_id
        self.logger = logging.getLogger(
            f"{self.__class__.__name__}[{self.agent_id}]"
        )
        self._inspections: Dict[str, BaseInspection] = {
            insp.name: insp for insp in inspections
        }
        self.logger.info(
            f"InspectionAgent '{self.agent_id}' initialized with "
            f"{len(self._inspections)} inspections."
        )

    def list_inspections(self) -> List[Dict[str, str]]:
        return [
            {"name": n, "description": i.description}
            for n, i in self._inspections.items()
        ]

    def perform_inspection(
        self, inspection_name: str, target: Any, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if inspection_name not in self._inspections:
            return {
                "agent_id": self.agent_id,
                "insp_name": inspection_name,
                "status": "error",
                "error": "Inspection not found",
            }
        try:
            result = self._inspections[inspection_name].execute(
                target=target, params=params
            )
            result["agent_id"] = self.agent_id
            return result
        except InspectionError as e_insp:
            return {
                "agent_id": self.agent_id,
                "insp_name": inspection_name,
                "status": "failed",
                "error": str(e_insp),
            }
        except Exception as e:
            return {
                "agent_id": self.agent_id,
                "insp_name": inspection_name,
                "status": "error",
                "error": f"Unexpected error: {e}",
            }


# --- SystemValidator (Class) -------------------------------------------------
import ast
import json
import logging
from typing import Dict, Any

# Again, assume these are imported elsewhere:
# from your_module import LLMClient, DebugSession

logger = logging.getLogger("autonomous_debugger")

class SystemValidator:
    """
    Runs syntax checks and a two-stage LLM validation:
      1) Does the code meet the objective?
      2) Multimodal check on the execution artifacts/results.
    """

    def __init__(self):
        self.inspection_agent = None
        self.error_memory = None
        self.llm_client = None
        self.debug_session = None
        logger.info("SystemValidator initialized.")

    def configure(self, inspection_agent: InspectionAgent, error_memory: ErrorMemory, llm_client: LLMClient):
        self.inspection_agent = inspection_agent
        self.error_memory = error_memory
        self.llm_client = llm_client

    def link_debug_session(self, debug_session: DebugSession):
        self.debug_session = debug_session

    async def validate_syntax(self, code: str) -> bool:
        """
        Quick AST parse. Returns False on any SyntaxError.
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            logger.warning(f"Syntax validation failed: {e}")
            return False

    async def validate_codebase(
        self,
        final_code: str,
        objective: str,
        subgoals: list,
        exec_result: Dict[str, Any]
    ) -> bool:
        """
        1) Syntax check
        2) LLM-based objective check → JSON
        3) Multimodal LLM check on exec_result/artifacts
        """
        # 1) Syntax
        if not await self.validate_syntax(final_code):
            return False

        # 2) Objective check
        rc = exec_result.get("returncode")
        stderr = bool(exec_result.get("stderr"))
        timed_out = exec_result.get("timeout", False)
        prompt = f"""You are verifying whether the Python script meets its objective.

Objective: {objective}

Execution result:
  • return code: {rc}
  • timed-out: {timed_out}
  • stderr present: {stderr}

Respond with JSON: {{"validation_passed": true|false, "reason": "<brief>"}}
"""
        resp = await self.llm_client.call_llm(prompt)
        try:
            data = json.loads(resp)
            ok = bool(data.get("validation_passed"))
            reason = data.get("reason", "")
        except Exception:
            ok = False
            reason = "Malformed LLM validation response"
        if self.debug_session:
            self.debug_session.log_validation_attempt(ok, reason)
        if not ok:
            logger.error(f"Objective validation failed: {reason}")
            return False

        # 3) Multimodal artifact check
        multi_ok = await self.validate_with_multimodal_analysis(exec_result, objective)
        if not multi_ok:
            logger.error("Multimodal validation failed")
        return multi_ok

    async def validate_with_multimodal_analysis(self, task_result: Dict[str, Any], original_goal: str) -> bool:
        """
        Leverage Gemini's vision/text capabilities to judge actual outputs.
        """
        prompt = f"""Multimodal check: Given the objective '{original_goal}' and the result {task_result}, 
decide whether the objective is fully met. 
Respond with JSON: {{"multimodal_passed": true|false}}."""
        resp = await self.llm_client.call_llm(prompt)
        try:
            data = json.loads(resp)
            return bool(data.get("multimodal_passed"))
        except Exception:
            logger.warning("Malformed multimodal validation response")
            return False


class MainExecutor:
    """
    Orchestrates the whole debugging pipeline.
    EXTRA INSTRUMENTATION: prints + debug logs at every stage.
    """

    def __init__(self, config: "Config", components: Dict[str, Any]):
        self.config           = config
        self.components       = components
        self.logger           = logging.getLogger(self.__class__.__name__)

        # required components -------------------------------------------------
        self.subgoal_agent    = components.get("subgoal_agent")
        self.process_manager  = components.get("process_manager")
        self.system_validator = components.get("system_validator")
        self.error_memory     = components.get("error_memory")
        self.codebase_manager = components.get("codebase_manager")
        self.code_executor    = components.get("code_executor")
        self.coding_agent     = components.get("coding_agent")
        self.debug_session    = components.get("debug_session")
        self.unified_logger   = components.get("unified_logger")

        missing = [k for k,v in {
            "subgoal_agent":self.subgoal_agent,
            "process_manager":self.process_manager,
            "system_validator":self.system_validator,
            "error_memory":self.error_memory,
            "codebase_manager":self.codebase_manager,
            "code_executor":self.code_executor,
            "coding_agent":self.coding_agent,
            "debug_session":self.debug_session,
            "unified_logger":self.unified_logger,
        }.items() if v is None]
        if missing:
            raise InitializationError(f"MainExecutor missing: {missing}")

        self._loaded_file_path : Optional[str] = None
        self.generated_subgoals: List[Dict[str,Any]] = []
        self.current_objective : str = getattr(config, "user_objective", "")

        self.logger.info("MainExecutor initialised (instrumented).")
        print("[MainExecutor] Initialised.")

    # ------------------------------------------------------------------ #
    # INTERNAL: load script
    # ------------------------------------------------------------------ #
    def _load_code(self) -> Optional[str]:
        path = self.config.target_file_path
        self.logger.debug(f"Loading script from {path}")
        print(f"[MainExecutor] Loading '{path}'")

        if not path or "/path/to/your" in path:
            print("[MainExecutor] Invalid path placeholder.")
            return None
        if not os.path.exists(path):
            print("[MainExecutor] File not found.")
            return None
        if not os.path.isfile(path):
            print("[MainExecutor] Path is not a file.")
            return None

        try:
            with open(path,"r",encoding="utf-8") as f:
                code = f.read()
            if not code.strip():
                print("[MainExecutor] File is empty.")
                return None
            self._loaded_file_path = path
            return textwrap.dedent(code).strip()
        except Exception as e:
            print(f"[MainExecutor] Error reading file: {e}")
            return None

    # ------------------------------------------------------------------ #
    # INTERNAL: save script
    # ------------------------------------------------------------------ #
    def _save_debugged_code(self, code: str, suffix: str="") -> bool:
        base = self.config.output_filename_base
        if self._loaded_file_path:
            base = f"{os.path.splitext(os.path.basename(self._loaded_file_path))[0]}_debugged"
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname= f"{base}{('_'+suffix) if suffix else ''}_{ts}.py"

        lint = preemptive_linting(code)
        if not lint.passed:
            print(f"[MainExecutor] Lint failed – NOT saving ({lint.message}).")
            return False
        try:
            with open(fname,"w",encoding="utf-8") as f:
                f.write(code if code.endswith("\n") else code+"\n")
            print(f"[MainExecutor] Saved: {fname}")
            return True
        except Exception as e:
            print(f"[MainExecutor] Save failed: {e}")
            return False

    # ------------------------------------------------------------------ #
    # PUBLIC PIPELINE STEP 1 – initialise
    # ------------------------------------------------------------------ #
    async def initialize(self) -> bool:
        print("\n=== Phase: INITIALISATION ===")
        code_raw = self._load_code()
        if code_raw is None:
            self.logger.error("Initial load failed.")
            return False

        lint = preemptive_linting(code_raw)
        if lint.passed:
            print("[MainExecutor] Script syntax-OK on first load.")
            self.codebase_manager.initialize_codebase(code_raw)
        else:
            print(f"[MainExecutor] Syntax error L{lint.details.get('line')}: {lint.message}")
            fixed = await self.coding_agent.generate_initial_syntax_fix(code_raw, lint.details or {})
            if not fixed:
                print("[MainExecutor] CodingAgent could not auto-fix syntax.")
                return False
            lint2 = preemptive_linting(fixed)
            if not lint2.passed:
                print("[MainExecutor] Auto-fix still invalid – abort.")
                return False
            print("[MainExecutor] Auto-fix accepted.")
            self._save_debugged_code(fixed, "initial_fix")
            self.codebase_manager.initialize_codebase(fixed)

        # objective ----------------------------------------------------------------
        if not self.current_objective:
            fn = os.path.basename(self._loaded_file_path)
            self.current_objective = f"Debug the Python script '{fn}' so it exits with RC=0."

        print(f"[MainExecutor] Objective: {self.current_objective}")

        # subgoals -----------------------------------------------------------------
        self.generated_subgoals = await self.subgoal_agent.generate_subgoals(self.current_objective)
        print(f"[MainExecutor] {len(self.generated_subgoals)} subgoal(s) created.")
        return True

    # ------------------------------------------------------------------ #
    # PUBLIC PIPELINE STEP 2 – process subgoals
    # ------------------------------------------------------------------ #
    async def run_subgoal_processing(self) -> bool:
        print("\n=== Phase: SUBGOAL PROCESSING ===")
        ok = await self.process_manager.process_subgoal(
            self.generated_subgoals[0], 1, self.current_objective
        )
        print(f"[MainExecutor] Subgoal processing {'succeeded' if ok else 'failed'}.")
        return ok

    # ------------------------------------------------------------------ #
    # PUBLIC PIPELINE STEP 3 – final validation
    # ------------------------------------------------------------------ #
    async def run_final_steps(self) -> bool:
        print("\n=== Phase: FINAL STEPS ===")
        final_code = self.codebase_manager.get_full_codebase()
        exec_res   = await self.code_executor.execute_code(final_code,
                            timeout=self.config.final_execution_timeout_seconds)

        rc     = exec_res.get("returncode")
        t_out  = exec_res.get("timeout", False)
        print(f"[MainExecutor] Final execution RC={rc} timeout={t_out}")

        valid = await self.system_validator.validate_codebase(
            final_code, self.current_objective, self.generated_subgoals, exec_res
        )
        print(f"[MainExecutor] LLM validation {'passed' if valid else 'failed'}.")

        if rc==0 and not t_out and valid:
            self._save_debugged_code(final_code)
            return True
        return False

    # ------------------------------------------------------------------ #
    # PUBLIC ORCHESTRATOR
    # ------------------------------------------------------------------ #
    async def run_process(self) -> bool:
        start = time.monotonic()
        try:
            if not await self.initialize():
                return False
            if not await self.run_subgoal_processing():
                return False
            return await self.run_final_steps()
        finally:
            dur = time.monotonic() - start
            print(f"[MainExecutor] TOTAL DURATION: {dur:.2f}s")


# --- Utility Functions ---

# Assume logging, sys, os, GOOGLE_GENAI_AVAILABLE, genai, google libs defined/imported
# --- Helper Functions for __main__ ---
def configure_logging_main(level_str: str):
    """Sets up global logging based on configuration."""
    level = getattr(logging, level_str.upper(), logging.INFO)
    root_logger = logging.getLogger(); root_logger.setLevel(level)
    # Remove existing handlers before adding new ones
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler); handler.close()
    formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(name)-20s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout); console_handler.setLevel(level); console_handler.setFormatter(formatter); root_logger.addHandler(console_handler)
    # Set levels for noisy libraries AFTER configuring root/handlers
    logging.getLogger('blib2to3').setLevel(logging.WARNING); logging.getLogger('asyncio').setLevel(logging.INFO); logging.getLogger('aiolimiter').setLevel(logging.WARNING)
    if GOOGLE_GENAI_AVAILABLE: logging.getLogger('google.api_core').setLevel(logging.WARNING); logging.getLogger('google.generativeai').setLevel(logging.INFO)
    logger = logging.getLogger('autonomous_debugger'); # Get main logger again after setup
    logger.info(f"--- Logging Configured (Level: {logging.getLevelName(level)}) ---"); print(f"--- Logging Configured (Level: {logging.getLevelName(level)}) ---")

def load_config() -> Config:
    """Loads or creates the main configuration object."""
    logger_cfg = logging.getLogger('autonomous_debugger.config') # Use main logger subsystem
    print("[Setup] Loading configuration...")
    config = Config() # Create instance with defaults
    # --- SET ACTUAL PATHS/KEYS HERE ---
    config.target_file_path = "/Users/williamwhite/testcodebase5.py" # <<< UPDATED PATH
    # API Key is loaded via RAW_KEYS in setup_api now
    config.api_key = "TEMPORARY_PLACEHOLDER" # Set temporary, will be overwritten
    # --- Validation ---
    if not config.target_file_path or "/path/to/your" in config.target_file_path: msg = "Target file path missing or placeholder in Config."; logger_cfg.critical(msg); print(f"CRITICAL: {msg}"); raise InitializationError(msg)
    logger_cfg.info("Configuration object created with defaults/paths."); print(f"[Setup] Target file path set to: {config.target_file_path}"); print(f"[Setup] Default LLM Model: {config.llm_model_name}"); return config

def cleanup_resources(temp_file_prefix: str = "temp_code_"):
    """Removes temporary files created during execution."""
    files_cleaned = 0; current_dir = "."; cl_logger = logging.getLogger("cleanup_resources")
    try: file_list = os.listdir(current_dir)
    except OSError as e: cl_logger.error(f"Cannot list dir '{current_dir}' for cleanup: {e}"); return
    prefixes_to_clean = (temp_file_prefix, "temp_test_")
    for filename in file_list:
        if filename.endswith(".py") and any(filename.startswith(p) for p in prefixes_to_clean):
            file_path = os.path.join(current_dir, filename)
            try: os.remove(file_path); cl_logger.debug(f"Cleaned temp file: {filename}"); files_cleaned += 1
            except OSError as e: cl_logger.error(f"Error removing temp file {filename}: {e}")
            except Exception as e: cl_logger.error(f"Unexpected error cleaning {filename}: {e}", exc_info=False)
    if files_cleaned > 0: cl_logger.info(f"Resource cleanup finished. Removed {files_cleaned} temp files.")
    else: cl_logger.info("Resource cleanup finished. No relevant temp files found.")


# --- setup_api Function (Using Rotation) ---
def setup_api(config: Config) -> bool:
    """Selects a random key, configures SDK, and verifies."""
    logger_setup = logging.getLogger('autonomous_debugger.setup')
    if not GOOGLE_GENAI_AVAILABLE: logger_setup.critical("google-generativeai missing."); print("\n[Setup Error] google-generativeai missing."); return False
    if not RAW_KEYS: logger_setup.critical("RAW_KEYS empty."); raise InitializationError("RAW_KEYS list is empty.")

    # Select and configure with a random key
    selected_key = random.choice(RAW_KEYS)
    config.api_key = selected_key # Update config object
    logger_setup.info(f"Configuring API with selected key ending: ****{selected_key[-4:]}")
    print(f"[Setup] Configuring with Rotated API Key: ****{selected_key[-4:]}")

    try:
        genai.configure(api_key=selected_key)
        logger_setup.info("Global genai.configure called successfully.")
        if config.verify_api_key_on_startup:
            logger_setup.info("Verifying selected API key via list_models...")
            try: models = [m.name for m in genai.list_models()]; logger_setup.info(f"API key OK ({len(models)} models found)."); print("[Setup] Selected API Key OK.")
            except google.auth.exceptions.DefaultCredentialsError as e_auth: logger_setup.error(f"Selected API key FAIL (list_models): {e_auth}"); print(f"\n[Setup Error] Selected API Key invalid/no permission: {e_auth}"); return False # Treat as fatal for setup
            except Exception as api_err: logger_setup.warning(f"Selected API key verify failed: {api_err}"); print(f"[Setup Warn] Selected API verify failed: {api_err}") # Non-fatal
        return True
    except Exception as e_cfg: logger_setup.critical(f"Failed genai.configure with selected key: {e_cfg}", exc_info=True); print(f"\n[Setup Error] Failed SDK config: {e_cfg}"); return False

# --- End setup_api ---
def configure_logging_main(level_str: str):
    """Sets up global logging based on configuration."""
    level = getattr(logging, level_str.upper(), logging.INFO)
    root_logger = logging.getLogger(); root_logger.setLevel(level)
    # Remove existing handlers before adding new ones
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler); handler.close()
    formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(name)-20s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout); console_handler.setLevel(level); console_handler.setFormatter(formatter); root_logger.addHandler(console_handler)

    # Set levels for noisy libraries AFTER configuring root/handlers
    logging.getLogger('blib2to3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.INFO) # Keep asyncio INFO for event loop messages

    # --- REDUCE VERBOSITY HERE ---
    logging.getLogger('aiolimiter').setLevel(logging.WARNING)
    # Use the full logger names as defined in the classes
    logging.getLogger('autonomous_debugger.ApiKeyManager').setLevel(logging.WARNING) # Was INFO/DEBUG
    logging.getLogger('autonomous_debugger.LLMClient').setLevel(logging.WARNING)     # Was INFO/DEBUG
    # Reduce component init noise by default (unless overall level is DEBUG)
    logging.getLogger('autonomous_debugger.init').setLevel(logging.INFO if level <= logging.INFO else logging.DEBUG)
    logging.getLogger('CodeExecutor').setLevel(logging.INFO if level <= logging.INFO else logging.DEBUG) # Allow INFO for execution start/end
    logging.getLogger('RealityChecker').setLevel(logging.INFO if level <= logging.INFO else logging.DEBUG)
    logging.getLogger('CodebaseManager').setLevel(logging.INFO if level <= logging.INFO else logging.DEBUG)
    logging.getLogger('CodingAgent').setLevel(logging.INFO if level <= logging.INFO else logging.DEBUG)
    logging.getLogger('SystemValidator').setLevel(logging.INFO if level <= logging.INFO else logging.DEBUG)
    logging.getLogger('ProcessManager').setLevel(logging.INFO if level <= logging.INFO else logging.DEBUG)
    logging.getLogger('SubgoalAgent').setLevel(logging.INFO if level <= logging.INFO else logging.DEBUG)


    if GOOGLE_GENAI_AVAILABLE:
        logging.getLogger('google.api_core').setLevel(logging.WARNING)
        logging.getLogger('google.generativeai').setLevel(logging.WARNING) # Was INFO/WARNING

    logger = logging.getLogger('autonomous_debugger'); # Get main logger again after setup
    logger.info(f"--- Logging Configured (Overall Level: {logging.getLevelName(level)}) ---")
    print(f"--- Logging Configured (Overall Level: {logging.getLevelName(level)}) ---")


class UnifiedLogger:
    _instance = None
    def __new__(cls, log_file: str = 'unified_output_log.json', max_entries: int = 100):
        if cls._instance is None:
            cls._instance = super(UnifiedLogger, cls).__new__(cls)
            cls._instance.log_file = log_file; cls._instance.max_entries = max_entries
            cls._instance.actions = []; cls._instance.logger = logging.getLogger(f"{__name__}.{cls.__name__}")
            # Log initialization only once
            if not getattr(cls._instance, '_initialized', False):
                 cls._instance.logger.info(f"UnifiedLogger initialized: File='{cls._instance.log_file}', MaxEntries={cls._instance.max_entries}")
                 cls._instance._initialized = True # Mark as initialized
        return cls._instance
    def log_action(self, component: str, content: str, context: Any = ""):
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try: ctx_str = json.dumps(context)
        except TypeError: ctx_str = str(context)
        act = {"timestamp": ts, "component": component, "content": content, "context": ctx_str}
        self.actions.append(act)
        if len(self.actions) > self.max_entries: self.actions = self.actions[-self.max_entries:]
    def get_logs(self) -> List[Dict[str, Any]]: return self.actions[:]
    def save_log(self):
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f: json.dump(self.actions, f, indent=2)
            self.logger.info(f"Unified log saved to {self.log_file}")
        except Exception as e: self.logger.error(f"Failed to save unified log: {e}", exc_info=True)


# --- Data Classes (Section #2) ---
@dataclass
class ValidationResult:
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None        


async def run_debugger_loop(components: Dict[str, Any]) -> bool:
    """Runs the main debugger process using the MainExecutor."""
    logger_run = logging.getLogger('autonomous_debugger.run'); logger_run.info("Starting main asynchronous execution process..."); print("\n[Execution] Starting main debugger loop...")
    main_executor = components.get("main_executor")
    if not main_executor: logger_run.critical("MainExecutor missing."); return False
    try: success = await main_executor.run_process(); return success
    except Exception as e_run: logger_run.critical(f"Critical error during main loop: {e_run}", exc_info=True); print(f"\n[CRITICAL RUNTIME ERROR] {type(e_run).__name__}: {e_run}"); traceback.print_exc(); return False

# --- Corrected cleanup_main Function ---
def cleanup_main(components: Dict[str, Any], runtime_started: bool, total_runtime_str: Optional[str]):
    """Handles final cleanup and logging of stats."""
    print("\n[Shutdown] Starting cleanup procedures..."); logger_cleanup = logging.getLogger('autonomous_debugger.cleanup'); logger_cleanup.info("Entering final cleanup phase.")
    runtime_counter = components.get("runtime_counter"); runtime_metrics = components.get("runtime_metrics")
    debug_session = components.get("debug_session"); unified_logger = components.get("unified_logger"); code_executor = components.get("code_executor")
    if runtime_counter and runtime_started: print(f"[Info] Total Execution Runtime: {total_runtime_str or 'N/A'}")
    else: print("[Info] Runtime counter was not started or available.")
    if runtime_metrics and PSUTIL_AVAILABLE:
        try: # Correctly indented try block
            avg = runtime_metrics.get_avg()
            print(f"[Info] Average Resource Usage: Memory={avg.get('avg_mem', 0):.1f}%, CPU={avg.get('avg_cpu', 0):.1f}%")
        except Exception as e:
            logger_cleanup.error(f"Failed to get runtime metrics during shutdown: {e}")
    if debug_session and isinstance(debug_session, DebugSession):
        # Correctly indented try block
        try:
            debug_session.log_stats()
        except Exception as e:
            logger_cleanup.error(f"Error logging final stats: {e}")
    else:
        logger_cleanup.warning("DebugSession not available for final stats logging.")
    if unified_logger and isinstance(unified_logger, UnifiedLogger):
         try: unified_logger.save_log()
         except Exception as e: logger_cleanup.error(f"Error saving final unified log: {e}")
    else: logger_cleanup.warning("UnifiedLogger not available for saving log.")
    try: prefix = getattr(code_executor, 'prefix', 'temp_code_') if code_executor else 'temp_code_'; cleanup_resources(temp_file_prefix=prefix)
    except Exception as e: logger_cleanup.error(f"Error during resource cleanup: {e}", exc_info=True)
# --- End cleanup_main ---

async def main() -> bool:
    """Main async entry point, orchestrates setup, run, and cleanup."""
    components = {}; runtime_counter = None; runtime_started = False; final_success_status = False; total_runtime_str = "N/A"; loop = None
    try:
        # Load config and set up logging first
        config = load_config()
        configure_logging_main(config.log_level)

        # REMOVED: The setup_api call is no longer needed here,
        # as ApiKeyManager is initialized inside initialize_components
        # and LLMClient configures genai per-call.
        # if not setup_api(config): raise InitializationError("API Setup failed.")

        # Initialize all components, including ApiKeyManager and LLMClient
        components = initialize_components(config)
        runtime_counter = components.get("runtime_counter")

        # Check for essential components after initialization
        # (Moved check here as components dict is now populated)
        main_executor = components.get("main_executor") # Example essential check
        if not main_executor:
             # NOTE: InitializationError class IS defined globally now
             raise InitializationError(f"MainExecutor missing after component initialization.")

        # Start runtime counter if available
        if runtime_counter:
            runtime_counter.start()
            runtime_started = True

        # Run the main debugger loop
        final_success_status = await run_debugger_loop(components)

    except InitializationError as e_init_main:
        # Log initialization errors specifically
        logger.critical(f"Initialization Error in main: {e_init_main}", exc_info=True)
        final_success_status = False
        # Optionally print if logging might not be fully set up
        print(f"\n[Setup Error] Initialization failed: {e_init_main}")
        traceback.print_exc()

    except Exception as e_main_async:
        # Catch any other critical errors during the main async execution
        logger.critical(f"Critical error during main async execution: {e_main_async}", exc_info=True)
        print(f"\n[CRITICAL ERROR] {type(e_main_async).__name__}: {e_main_async}")
        traceback.print_exc()
        final_success_status = False

    finally:
        # --- Cleanup Phase ---
        # Stop runtime counter before logging stats
        if runtime_counter and runtime_started:
            total_runtime_str = runtime_counter.stop()

        # Call the helper function for detailed cleanup and stats logging
        cleanup_main(components, runtime_started, total_runtime_str)

        # --- Event Loop Cleanup (Revised - No explicit close) ---
        # This logic attempts graceful shutdown of remaining tasks and generators
        try:
            # Check if a loop is running *without* creating one if none exists
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            loop = None # No loop was set or running

        if loop and not loop.is_closed():
            logger.info("Attempting graceful asyncio cleanup...")
            try:
                running = loop.is_running()
                all_tasks = list(asyncio.all_tasks(loop))
                # Avoid cancelling the task running 'main' itself if loop is running
                current = asyncio.current_task(loop) if running else None
                tasks_to_cancel = [t for t in all_tasks if t is not current]

                if tasks_to_cancel:
                    logger.info(f"Cancelling {len(tasks_to_cancel)} outstanding tasks...")
                    for task in tasks_to_cancel:
                        task.cancel()
                    try:
                        # Wait for cancellations to process
                        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
                        logger.info("Outstanding tasks cancellation processed.")
                    except RuntimeError as e_gather: # Handle case where loop stops mid-gather
                         logger.warning(f"RuntimeError during task gathering on shutdown (may be ok if loop stopped): {e_gather}")
                    except asyncio.CancelledError:
                         logger.warning("Gather itself was cancelled during shutdown.")
                else:
                    logger.info("No outstanding tasks to cancel.")

                # Shutdown async generators (best effort)
                if running: # Only attempt if loop indicates it's still running
                     try:
                         await loop.shutdown_asyncgens()
                         logger.info("Async generators shut down.")
                     except RuntimeError as e_sh: # Can happen if loop stops unexpectedly
                          logger.warning(f"Could not shutdown async generators (loop might have stopped): {e_sh}")
                     except Exception as e_gens:
                          logger.error(f"Error shutting down async generators: {e_gens}")

                # *** Explicit loop.close() REMOVED ***
                # asyncio.run() handles the final closing of the loop it creates.
                logger.info(f"Event loop cleanup finished (was_running={running}). Letting asyncio.run manage final closing.")

            except Exception as e_loop_close:
                logger.error(f"Error during event loop cleanup steps: {e_loop_close}", exc_info=True)
        elif loop and loop.is_closed():
             logger.info("Event loop was already closed before final cleanup check.")
        else:
             logger.info("No active event loop found during shutdown check.")
        # --- End Event Loop Cleanup ---

        # Log final status banner consistently
        final_status_msg = 'Success' if final_success_status else 'Failed/Incomplete'
        log_msg = f"--- Autonomous Debugger Shutdown --- Final Status: {final_status_msg} ---"
        print_msg = (f"\n{'='*70}\n"
                     f"{f' Process Finished: {final_status_msg} '.center(70, '=')}\n"
                     f"{f' Total Time: {total_runtime_str} '.center(70)}\n"
                     f"{'='*70}\n"
                     f"\n[Shutdown] Complete. Final Status: {final_status_msg}\n"
                     f"--- SCRIPT END ---")

        logger.info(log_msg)
        print(print_msg)

    # Return the final success status for the exit code
    return final_success_status
# --- End main ---
# 
# 
# ntry Point ---
if __name__ == "__main__":
    final_status = False
    e_top_level = None # Define outside try
    try:
        # asyncio.run handles loop creation, running main(), and closing the loop
        final_status = asyncio.run(main())
    except Exception as top_level_exception:
        e_top_level = top_level_exception # Assign if exception occurs
        print(f"\n[TOP LEVEL CRITICAL ERROR] {type(e_top_level).__name__}: {e_top_level}")
        traceback.print_exc()
        try: # Try logging
            if 'logger' in globals() and logger: # Check logger exists
                 logger.critical(f"Top-level error: {e_top_level}", exc_info=True)
            else:
                 print(f"FATAL (No logger): Top-level error: {e_top_level}")
        except Exception as log_err:
             print(f"FATAL: Error occurred during critical error logging: {log_err}")
             if e_top_level: print(f"Original top-level error was: {e_top_level}")
        final_status = False # Ensure failure state

    # Exit with appropriate status code
    sys.exit(0 if final_status else 1)
# --- End __main__ --- End main ---
import os
from dotenv import load_dotenv
from langchain_community.chat_models import AzureChatOpenAI

def _pick_env(suffix: str | None):
    """Return a dict of Azure OpenAI creds for the given suffix.
    suffix=None -> base env (gpt-5-mini), '1' -> 4.1, '2' -> 4.0
    """
    load_dotenv()
    suf = "" if suffix is None else str(suffix)
    return {
        "deployment": os.getenv(f"AZURE_OPENAI_DEPLOYMENT_NAME{suf}"),
        "api_key": os.getenv(f"AZURE_OPENAI_API_KEY{suf}"),
        "endpoint": os.getenv(f"AZURE_OPENAI_ENDPOINT{suf}"),
        "api_version": os.getenv(f"AZURE_OPENAI_API_VERSION{suf}"),
    }

def get_llm(model_choice: str):
    """
    model_choice ∈ {'gpt-5-mini','gpt-4.1','gpt-4.0'}
    Routes to the right env var set and temperature.
    """
    if model_choice == "gpt-5-mini":
        env = _pick_env(None)
        temperature = 1.0
    elif model_choice == "gpt-4.1":
        env = _pick_env("1")
        temperature = 0.7
    elif model_choice == "gpt-4.0":
        env = _pick_env("2")
        temperature = 0.2
    else:
        raise ValueError(f"Unknown model_choice: {model_choice}")

    if not all(env.values()):
        missing = [k for k, v in env.items() if not v]
        raise RuntimeError(f"Missing Azure env vars for {model_choice}: {missing}")

    return AzureChatOpenAI(
        azure_deployment=env["deployment"],
        api_key=env["api_key"],
        azure_endpoint=env["endpoint"],
        api_version=env["api_version"],
        temperature=temperature,
    )

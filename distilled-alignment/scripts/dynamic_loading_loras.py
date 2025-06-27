def load_vllm_lora_adapter(adapter_hf_name: str, vllm_port: int = 8000):
    """
    Loads the vLLM LORA adapter by sending a POST request.

    Args:
        adapter_hf_name: Name of the adapter to load
        vllm_port: Port where vLLM server is running (default: 8000)

    Raises:
        requests.exceptions.RequestException: If the request fails or returns non-200 status,
            except for the case where the adapter is already loaded
    """
    # Check if the model is already loaded and return early if so
    response = requests.get(f"http://localhost:{vllm_port}/v1/models", timeout=10 * 60)
    response.raise_for_status()
    if adapter_hf_name in [model["id"] for model in response.json().get("data", [])]:
        print(f"LORA adapter {adapter_hf_name} is already loaded")
        return

    try:
        print(f"Loading LORA adapter {adapter_hf_name} on port {vllm_port}...")
        response = requests.post(
            f"http://localhost:{vllm_port}/v1/load_lora_adapter",
            json={"lora_name": adapter_hf_name, "lora_path": adapter_hf_name},
            headers={"Content-Type": "application/json"},
            timeout=20 * 60,  # Wait up to 20 minutes for response, loading can be slow if many in parallel
        )

        # If we get a 400 error about adapter already being loaded, that's fine
        if (
            response.status_code == 400
            and "has already been loaded" in response.json().get("message", "")
        ):
            print("LORA adapter was already loaded")
            return

        # For all other cases, raise any errors
        response.raise_for_status()
        print("LORA adapter loaded successfully!")

    except requests.exceptions.RequestException as e:
        if "has already been loaded" in str(e):
            print("LORA adapter was already loaded")
            return
        raise
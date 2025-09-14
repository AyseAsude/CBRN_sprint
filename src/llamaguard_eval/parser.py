def parse_output(raw_output: str):
    """
    Parse Llama Guard raw text output into is_safe (True/False).
    """
    raw_lower = raw_output.lower()
    is_safe = not ("unsafe" in raw_lower)
    return is_safe
 

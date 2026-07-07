#!/usr/bin/env python3
"""Fix sglang v0.5.9 gpt-oss tool-call parsing (run inside the official image
before launch_server; idempotent).

Two stock-parser bugs make it silently swallow tool calls (no content, no
tool_calls, only usage tokens) — SWE-agent then dies with "repeated format
errors":

1. `tool_extract_pattern` requires `<|constrain|>json` between `to=NAME` and
   `<|message|>`, but gpt-oss-120b frequently omits or varies it:
       <|channel|>commentary to=functions.bash<|message|>{...}<|call|>
       <|channel|>analysis to=functions.bash code<|message|>{...}<|call|>
2. Tool calls emitted on the *analysis* channel
   (`<|channel|>analysis to=functions.X ...<|call|>`) are not recognised by
   the detector's channel gates, and the partial-analysis streamer leaks the
   call's JSON args into reasoning_content.
3. Malformed JSON arguments cause the whole tool call to be dropped
   (client sees an empty message). OpenAI-compatible behaviour is to forward
   the arguments string verbatim and let the client handle repair/retry.

Verified against sglang v0.5.9 (identical parser code in v0.5.12.post1).
"""
import re
import sys

DET = "/sgl-workspace/sglang/python/sglang/srt/function_call/gpt_oss_detector.py"
HAR = "/sgl-workspace/sglang/python/sglang/srt/parser/harmony_parser.py"
MARK = "# AGENTCAP_GPTOSS_PATCH"


def patch_file(path, replacements):
    with open(path) as f:
        text = f.read()
    if MARK in text:
        print(f"already patched: {path}")
        return
    for old, new in replacements:
        if old not in text:
            print(f"ERROR: pattern not found in {path}:\n{old}", file=sys.stderr)
            sys.exit(2)
        text = text.replace(old, new, 1)
    text = f"{MARK}\n" + text
    with open(path, "w") as f:
        f.write(text)
    print(f"patched: {path}")


patch_file(DET, [
    # 1. constrain marker is optional and varies (json / code / absent)
    (
        '            r"to=([a-zA-Z_][a-zA-Z0-9_.-]*)\\s*<\\|constrain\\|>json<\\|message\\|>(.*?)(?:<\\|call\\|>|$)",',
        '            r"to=([a-zA-Z_][a-zA-Z0-9_.-]*)\\s*(?:<\\|constrain\\|>)?\\s*\\w*\\s*<\\|message\\|>(.*?)(?:<\\|call\\|>|$)",',
    ),
    # 2a. detect tool calls on the analysis channel too (non-streaming gate)
    (
        '        return self.bot_token in text',
        '        return self.bot_token in text or "<|channel|>analysis to=" in text',
    ),
    # 2b. same for the streaming quick-check
    (
        '            "<|channel|>commentary to=" not in self._buffer\n'
        '            and not self.current_tool_name_sent\n',
        '            "<|channel|>commentary to=" not in self._buffer\n'
        '            and "<|channel|>analysis to=" not in self._buffer\n'
        '            and not self.current_tool_name_sent\n',
    ),
    # 3a. repair common gpt-oss JSON quirks (invalid \' escapes, literal
    #     control chars) instead of dropping the call; forward raw as a
    #     last resort so the client can handle it
    (
        '        try:\n'
        '            arguments = json.loads(json_content) if json_content.strip() else {}\n'
        '        except json.JSONDecodeError as e:\n'
        '            logger.debug(f"Failed to parse JSON arguments: {e}")\n'
        '            return None\n',
        '        try:\n'
        '            arguments = json.loads(json_content) if json_content.strip() else {}\n'
        '        except json.JSONDecodeError as e:\n'
        '            arguments = None\n'
        '            for _cand in (json_content, json_content.replace("\\\\\'", "\'")):\n'
        '                try:\n'
        '                    arguments = json.loads(_cand, strict=False)\n'
        '                    break\n'
        '                except json.JSONDecodeError:\n'
        '                    continue\n'
        '            if arguments is None:\n'
        '                logger.debug(f"Failed to parse JSON arguments: {e}")\n'
        '                return ToolCallItem(\n'
        '                    tool_index=tool_index,\n'
        '                    name=function_name,\n'
        '                    parameters=json_content,\n'
        '                )\n',
    ),
    # 3b. guard the streaming loop against non-JSON arguments
    (
        '                    self.prev_tool_call_arr[self.current_tool_id] = {\n'
        '                        "name": tool_call_info.name,\n'
        '                        "arguments": json.loads(tool_call_info.parameters),\n'
        '                    }\n',
        '                    try:\n'
        '                        _args = json.loads(tool_call_info.parameters)\n'
        '                    except json.JSONDecodeError:\n'
        '                        _args = tool_call_info.parameters\n'
        '                    self.prev_tool_call_arr[self.current_tool_id] = {\n'
        '                        "name": tool_call_info.name,\n'
        '                        "arguments": _args,\n'
        '                    }\n',
    ),
])

patch_file(HAR, [
    # 2c. don't stream partial analysis blocks that are tool calls
    #     (wait for the complete block so it becomes a tool_call event)
    (
        '        channel_type = self._extract_channel_type(channel_header)\n'
        '        if channel_type != "analysis":\n'
        '            return None  # Only stream analysis content - tool calls wait for completion\n',
        '        channel_type = self._extract_channel_type(channel_header)\n'
        '        if channel_type != "analysis":\n'
        '            return None  # Only stream analysis content - tool calls wait for completion\n'
        '        if "to=" in channel_header:\n'
        '            return None  # tool call on analysis channel - wait for complete block\n',
    ),
])

print("sglang gpt-oss parser patch applied OK")

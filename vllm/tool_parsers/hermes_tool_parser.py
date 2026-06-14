# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.utils.mistral import is_mistral_tokenizer

logger = init_logger(__name__)


def _partial_tag_overlap(text: str, tag: str) -> int:
    """Length of the longest prefix of `tag` that matches a suffix of `text`.

    E.g. text ending in "<tool_" returns 6 when tag is "<tool_call>".
    Returns 0 if there is no overlap.
    """
    max_check = min(len(tag) - 1, len(text))
    for k in range(max_check, 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


class Hermes2ProToolParser(ToolParser):
    # When using structural_tag for guided decoding, the model outputs
    # Hermes-format JSON like {"name": "...", "arguments": {...}} wrapped
    # in <tool_call_start>/<tool_call_end> tokens. The serving layer's
    # "required" branch expects FunctionDefinition format
    # [{"name": "...", "parameters": {...}}] which is incompatible.
    # Set False so that "required" and named tool_choice also go through
    # extract_tool_calls(), which correctly parses the Hermes format.
    supports_required_and_named: bool = False

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        if is_mistral_tokenizer(tokenizer):
            logger.error("Detected Mistral tokenizer when using a Hermes model")
            self.model_tokenizer = tokenizer.tokenizer

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"

        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
        )
        self.scratch_pad_regex = re.compile(
            r"<scratch_pad>(.*?)</scratch_pad>", re.DOTALL
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        # Streaming state: what has been sent to the client.
        self._sent_content_idx: int = 0

    def get_structural_tag(self, request):
        """Return structural tag for hermes-style tool calling.

        This enables guided decoding to properly handle the tool_call_end
        token, which plain JSON-schema guided decoding would block,
        causing the model to loop indefinitely.
        """
        from vllm.tool_parsers.structural_tag_registry import (
            _get_function_parameters,
        )
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionNamedToolChoiceParam,
        )
        from xgrammar import StructuralTag
        from xgrammar.structural_tag import (
            AnyTextFormat,
            ConstStringFormat,
            JSONSchemaFormat,
            SequenceFormat,
            TagFormat,
            TagsWithSeparatorFormat,
            TriggeredTagsFormat,
        )

        # Hermes tool call format:
        #   <tool_call_start>\n{"name": ..., "arguments": ...}\n<tool_call_end>
        start_tok = self.tool_call_start_token
        end_tok = self.tool_call_end_token
        tool_call_begin = start_tok + "\n"
        tool_call_end_str = "\n" + end_tok
        tool_call_trigger = start_tok
        think_tag_end = "<|im_end|>"
        think_exclude_tokens = ["<|think|>", "<|/think|>"]

        # Detect reasoning from the request's thinking configuration
        reasoning = False
        if hasattr(request, 'thinking') and request.thinking:
            reasoning = True

        # Normalize tool_choice
        tool_choice = request.tool_choice
        if tool_choice is None or tool_choice == "none":
            return None
        elif tool_choice == "auto":
            simplified_choice = "auto"
        elif tool_choice == "required":
            simplified_choice = "required"
        else:
            simplified_choice = "forced"

        tools = request.tools
        if not tools:
            return None

        if simplified_choice == "auto":
            tags = []
            for tool in tools:
                function = tool.function
                parameters = _get_function_parameters(function)
                tags.append(
                    TagFormat(
                        begin=tool_call_begin + '{"name": "' + function.name + '", "arguments": ',
                        content=JSONSchemaFormat(
                            json_schema=parameters,
                            style="json",
                        ),
                        end='}' + tool_call_end_str,
                    )
                )

            if tags:
                suffix_tag = TriggeredTagsFormat(
                    triggers=[tool_call_trigger],
                    tags=tags,
                    excludes=think_exclude_tokens,
                )
            else:
                suffix_tag = AnyTextFormat(excludes=think_exclude_tokens)

        elif simplified_choice == "forced":
            # When tool_choice is a named function, filter tools to the
            # specified one (matching _normalize_tool_choice behavior).
            if isinstance(request.tool_choice,
                          ChatCompletionNamedToolChoiceParam):
                tool_name = request.tool_choice.function.name
                filtered_tools = [
                    tool for tool in tools
                    if tool.function.name == tool_name
                ]
                if not filtered_tools:
                    raise ValueError(
                        f"The tool with name '{tool_name}' is not found "
                        "in the tools list."
                    )
                forced_tool = filtered_tools[0]
            else:
                forced_tool = tools[0]
            function = forced_tool.function
            parameters = _get_function_parameters(function)
            suffix_tag = TagFormat(
                begin=tool_call_begin + '{"name": "' + function.name + '", "arguments": ',
                content=JSONSchemaFormat(
                    json_schema=parameters,
                    style="json",
                ),
                end='}' + tool_call_end_str,
            )

        elif simplified_choice == "required":
            tags = []
            for tool in tools:
                function = tool.function
                parameters = _get_function_parameters(function)
                tags.append(
                    TagFormat(
                        begin=tool_call_begin + '{"name": "' + function.name + '", "arguments": ',
                        content=JSONSchemaFormat(
                            json_schema=parameters,
                            style="json",
                        ),
                        end='}' + tool_call_end_str,
                    )
                )
            assert len(tags) > 0
            suffix_tag = TagsWithSeparatorFormat(
                tags=tags,
                separator="",
                at_least_one=True,
            )

        if not reasoning:
            return StructuralTag(format=suffix_tag)

        prefix_tag = SequenceFormat(
            elements=[
                TagFormat(begin="", content=AnyTextFormat(), end=think_tag_end),
                ConstStringFormat(value="\n\n"),
            ]
        )
        return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        if request.tools and request.tool_choice != "none":
            # do not skip special tokens because the tool_call tokens are
            # marked "special" in some models. Since they are skipped
            # prior to the call to the tool parser, it breaks tool calling.
            request.skip_special_tokens = False

            # Use structural_tag for guided decoding first.
            # This properly handles the tool_call_end token, which plain
            # JSON-schema guided decoding would block, causing the model
            # to loop indefinitely after generating the tool call JSON.
            if isinstance(request, ChatCompletionRequest):
                structural_tag = self.get_structural_tag(request)
                if structural_tag is not None:
                    import json as _json
                    from vllm.sampling_params import StructuredOutputsParams
                    if request.structured_outputs is None:
                        request.structured_outputs = StructuredOutputsParams(
                            structural_tag=_json.dumps(
                                structural_tag.model_dump()),
                        )
                    else:
                        request.structured_outputs.structural_tag = _json.dumps(
                            structural_tag.model_dump()
                        )
                    # Clear the json schema to avoid conflict
                    if (request.structured_outputs.json is not None
                            and request.structured_outputs.structural_tag
                            is not None):
                        request.structured_outputs.json = None

            # Add the tool_call_end_token to stop_token_ids so that
            # generation stops after a complete tool call.
            tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
            if (tool_call_end_token_id is not None
                    and tool_call_end_token_id not in
                    (request.stop_token_ids or [])):
                request.stop_token_ids = request.stop_token_ids or []
                request.stop_token_ids.append(tool_call_end_token_id)

            # Skip super().adjust_request() when structural_tag is set,
            # to avoid it being overridden by plain JSON schema.
            if (isinstance(request, ChatCompletionRequest)
                    and request.structured_outputs is not None
                    and request.structured_outputs.structural_tag is not None):
                return request

        request = super().adjust_request(request)
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # sanity check; avoid unnecessary processing
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        else:
            try:
                # there are two possible captures - between tags, or between a
                # tag and end-of-string so the result of
                # findall is an array of tuples where one is a function call and
                # the other is None
                function_call_tuples = self.tool_call_regex.findall(model_output)

                # load the JSON, and then use it to build the Function and
                # Tool Call
                raw_function_calls = [
                    json.loads(match[0] if match[0] else match[1])
                    for match in function_call_tuples
                ]
                tool_calls = [
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=function_call["name"],
                            # function call args are JSON but as a string
                            arguments=json.dumps(
                                function_call["arguments"], ensure_ascii=False
                            ),
                        ),
                    )
                    for function_call in raw_function_calls
                ]

                content = model_output[: model_output.find(self.tool_call_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            except Exception:
                logger.exception("Error in extracting tool call from response.")
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

    def _extract_content(self, current_text: str) -> str | None:
        """Return unsent non-tool-call text, or None.

        Holds back any suffix that could be a partial <tool_call> tag.
        """
        if self.tool_call_start_token not in current_text:
            overlap_length = _partial_tag_overlap(
                current_text, self.tool_call_start_token
            )
            sendable_idx = len(current_text) - overlap_length
        else:
            sendable_idx = current_text.index(self.tool_call_start_token)

        if sendable_idx > self._sent_content_idx:
            content = current_text[self._sent_content_idx : sendable_idx]
            self._sent_content_idx = sendable_idx
            return content
        return None

    def _extract_tool_call_jsons(self, text: str) -> list[tuple[str, bool]]:
        """Extract (json_text, is_complete) for each <tool_call> region."""
        results: list[tuple[str, bool]] = []
        pos = 0
        while True:
            start = text.find(self.tool_call_start_token, pos)
            if start == -1:
                break
            json_start = start + len(self.tool_call_start_token)
            json_end = text.find(self.tool_call_end_token, json_start)
            if json_end != -1:
                results.append((text[json_start:json_end].strip(), True))
                pos = json_end + len(self.tool_call_end_token)
            else:
                raw = text[json_start:]
                # Strip partial </tool_call> suffix if present.
                overlap = _partial_tag_overlap(raw, self.tool_call_end_token)
                if overlap:
                    raw = raw[:-overlap]
                tc_json = raw.strip()
                # Valid JSON without closing tag = complete body,
                # tag tokens just haven't arrived yet.
                is_complete = _is_valid_json(tc_json) if tc_json else False
                results.append((tc_json, is_complete))
                break
        return results

    @staticmethod
    def _extract_tool_name(tc_json: str) -> str | None:
        """Extract tool name, or None if the name isn't complete yet."""
        match = re.search(r'"name"\s*:\s*"([^"]+)"', tc_json)
        return match.group(1) if match else None

    @staticmethod
    def _extract_tool_args(tc_json: str, is_complete: bool) -> str | None:
        """Extract tool arguments from the tool call JSON.

        Given {"name": "f", "arguments": {"x": 1}}, returns '{"x": 1}'.
        When is_complete, strips the trailing '}' that closes the outer
        object (not the arguments). For partial JSON, returns as-is.
        """
        match = re.search(r'"arguments"\s*:\s*', tc_json)
        if not match:
            return None
        raw = tc_json[match.end() :]
        if is_complete:
            raw = raw.rstrip()
            if raw.endswith("}"):
                raw = raw[:-1].rstrip()
        return raw

    def _compute_args_diff(
        self, index: int, tc_json: str, is_complete: bool
    ) -> str | None:
        """Return new argument text not yet sent for tool `index`, or None."""
        args = self._extract_tool_args(tc_json, is_complete)
        if args is None or len(args) <= len(self.streamed_args_for_tool[index]):
            return None
        diff = args[len(self.streamed_args_for_tool[index]) :]
        self.streamed_args_for_tool[index] = args
        self.prev_tool_call_arr[index]["arguments"] = args
        return diff

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Incrementally stream tool call deltas from accumulated output.

        On each invocation, re-parses the full ``current_text`` to find
        ``<tool_call>`` regions, then diffs against previously sent state
        to emit only new content, tool names, or argument fragments.

        Returns a ``DeltaMessage`` containing either plain content (for
        text preceding any tool call) or one or more ``DeltaToolCall``
        entries, or ``None`` if there is nothing new to send yet."""
        try:
            # Extract any content before tool calls.
            content = self._extract_content(current_text)
            tool_call_jsons = self._extract_tool_call_jsons(current_text)
            tool_call_deltas: list[DeltaToolCall] = []

            for i, (tc_json, is_complete) in enumerate(tool_call_jsons):
                if i >= len(self.prev_tool_call_arr):
                    self.prev_tool_call_arr.append({})
                    self.streamed_args_for_tool.append("")

                # Stream back tool name.
                if "name" not in self.prev_tool_call_arr[i]:
                    name = self._extract_tool_name(tc_json)
                    if not name:
                        # Can't skip to tool i+1 if i isn't ready
                        break
                    self.prev_tool_call_arr[i]["name"] = name
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=i,
                            type="function",
                            id=make_tool_call_id(),
                            function=DeltaFunctionCall(name=name).model_dump(
                                exclude_none=True
                            ),
                        )
                    )

                # Stream back new tool args by diffing against what was sent.
                args_diff = self._compute_args_diff(i, tc_json, is_complete)
                if args_diff:
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=i,
                            function=DeltaFunctionCall(arguments=args_diff).model_dump(
                                exclude_none=True
                            ),
                        )
                    )

            if content or tool_call_deltas:
                return DeltaMessage(
                    content=content,
                    tool_calls=tool_call_deltas,
                )

            return None

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None

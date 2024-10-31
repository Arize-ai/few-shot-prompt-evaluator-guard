# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Optional

from langchain_core.language_models.llms import BaseLLM

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import ActionResult, action
from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.llm.params import llm_params
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.llm.types import Task
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.utils import new_event_dict
from opentelemetry import trace

tracer = trace.get_tracer(__name__)
log = logging.getLogger(__name__)


@action(is_system_action=True)
async def self_check_input(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    llm: Optional[BaseLLM] = None,
    config: Optional[RailsConfig] = None,
):
    """Checks the input from the user.

    Prompt the LLM, using the `check_input` task prompt, to determine if the input
    from the user should be allowed or not.

    Returns:
        True if the input should be allowed, False otherwise.
    """
    with tracer.start_as_current_span("nemo_guardrails_self_check_output") as span:

        _MAX_TOKENS = 3
        user_input = context.get("user_message")
        task = Task.SELF_CHECK_INPUT
        span.set_attribute("user_input", user_input)

        if user_input:
            prompt = llm_task_manager.render_task_prompt(
                task=task,
                context={
                    "user_input": user_input,
                },
            )

            # Initialize the LLMCallInfo object
            llm_call_info_var.set(LLMCallInfo(task=task.value))

            response = await llm_call(llm, prompt)

            log.warning(f"Input self-checking result is: `{response}`.")

            is_safe = False if "yes" in response.lower() else True
            span.set_attribute("is_safe", is_safe)
            span.set_attribute("explanation", response)

            return is_safe

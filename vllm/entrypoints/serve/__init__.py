# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import FastAPI

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def register_vllm_serve_api_routers(app: FastAPI):
    if envs.VLLM_SERVER_DEV_MODE:
        logger.warning(
            "SECURITY WARNING: Development endpoints are enabled! "
            "This should NOT be used in production!"
        )

    from vllm.entrypoints.serve.lora.api_router import (
        attach_router as attach_lora_router,
    )

    attach_lora_router(app)

    from vllm.entrypoints.serve.profile.api_router import (
        attach_router as attach_profile_router,
    )

    attach_profile_router(app)

    from vllm.entrypoints.serve.sleep.api_router import (
        attach_router as attach_sleep_router,
    )

    attach_sleep_router(app)

    from vllm.entrypoints.serve.rpc.api_router import (
        attach_router as attach_rpc_router,
    )

    attach_rpc_router(app)

    from vllm.entrypoints.serve.cache.api_router import (
        attach_router as attach_cache_router,
    )

    attach_cache_router(app)

    from vllm.entrypoints.serve.tokenize.api_router import (
        attach_router as attach_tokenize_router,
    )

    attach_tokenize_router(app)

    from .instrumentator import register_instrumentator_api_routers

    register_instrumentator_api_routers(app)


def register_vllm_community_routers(app: FastAPI):
    """Hook for third-party/community plugins to attach additional routers.

    The default implementation is a no-op so the vLLM core stays unchanged
    for users that do not register any plugin against this hook. Plugins
    (for example ``vllm_houmo``) can monkey-patch this symbol at import
    time to attach their own routers without forking vLLM.
    """
    return None


async def init_vllm_community_app_state(engine_client, state, args):
    """Hook for third-party/community plugins to populate app state.

    The default implementation is a no-op. Plugins can monkey-patch this
    symbol at import time to perform additional initialisation steps
    (e.g. lazy resource allocation) once vLLM's core ``init_app_state``
    has finished.
    """
    return None

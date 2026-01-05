from fastapi import APIRouter  # FastAPI router object for grouping endpoints
from app.core.config import settings  # Centralized config (Supabase table/function + default OpenAI model)

# Create a router for health endpoints.
# tags=["health"] groups it in Swagger UI under a "health" section.
router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    # -------------------------------------------------------------------------
    # PURPOSE OF THIS ENDPOINT
    # -------------------------------------------------------------------------
    # This is a lightweight "service is alive" endpoint.
    # It lets you (or a load balancer / uptime monitor / deployment pipeline) quickly verify:
    # - the API server is running
    # - configuration values are being loaded correctly (env vars -> settings)
    #
    # Why it’s important:
    # 1) Debugging deployments:
    #    If /health works, then FastAPI + routing is up.
    #    If /health shows the wrong table/function/model, your environment variables/settings
    #    are misconfigured (common when deploying to EC2/Docker).
    #
    # 2) Operational monitoring:
    #    Many systems (Kubernetes, AWS ALB health checks, uptime monitors) ping a health endpoint
    #    to decide whether an instance is healthy and should receive traffic.
    #
    # 3) Quick confirmation of RAG wiring:
    #    Returning supabase_table + match_function tells you which backend storage/search
    #    configuration this server thinks it is using.
    # -------------------------------------------------------------------------

    return {
        # Simple liveness indicator (if you get a 200 response with this JSON, the server is up).
        "status": "ok",

        # Which Supabase table you are configured to store chunks/embeddings in.
        # Useful when you have multiple environments (dev/prod) or multiple tables.
        "supabase_table": settings.supabase_table,

        # Which Supabase RPC function you are configured to use for similarity search.
        # For your SQL, this should be "match_chunks".
        "match_function": settings.supabase_match_fn,

        # Which OpenAI model your chat endpoint will use by default (unless overridden per request).
        "model": settings.openai_model,
    }

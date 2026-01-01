# supabase-rag-chatbot

https://ayeshasahar.hashnode.dev/creating-a-rag-chatbot-with-supabase-openai-python-langchain

rag_api/
  app/
    __init__.py
    main.py
    core/
      __init__.py
      config.py
      logging.py
    clients/
      __init__.py
      openai_client.py
      supabase_client.py
      embeddings.py
    services/
      __init__.py
      ingest_service.py
      retrieval_service.py
      chat_service.py
    api/
      __init__.py
      routes/
        __init__.py
        health.py
        ingest.py
        chat.py
    models/
      __init__.py
      schemas.py
  tmp/              # optional temp uploads
  .env
  requirements.txt

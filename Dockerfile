FROM python:3.11-slim

# Hugging Face Spaces і більшість PaaS запускають контейнер від НЕ-root
# користувача з uid 1000. Якщо лишити теку власністю root, сервіс упаде при
# першій спробі створити state/agent.sqlite3.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user src/ src/
COPY --chown=user web/ web/
COPY --chown=user data/ data/
COPY --chown=user eval/ eval/
COPY --chown=user main.py .

# Стан агента — похідне від прогону, в образ не копіюється.
RUN mkdir -p state output

ENV PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f\"http://localhost:{os.environ.get('PORT','7860')}/health\").status==200 else 1)"

# GEMINI_API_KEY передається через `docker run -e ...` або --env-file.
# Без ключа сервіс піднімається: черга підтверджень і перегляд трас працюють,
# приймання нових запитів віддає 503.
ENTRYPOINT ["python", "main.py", "serve"]

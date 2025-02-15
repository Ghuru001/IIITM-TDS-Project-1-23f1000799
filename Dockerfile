FROM python:3.12-slim

WORKDIR /app

COPY final_api.py ./

RUN pip install fastapi uvicorn openai python-dotenv python-dateutil scikit-learn numpy pillow markdown

EXPOSE 8000

CMD ["uvicorn", "final_api:app", "--host", "0.0.0.0", "--port", "8000"]

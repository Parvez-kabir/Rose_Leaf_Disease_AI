FROM python:3.9

WORKDIR /code

# Requirements ফাইল কপি এবং ইনস্টল করা
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# সব ফাইল কপি করা
COPY . .

# FastAPI রান করার কমান্ড
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

FROM python:3.7

WORKDIR /usr/src/app

COPY evaluate/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# VOLUME [ "/usr/src/app/logs" ]

EXPOSE 8001

CMD ["sh", "start_step2.sh"]
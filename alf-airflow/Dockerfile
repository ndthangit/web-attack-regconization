# https://airflow.apache.org/docs/docker-stack/build.html
FROM apache/airflow:3.0.6

USER root

# Cài đặt OpenJDK và các dependencies cần thiết
COPY spark-3.5.1-bin-hadoop3.tgz /tmp/spark-3.5.1-bin-hadoop3.tgz

# Cài đặt OpenJDK 17 và các dependencies cần thiết
RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Spark từ file đã tải sẵn
ENV SPARK_HOME=/opt/spark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$SPARK_HOME/bin:$JAVA_HOME/bin:$PATH

RUN tar -xzf /tmp/spark-3.5.1-bin-hadoop3.tgz -C /opt && \
    mv /opt/spark-3.5.1-bin-hadoop3 /opt/spark && \
    rm /tmp/spark-3.5.1-bin-hadoop3.tgz

# Tạo thư mục dữ liệu và cấp quyền
RUN mkdir -p /tmp/data && \
    chown -R airflow:root /tmp/data && \
    chmod -R 777 /tmp/data

# Chuyển sang user airflow (giữ nguyên GID=0)
USER airflow

USER airflow
#USER 1001

# Đảm bảo thư mục airflow có quyền phù hợp


COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt


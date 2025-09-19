FROM bitnami/spark:3.5.1
USER root

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt pip và các thư viện Python cần thiết
RUN pip3 install --no-cache-dir pyspark==3.5.1

# Tạo user mới với UID=1001 và GID=0 (root)
RUN groupadd -g 1001 sparkgroup && \
    useradd -u 1001 -g 0 -G root,sparkgroup sparkuser && \
    mkdir -p /home/sparkuser && \
    chown -R sparkuser:root /home/sparkuser

# Tạo thư mục cho ứng dụng Spark
RUN mkdir -p /opt/spark-apps && \
    chown -R sparkuser:root /opt/spark-apps && \
    chmod -R 775 /opt/spark-apps

# Tạo thư mục dữ liệu và cấp quyền
RUN mkdir -p /tmp/data && \
    chown -R sparkuser:root /tmp/data && \
    chmod -R 777 /tmp/data

# Chuyển sang user sparkuser (UID=1001, GID=0)
USER sparkuser

# Thiết lập environment variables
ENV SPARK_HOME=/opt/bitnami/spark
ENV PATH=$SPARK_HOME/bin:$PATH

COPY . /opt/spark_jobs
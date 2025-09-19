# spark_batch_job.py
import os
import glob
import shutil
import logging

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.utils import AnalysisException

import pandas as pd

# ====== Import tiền xử lý ======
# File DataProcessing.py phải nằm trong PYTHONPATH / working dir
from DataProcessing import PreTrainingLayer  # :contentReference[oaicite:2]{index=2}

# ====== Config MinIO / S3A ======
MINIO_ENDPOINT = "minio"
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minio123"

# Bucket/raw path
RAW_DATA_PATH = "s3a://aminer/"  # ví dụ: s3a://aminer/folder/*.json  :contentReference[oaicite:3]{index=3}

# Local export
LOCAL_EXPORT_DIR = "/tmp/csv_export"  # yêu cầu của bạn
os.makedirs(LOCAL_EXPORT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def list_s3a_files(spark, root_path: str) -> list[str]:
    """
    Duyệt liệt kê file trong S3A/MinIO bằng Hadoop FileSystem.
    Trả về danh sách uri đầy đủ (s3a://...).
    """
    jsc = spark._jsc
    sc = spark.sparkContext
    hadoop_conf = jsc.hadoopConfiguration()

    # Bảo đảm endpoint/cred đã set
    hadoop_conf.set("fs.s3a.endpoint", MINIO_ENDPOINT)
    hadoop_conf.set("fs.s3a.access.key", MINIO_ACCESS_KEY)
    hadoop_conf.set("fs.s3a.secret.key", MINIO_SECRET_KEY)
    hadoop_conf.set("fs.s3a.path.style.access", "true")

    uri = jsc._gateway.jvm.java.net.URI(root_path)
    path = jsc._gateway.jvm.org.apache.hadoop.fs.Path(root_path)
    fs = jsc._gateway.jvm.org.apache.hadoop.fs.FileSystem.get(uri, jsc.hadoopConfiguration())

    files = []
    # Nếu root_path là folder, duyệt tất cả entry bên trong
    for status in fs.listStatus(path):
        if status.isFile():
            files.append(status.getPath().toString())
        else:
            # duyệt đệ quy 1 cấp (nếu cần sâu hơn có thể dùng listFiles)
            it = fs.listFiles(status.getPath(), True)
            while it.hasNext():
                files.append(it.next().getPath().toString())

    # Lọc những file có phần mở rộng có thể đọc (json/csv) — tuỳ data của bạn
    cand = [f for f in files if f.lower().endswith((".json", ".ndjson", ".csv"))]
    return cand


def apply_pretrainlayer_map_in_pandas(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Nhận 1 partition dạng pandas.DataFrame, áp dụng:
        text_and_timestamps = data_raw.apply(lambda row: data_processing.format_sample_to_text(row), axis=1)
        data_raw['text_representation'] = text_and_timestamps.apply(lambda x: x[0])
        data_raw['timestamps'] = text_and_timestamps.apply(lambda x: x[1][0])
    Sau đó giữ lại thêm cột Label nếu tồn tại.
    """
    processor = PreTrainingLayer()  # :contentReference[oaicite:4]{index=4}

    # Row ở đây là pandas.Series; chuyển sang dict để giữ nguyên hành vi format_sample_to_text
    series_out = pdf.apply(lambda row: processor.format_sample_to_text(row.to_dict()), axis=1)

    texts = []
    ts_first = []
    for tup in series_out:
        # tup = (text, timestamp_value) trong đó timestamp_value có thể là list hoặc None
        text = tup[0]
        ts_val = tup[1]
        ts0 = None
        if isinstance(ts_val, (list, tuple)) and len(ts_val) > 0:
            ts0 = ts_val[0]
        elif isinstance(ts_val, (int, float)):
            ts0 = float(ts_val)
        texts.append(text)
        ts_first.append(ts0)

    out = pd.DataFrame({
        "text_representation": texts,
        "timestamp": ts_first,
    })

    # Giữ Label nếu có
    if "Label" in pdf.columns:
        out["Label"] = pdf["Label"]
    else:
        out["Label"] = None

    return out


def process_one_file(spark: SparkSession, input_uri: str):
    """
    Đọc một file từ S3A (json/ndjson/csv), áp dụng tiền xử lý, và export CSV ra /tmp/csv_export/<basename>.csv
    Ghi đè nếu tồn tại.
    """
    logging.info(f"Processing file: {input_uri}")

    # Heuristic đọc file: ưu tiên json/ndjson, fallback csv
    lower = input_uri.lower()
    if lower.endswith((".json", ".ndjson")):
        df_raw = spark.read.option("multiLine", "false").json(input_uri)
    elif lower.endswith(".csv"):
        df_raw = spark.read.option("header", "true").csv(input_uri)
    else:
        logging.warning(f"Skip unsupported file type: {input_uri}")
        return

    # Nếu không có dòng nào thì bỏ qua
    if df_raw.rdd.isEmpty():
        logging.warning(f"No data in {input_uri}, skipping.")
        return

    # Schema đầu ra cho mapInPandas
    out_schema = StructType([
        StructField("text_representation", StringType(), True),
        StructField("timestamp", DoubleType(), True),
        StructField("Label", StringType(), True),
    ])

    # Áp dụng PreTrainingLayer theo đúng logic pandas.apply bạn gửi
    df_processed = df_raw.mapInPandas(apply_pretrainlayer_map_in_pandas, schema=out_schema)

    # (optional) add source file column để trace
    df_processed = df_processed.withColumn("source_file", lit(input_uri))

    # Ghi ra local /tmp/csv_export/<basename>.csv (overwrite)
    base = os.path.basename(input_uri)
    base_no_ext = os.path.splitext(base)[0]
    temp_dir = os.path.join(LOCAL_EXPORT_DIR, f"tmp_{base_no_ext}")
    final_csv = os.path.join(LOCAL_EXPORT_DIR, f"{base_no_ext}.csv")

    # Dọn temp nếu còn
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Ghi 1 file (coalesce(1)) + header
    df_processed.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"file://{temp_dir}")

    # Move part-*.csv → <basename>.csv (overwrite nếu tồn tại)
    part_files = glob.glob(os.path.join(temp_dir, "part-*.csv"))
    if not part_files:
        logging.warning(f"No CSV writer output for {input_uri}")
    else:
        # nếu final_csv tồn tại → ghi đè
        if os.path.exists(final_csv):
            os.remove(final_csv)
        shutil.move(part_files[0], final_csv)
        logging.info(f"Saved: {final_csv}")

    # cleanup temp dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def process_data():
    spark = SparkSession.builder.appName("AirflowSparkDockerDemo").getOrCreate()

    # Dữ liệu mẫu
    data = [("Alice", "Engineering", 4500),
            ("Bob", "Sales", 3200),
            ("Charlie", "Engineering", 3700)]

    columns = ["name", "department", "salary"]
    df = spark.createDataFrame(data, columns)

    # Thực hiện một phép biến đổi đơn giản: lọc nhân viên Engineering
    df_engineering = df.filter(df.department == "Engineering")

    # Ghi kết quả ra file. Đường dẫn /tmp/data là volume đã mount
    output_path = "/tmp/data/spark_output"
    df_engineering.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

    # Hiển thị kết quả trong log
    print("=== KẾT QUẢ SAU KHI LỌC ===")
    df_engineering.show()

    spark.stop()

def main():
    try:
        spark = (
            SparkSession.builder
            .appName("PretrainLayerBatchExport")
            .getOrCreate()
        )
        jsc = spark.jsc

        # Cấu hình MinIO S3A
        jsc.hadoopConfiguration().set("fs.s3a.endpoint", MINIO_ENDPOINT)
        jsc.hadoopConfiguration().set("fs.s3a.access.key", MINIO_ACCESS_KEY)
        jsc.hadoopConfiguration().set("fs.s3a.secret.key", MINIO_SECRET_KEY)
        jsc.hadoopConfiguration().set("fs.s3a.path.style.access", "true")

        # Liệt kê file trong RAW_DATA_PATH
        input_files = list_s3a_files(spark, RAW_DATA_PATH)  # :contentReference[oaicite:5]{index=5}
        if not input_files:
            logging.warning(f"No files found under {RAW_DATA_PATH}")
        else:
            logging.info(f"Found {len(input_files)} files under {RAW_DATA_PATH}")

        for f in input_files:
            process_one_file(spark, f)

        spark.stop()
        logging.info("Done. All files processed.")

    except AnalysisException as e:
        logging.error(f"Spark operation failed: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Batch job failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
    # process_data()
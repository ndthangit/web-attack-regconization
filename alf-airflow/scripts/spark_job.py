from pyspark.sql import SparkSession
import sys


def main():
    spark = SparkSession.builder.appName("LineCount").getOrCreate()

    try:
        # Đọc file input từ tham số command line hoặc đường dẫn mặc định
        input_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/data/input.txt"
        text_file = spark.read.text(input_path)
        line_count = text_file.count()

        print(f"Line count: {line_count}")
        print("Spark job completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise e
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
from google.cloud import bigquery
from google.cloud import storage


import os

# from storegrowth.configs.config import PROJECT_ROOT_PATH

PROJECT_ROOT_PATH = './'


class GCPConnection:

    def __init__(self, creds_json=None):
        # self.conn = storage.get_connection("dh-darkstores-stg",
        #                               "store-growth-dev@quick-commerce-data.iam.gserviceaccount.com",
        #                                   private_key_path)

        self.client = storage.Client.from_service_account_json('../../cred/credentials.json')


class GCSOperations(GCPConnection):
    def __init__(self, bucket_name):
        super(GCSOperations, self).__init__()
        self.bucket = self.client.bucket(bucket_name)

    def get_content(self, ):
        blobs = self.bucket.list_blobs()
        for blob in blobs:
            print(blob.name)
        return blobs

    def upload_to_bucket(self, dest_dir: str, source_path_to_file:str):
        path_to_file = f"{dest_dir}/{source_path_to_file}"
        try:
            blob = self.bucket.blob(path_to_file)
            blob.upload_from_filename(source_path_to_file)
        except Exception as ex:
            raise ex
        else:
            return {
                'file_name': path_to_file,
                'url': blob.public_url,
                'size': blob.size
            }
        finally:
            pass
            #os.remove(source_path_to_file)

    def download_file_from_gcs(self, source_blob_name, local_path):
        try:
            blob = self.bucket.blob(source_blob_name)
            blob.download_to_filename(local_path)
            logger.info("downloaded successfully.")
            return True
        except Exception as ex:
            logger.exception(ex)
            return False

    def download_dir_from_gcs(self, source_dir_path, local_path):
        blobs = self.bucket.list_blobs(prefix=source_dir_path)  # Get list of files
        for blob in blobs:
            filename = blob.name.replace('/', '_')
            blob.download_to_filename(local_path + filename)


class BigQueryOperations(GCPConnection):
    def __init__(self, project_id):
        super(BigQueryOperations, self).__init__()
        self.bg_client = bigquery.Client(project=project_id)

    def execute_query(self, sql):
        query_job = self.bg_client.query(sql)
        results = query_job.result()
        for row in results:
            print("{} : {} views".format(row.order_id, row.order_date))
        return results

    def get_table_data_all(self, dataset_name, table_name):
        dataset = self.bg_client.dataset(dataset_name)
        table = dataset.table(name=table_name)
        job = self.bg_client.run_async_query('my-job', table)
        job.destination = table
        job.write_disposition = 'WRITE_TRUNCATE'
        job.begin()


if __name__ == "__main__":
    # test_conn = BigQueryOperations("quick-commerce-data")
    # resp = test_conn.execute_query("SELECT * FROM `fulfillment-dwh-production.curated_data_shared_central_dwh.orders` LIMIT 100")

    gcs_conn = GCSOperations("qc-store-growth-expansion-stg")
    resp = gcs_conn.get_content()
    #resp = gcs_conn.upload_to_bucket("existing_dmart_bucket", "nohup.out")
    resp = gcs_conn.download_file_from_gcs( "existing_dmart_bucket/dmart-polygon-SG-drive-time-1-15.json", "dmart-polygon-SG-drive-time-1-15.json")
    print(resp)
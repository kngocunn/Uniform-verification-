import mysql.connector
import numpy as np


class EmbeddingDatabase:

    def __init__(self):

        self.conn = mysql.connector.connect(
            host="localhost",
            port=3307,
            user="root",
            password="kunmuradstsv12",
            database="quanly_taixe"
        )

        self.cursor = self.conn.cursor()

        # load cache
        self.embeddings = self.load_embeddings()


    def load_embeddings(self):

        query = """
        SELECT drv_id, embedding_vector
        FROM embeddings
        """

        self.cursor.execute(query)
        results = self.cursor.fetchall()

        embeddings = {}

        for drv_id, emb_bytes in results:

            embedding = np.frombuffer(
                emb_bytes,
                dtype=np.float32
            )

            embeddings[drv_id] = embedding

        print(f"Loaded {len(embeddings)} embeddings")

        return embeddings


    def get_embedding_by_id(self, drv_id):

        return self.embeddings.get(drv_id)


    def close(self):

        self.cursor.close()
        self.conn.close()
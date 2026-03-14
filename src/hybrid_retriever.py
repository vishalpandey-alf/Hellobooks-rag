from rank_bm25 import BM25Okapi


class HybridRetriever:

    def __init__(self, vector_db):

        self.vector_db = vector_db

        # get documents stored in FAISS
        self.docs = list(vector_db.docstore._dict.values())

        # build corpus for BM25
        corpus = [doc.page_content.split() for doc in self.docs]

        self.bm25 = BM25Okapi(corpus)

    def search(self, query, k=4):

        # Vector search
        vector_results = self.vector_db.similarity_search(query, k=k)

        # BM25 search
        scores = self.bm25.get_scores(query.split())

        ranked = sorted(
            zip(self.docs, scores),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        bm25_docs = [doc[0] for doc in ranked]

        # Combine results
        combined = vector_results + bm25_docs

        unique = {}

        for doc in combined:
            unique[doc.page_content] = doc

        return list(unique.values())
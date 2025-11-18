import asyncio
from pathlib import Path
from multi_doc_chat.src.document_ingestion.data_ingestion import DataIngestor
from multi_doc_chat.logger import GLOBAL_LOGGER as log


class LocalFile:
    def __init__(self, path: Path):
        self.filename = path.name
        self.filepath = path

    @property
    def file(self):
        return open(self.filepath, "rb")


async def test_document_ingestion():
    print("\n===========================================")
    print("   STARTING DOCUMENT INGESTION TEST")
    print("===========================================\n")

    # 1. Initialize DataIngestor (creates session dirs)
    ingestor = DataIngestor(
        temp_base="data",
        faiss_base="faiss_index",
        use_session_dirs=True
    )

    print(f"Session ID: {ingestor.session_id}")
    print(f"Temp directory: {ingestor.temp_dir}")
    print(f"FAISS index directory: {ingestor.faiss_dir}")

    print("\nSTEP 1: Preparing test files...")

    # Ensure test file exists
    sample_file = Path("test/attention_is_all_you_need.pdf")

    if not sample_file.exists():
        print("\n❌ ERROR: test file not found!")
        print("➡️ Place a PDF/TXT/DOCX file inside:  test_docs/sample.pdf")
        return

    print(f"✔ Found test document: {sample_file}")

    print("\nSTEP 2: Running ingestion pipeline...")

    test_file = LocalFile(sample_file)
    # ingestion does: save → load docs → multimodal split → FAISS index build
    retriever = await ingestor.built_retriever(
        paths=[test_file],
        chunk_size=800,
        chunk_overlap=100,
        k=5,
        search_type="mmr",
        fetch_k=35,
        lambda_mult=0.5
    )

    print("\n✔ FAISS retriever successfully created.")
    # print(f"Retriever object: {retriever}")

    print("\nSTEP 3: Verifying FAISS index files...")

    faiss_index_faiss = ingestor.faiss_dir / "index.faiss"
    faiss_index_pkl   = ingestor.faiss_dir / "index.pkl"
    meta_file         = ingestor.faiss_dir / "ingested_meta.json"

    print(f"index.faiss exists? {faiss_index_faiss.exists()}")
    print(f"index.pkl exists?   {faiss_index_pkl.exists()}")
    print(f"metadata file?       {meta_file.exists()}")

    if faiss_index_faiss.exists() and faiss_index_pkl.exists():
        print("\n🎉 SUCCESS: FAISS index was created correctly!")
    else:
        print("\n❌ ERROR: FAISS index missing!")

    print("\nSTEP 4: Printing few meta entries...")

    if meta_file.exists():
        import json
        meta = json.loads(meta_file.read_text())
        rows = list(meta.get("rows", {}).keys())
        print(f"Total ingested chunks: {len(rows)}")
        # print("Sample fingerprints:")
        # for fp in rows[:5]:
        #     print(" -", fp)

    print("\n===========================================")
    print("       INGESTION PIPELINE TEST COMPLETE")
    print("===========================================\n")



if __name__ == "__main__":
    asyncio.run(test_document_ingestion())

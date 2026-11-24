import io
import logging
import time
from pathlib import Path
from docx import Document
from pypdf import PdfReader
from app.aws import client, embed
from app.vector import upsert
from app.config import settings
from app.db import connection, audit
from app.observability import trace, initialize

initialize()

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s [%(name)s] %(message)s')
log=logging.getLogger('atlas.worker')


def parse(blob:bytes, suffix:str):
    if suffix=='.pdf':
        reader=PdfReader(io.BytesIO(blob))
        return [(i+1,page.extract_text() or '') for i,page in enumerate(reader.pages)]
    if suffix=='.docx':
        doc=Document(io.BytesIO(blob))
        return [(1,'\n'.join(p.text for p in doc.paragraphs))]
    return [(1,blob.decode('utf-8'))]



def chunks(text, size=1200, overlap=180):
    text=' '.join(text.split())
    pos=0
    while pos<len(text):
        yield text[pos:pos+size]
        if pos+size>=len(text): break
        pos+=size-overlap




@trace("process_s3_key")
def process_s3_key(key):
    s=settings()
    log.info('Ingestion started: key=%s bucket=%s', key, s.s3_bucket)
    with connection() as conn:
        doc=conn.execute('SELECT * FROM documents WHERE s3_key=%s FOR UPDATE',(key,)).fetchone()
        if not doc:
            log.warning('Ingestion skipped: no document record for key=%s', key)
            return
        if doc['status']=='ready':
            log.info('Ingestion skipped: document already ready document_id=%s key=%s', doc['id'], key)
            return
        if doc['status'] not in ('pending','failed'):
            log.info('Ingestion skipped: document status=%s document_id=%s key=%s', doc['status'], doc['id'], key)
            return
        conn.execute("UPDATE documents SET status='processing' WHERE id=%s",(doc['id'],))
        log.info('Document status set to processing: document_id=%s tenant=%s title=%s version=%s',
                 doc['id'], doc['tenant_id'], doc.get('title'), doc.get('current_version'))
    try:
        obj=client('s3').get_object(Bucket=s.s3_bucket,Key=key)
        content_length=obj.get('ContentLength')
        log.info('Fetched S3 object: document_id=%s key=%s content_length=%s', doc['id'], key, content_length)
        if content_length is not None and content_length>s.max_upload_bytes:
            raise ValueError(f'Upload too large ({content_length} bytes > {s.max_upload_bytes})')
        blob=obj['Body'].read(s.max_upload_bytes+1)
        log.info('Read S3 object body: document_id=%s bytes=%d', doc['id'], len(blob))
        if len(blob)>s.max_upload_bytes: raise ValueError('Upload too large')
        suffix=Path(key).suffix.lower()
        if suffix=='.pdf' and not blob.startswith(b'%PDF'): raise ValueError('Invalid PDF')
        pages=parse(blob,suffix)
        log.info('Parsed document: document_id=%s suffix=%s pages=%d', doc['id'], suffix, len(pages))
        parts=[(page,part) for page,text in pages for part in chunks(text)]
        if not parts: raise ValueError('No extractable text; scanned PDFs need Textract OCR')
        total_chunks=len(parts)
        log.info('Chunked document: document_id=%s chunks=%d', doc['id'], total_chunks)
        vectors=[]
        for i,(page,part) in enumerate(parts):
            log.info('Embedding chunk %d/%d: document_id=%s page=%s chars=%d', i+1, total_chunks, doc['id'], page, len(part))
            vectors.append({'id':f'{doc["id"]}:{doc["current_version"]}:{i}',
                'values':embed(part),'metadata':{'document_id':str(doc['id']),
                'title':doc['title'],'page':page,'text':part,
                'owner_sub':doc['owner_sub'],'allowed_groups':doc['allowed_groups']}})
            # Pace embedding calls to reduce Bedrock throttling on multi-chunk documents.
            if i < len(parts) - 1:
                time.sleep(0.2)
        # Keep batches small; Pinecone metadata must fit the index's limits.
        total_batches=(len(vectors)+49)//50
        for batch_num,i in enumerate(range(0,len(vectors),50), start=1):
            batch=vectors[i:i+50]
            log.info('Upserting vectors batch %d/%d: document_id=%s tenant=%s batch_size=%d',
                     batch_num, total_batches, doc['id'], doc['tenant_id'], len(batch))
            upsert(doc['tenant_id'],batch)
        with connection() as conn:
            conn.execute("UPDATE documents SET status='ready' WHERE id=%s",(doc['id'],))
            audit(conn,doc['owner_sub'],doc['tenant_id'],'ingest_ready',{'document_id':str(doc['id']),'chunks':len(parts)})
        log.info('Ingestion completed: document_id=%s chunks=%d vectors=%d status=ready', doc['id'], len(parts), len(vectors))
    except Exception:
        log.exception('Ingestion failed for key %s',key)
        with connection() as conn:
            conn.execute("UPDATE documents SET status='failed' WHERE id=%s",(doc['id'],))
            log.info('Document status set to failed: document_id=%s key=%s', doc['id'], key)
        raise
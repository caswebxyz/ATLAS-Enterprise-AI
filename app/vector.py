from app.observability import trace
import logging
import time
from functools import lru_cache
from pinecone import Pinecone, ServerlessSpec
from app.config import settings


log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def index():
    """Return an initialized Pinecone index. Safe for repeated requests in one process."""
    s = settings()
    if not s.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing from .env")
    name = s.pinecone_index_name or s.pinecone_index_host
    if not name:
        raise RuntimeError("PINECONE_INDEX_NAME (or PINECONE_INDEX_HOST) is missing from .env")
    pc = Pinecone(api_key=s.pinecone_api_key)
    # Describe rather than list: avoids listing every index and works for existing indexes.
    try:
        info = pc.describe_index(name)
    except Exception as exc:
        # Only create on an actual 404; never hide auth/network failures.
        if getattr(exc, 'status', None) != 404:
            raise
        log.info('Creating Pinecone index %s (dimension=%d, metric=cosine)', name, s.embed_dim)
        try:
            cloud = s.pinecone_cloud or s.pinecone_cloud_name or 'aws'
            pc.create_index(name=name, dimension=s.embed_dim, metric='cosine',
                spec=ServerlessSpec(cloud=cloud, region=s.pinecone_region))
        except Exception as create_exc:
            # Another process may have created the index concurrently.
            if getattr(create_exc, 'status', None) != 409:
                raise
        info = pc.describe_index(name)
    if int(info.dimension) != s.embed_dim:
        raise ValueError(f'Pinecone index {name} dimension={info.dimension}, expected {s.embed_dim}. '
                         'Use a different index name or matching EMBED_DIM.')
    if str(info.metric).lower() != 'cosine':
        raise ValueError(f'Pinecone index {name} must use cosine metric, got {info.metric}')
    deadline = time.monotonic() + s.pinecone_ready_timeout
    while not pc.describe_index(name).status['ready']:
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Pinecone index {name} was not ready within {s.pinecone_ready_timeout}s')
        time.sleep(3)
    log.info('Pinecone index %s ready', name)
    return pc.Index(name)

def namespace(tenant):
    return 'tenant-' + tenant

@trace("upsert")
def upsert(tenant, records):
    try:
        return index().upsert(vectors=records, namespace=namespace(tenant))
    except Exception as exc:
        message = str(exc)
        if 'document schema' in message.lower() or 'documents API' in message:
            raise RuntimeError(
                'Pinecone index is configured with a document schema and requires the '
                'documents/records API for writes. This app upserts precomputed Bedrock '
                'embedding vectors, so use a standard dense vector index (same dimension/metric) '
                'and set PINECONE_INDEX_NAME to it.'
            ) from exc
        raise

@trace("search")
def search(tenant, vector, groups, owner, limit=5):
    acl = {'$or': [{'allowed_groups': {'$in': list(groups)}},
                   {'owner_sub': {'$eq': owner}}]}
    try:
        result = index().query(vector=vector, top_k=limit, include_metadata=True,
                               namespace=namespace(tenant), filter=acl)
    except Exception as exc:
        message = str(exc)
        if 'document schema' in message.lower() or 'documents API' in message:
            raise RuntimeError(
                'Pinecone index is configured with a document schema and requires the '
                'documents/records API for queries. This app queries with precomputed '
                'embedding vectors, so use a standard dense vector index and set '
                'PINECONE_INDEX_NAME to it.'
            ) from exc
        raise
    return [{'id': m.id, 'score': m.score, 'metadata': m.metadata or {}}
            for m in result.matches]

def delete_document(tenant, ids):
    if ids:
        index().delete(ids=ids, namespace=namespace(tenant))
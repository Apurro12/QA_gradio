from rag_system.domain.document_loader import BaseDocumentLoader
from rag_system.infrastructure.offline_document_loader import OfflineDocumentLoader

DOCUMENT_RETRIVAL_CLASS_MAP: dict[str, type[BaseDocumentLoader]] = {
    "offline": OfflineDocumentLoader,
}
